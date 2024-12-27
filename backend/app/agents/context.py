"""
Context management module for SMILE agent.

This module is responsible for gathering, organizing, and formatting different types of contextual 
information to be injected into the prompt template. It acts as a central place for context 
management with methods to handle different types of context (preferences, entities, facts, 
summaries) and format them in a way that's optimal for LLM consumption.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID
from neo4j import GraphDatabase, Session
from app.services.neo4j import driver, vector_similarity_search
from app.services.embeddings import EmbeddingsService
from app.services.knowledge_search import KnowledgeSearchService

from app.models.memory import (
    Preference, PersonEntity, OrganizationEntity, 
    ConversationSummary, CognitiveMemory
)
from app.configs.settings import settings

# Initialize logger
logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context preparation and formatting for the SMILE agent.
    
    This class is responsible for:
    1. Gathering context from different sources (Neo4j, state, etc.)
    2. Filtering and organizing context based on relevance and importance
    3. Formatting context in a way that's optimal for LLM consumption
    
    Attributes:
        driver: Neo4j driver instance for database operations
        knowledge_service: Service for advanced knowledge search
        embeddings_service: Service for generating embeddings
    """
    
    def __init__(self, driver: GraphDatabase.driver):
        """
        Initialize the ContextManager.
        
        Args:
            driver: Neo4j driver instance for database operations
        """
        self.driver = driver
        self.knowledge_service = KnowledgeSearchService(driver)
        self.embeddings_service = EmbeddingsService()
        
    async def get_formatted_context(self, user_input: str) -> str:
        """
        Master method to gather and format all context for the LLM.
        
        Args:
            user_input: The user's current question or comment
            
        Returns:
            str: Formatted context string ready for LLM consumption
            
        Logs:
            DEBUG: Context gathering process details
            ERROR: When context gathering fails
        """
        try:
            logger.debug(f"Starting context gathering for input: {user_input}")
            
            # Get important preferences (high priority items that should always be included)
            important_preferences = self._get_important_preferences()
            logger.debug(f"Found {len(important_preferences)} important preferences")
            
            # Use knowledge search for all other context
            knowledge_results = self.knowledge_service.search_knowledge(
                query=user_input,
                filters={
                    "node_types": [
                        "Preference",
                        "Summary",
                        "Person",
                        "Organization",
                        "Document",
                        "CognitiveMemory"
                    ]
                },
                limit=10,
                min_score=0.7
            )
            logger.debug(f"Knowledge search returned {len(knowledge_results)} results")
            
            # Organize results by type
            organized_results = self._organize_knowledge_results(knowledge_results)
            logger.debug(f"Organized results by type: {organized_results.keys()}")
            
            # Format the context
            context_parts = []
            
            if organized_results.get('Document'):
                context_parts.append("RELEVANT DOCUMENTS:")
                context_parts.extend(self._format_documents(organized_results['Document']))
                logger.debug(f"Added {len(organized_results['Document'])} document contexts")
            
            if important_preferences:
                context_parts.append("\nIMPORTANT PREFERENCES:")
                context_parts.extend(self._format_preferences(important_preferences))
                logger.debug(f"Added {len(important_preferences)} important preference contexts")
            
            if organized_results.get('Preference'):
                context_parts.append("\nRELEVANT PREFERENCES:")
                context_parts.extend(self._format_preferences(organized_results['Preference']))
                logger.debug(f"Added {len(organized_results['Preference'])} relevant preference contexts")
            
            if organized_results.get('Person') or organized_results.get('Organization'):
                context_parts.append("\nRELEVANT ENTITIES:")
                if organized_results.get('Person'):
                    context_parts.extend(self._format_entities(organized_results['Person']))
                    logger.debug(f"Added {len(organized_results['Person'])} person contexts")
                if organized_results.get('Organization'):
                    context_parts.extend(self._format_entities(organized_results['Organization']))
                    logger.debug(f"Added {len(organized_results['Organization'])} organization contexts")
            
            if organized_results.get('CognitiveMemory'):
                context_parts.append("\nRELEVANT MEMORIES:")
                context_parts.extend(self._format_memories(organized_results['CognitiveMemory']))
                logger.debug(f"Added {len(organized_results['CognitiveMemory'])} memory contexts")
            
            formatted_context = "\n".join(context_parts)
            logger.debug(f"Final formatted context length: {len(formatted_context)}")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error gathering context: {str(e)}")
            return ""
    
    def _organize_knowledge_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize knowledge search results by node type.
        
        Args:
            results: List of search results from knowledge service
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Results organized by node type
        """
        organized = {}
        for result in results:
            node_type = result.get('node_label', 'Unknown')
            if node_type not in organized:
                organized[node_type] = []
            # Extract the node data from the result structure
            node_data = result.get('node', {})
            # Add score to node data for potential use in formatting
            node_data['score'] = result.get('score', 0.0)
            organized[node_type].append(node_data)
        return organized
    
    def _get_important_preferences(self) -> List[Preference]:
        """
        Get all preferences with high importance (>= 4).
        These are preferences that should always be considered regardless of the current context.
        
        Returns:
            List[Preference]: List of high-importance preferences
        """
        query = """
        MATCH (p:Preference)
        WHERE p.importance > 4
        RETURN p {
            .*, 
            embedding: null
        } as p
        ORDER BY p.importance DESC, p.created_at DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [self._neo4j_to_preference(record["p"]) for record in result]
    
    def _get_relevant_preferences(self, user_input: str) -> List[Preference]:
        """
        Get preferences relevant to the user's input using semantic search.
        This method finds preferences whose embeddings are semantically similar to the user's input.
        
        Args:
            user_input: The user's current question or comment
            
        Returns:
            List[Preference]: List of semantically relevant preferences
            
        Note:
            Preferences are ordered by semantic similarity and filtered to avoid duplicates
            with important preferences.
        """
        # First get important preferences to avoid duplicates
        important_preferences = {str(p.id) for p in self._get_important_preferences()}
        
        try:
            # Generate embedding for user input
            query_embedding = self.embeddings_service.generate_embedding(user_input)
            
            # Use similarity search from embeddings service
            results = self.embeddings_service.similarity_search(
                query_embedding=query_embedding,
                node_label="Preference",
                limit=5,
                min_score=0.7
            )
            
            # Filter out important preferences and convert to Preference objects
            return [
                self._neo4j_to_preference(result) 
                for result in results 
                if str(result.get('id')) not in important_preferences 
                and result.get('importance', 0) < 4
            ]
                
        except Exception as e:
            logger.error(f"Error getting relevant preferences: {str(e)}")
            # Fallback to getting recent preferences if semantic search fails
            fallback_query = """
            MATCH (p:Preference)
            WHERE NOT p.id IN $exclude_ids
                AND p.importance < 5
            RETURN p {
                .*, 
                embedding: null
            } as p
            ORDER BY p.created_at DESC
            LIMIT 5
            """
            with self.driver.session() as session:
                result = session.run(fallback_query, exclude_ids=list(important_preferences))
                return [self._neo4j_to_preference(record["p"]) for record in result]
    
    def _get_relevant_entities(self, user_input: str) -> List[Dict]:
        """Get entities relevant to the user's input using semantic search."""
        # TODO: Implement semantic search for entities
        return []
    
    def _get_important_facts(self) -> List[Dict]:
        """Get important facts that should always be shown."""
        # TODO: Implement once fact extraction is added
        return []
    
    def _get_relevant_facts(self, user_input: str) -> List[Dict]:
        """Get facts relevant to the user's input using semantic search."""
        # TODO: Implement once fact extraction is added
        return []
    
    def _get_relevant_memories(self, user_input: str) -> List[CognitiveMemory]:
        """
        Get cognitive memories relevant to the user's input using semantic search.
        
        Args:
            user_input: The user's current input to find relevant memories for
            
        Returns:
            List[CognitiveMemory]: List of relevant cognitive memories
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_service.generate_embedding(user_input)
            
            with self.driver.session() as session:
                # Use vector search to find relevant memories
                results = vector_similarity_search(
                    session=session,
                    index_name="memory_embeddings",
                    query_vector=query_embedding,
                    k=5,
                    min_score=0.7
                )
                
                return [
                    self._neo4j_to_cognitive_memory(result["node"]) 
                    for result in results
                ]
                
        except Exception as e:
            logger.error(f"Error getting relevant memories: {str(e)}")
            # Fallback to getting recent important memories if semantic search fails
            query = """
            MATCH (m:CognitiveMemory)
            WHERE m.importance >= 3
                RETURN m {
                    .*, 
                    embedding: null
                } as m
            ORDER BY m.created_at DESC
            LIMIT 5
            """
            with self.driver.session() as session:
                result = session.run(query)
                return [self._neo4j_to_cognitive_memory(record["m"]) for record in result]
    
    def _get_recent_summaries(self) -> List[ConversationSummary]:
        """
        Get the most recent conversation summaries from the last 24 hours.
        
        Returns:
            List[ConversationSummary]: List of recent conversation summaries
        """
        query = """
        MATCH (s:Summary)
        WHERE s.created_at >= datetime() - duration('P1D')  // Within last 24 hours
        RETURN s {
            .*, 
            embedding: null
        } as s  
        ORDER BY s.created_at DESC
        LIMIT 3
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [self._neo4j_to_conversation_summary(record["s"]) for record in result]
    
    def _get_relevant_summaries(self, user_input: str) -> List[ConversationSummary]:
        """
        Get summaries relevant to the user's input using semantic search.
        
        Args:
            user_input: The user's current input to find relevant summaries for
            
        Returns:
            List[ConversationSummary]: List of relevant conversation summaries
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_service.generate_embedding(user_input)
            
            with self.driver.session() as session:
                # Use vector search to find relevant summaries
                results = vector_similarity_search(
                    session=session,
                    index_name="summary_embeddings",
                    query_vector=query_embedding,
                    k=5,
                    min_score=0.7
                )
                
                return [
                    self._neo4j_to_conversation_summary(result["node"]) 
                    for result in results
                ]
                
        except Exception as e:
            logger.error(f"Error getting relevant summaries: {str(e)}")
            # Fallback to getting recent summaries if semantic search fails
            query = """
            MATCH (s:Summary)
            RETURN s {
                .*, 
                embedding: null
            } as s
            ORDER BY s.created_at DESC
            LIMIT 5
            """
            with self.driver.session() as session:
                result = session.run(query)
                return [self._neo4j_to_conversation_summary(record["s"]) for record in result]
    
    def _format_preferences(self, preferences: List[Preference]) -> List[str]:
        """Format preferences into human-readable strings."""
        formatted = []
        for pref in preferences:
            try:
                # Convert details from string to dict if needed
                details = pref.get('details', {})
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except:
                        details = {'value': details}
                
                details_str = ", ".join(f"{k}: {v}" for k, v in details.items())
                formatted.append(
                    f"- {pref.get('preference_type', 'Unknown')}: {details_str} "
                    f"(Importance: {pref.get('importance', 1)}, "
                    f"Score: {pref.get('score', 0.0):.2f})"
                )
            except Exception as e:
                logger.error(f"Error formatting preference: {str(e)}")
                continue
        return formatted
    
    def _format_entities(self, entities: List[Dict]) -> List[str]:
        """Format entities into human-readable strings."""
        formatted = []
        for entity in entities:
            try:
                formatted.append(
                    f"- {entity.get('name', 'Unknown')} "
                    f"({entity.get('type', 'Unknown')}, "
                    f"Score: {entity.get('score', 0.0):.2f})"
                )
            except Exception as e:
                logger.error(f"Error formatting entity: {str(e)}")
                continue
        return formatted
    
    def _format_facts(self, facts: List[Dict]) -> List[str]:
        """Format facts into human-readable strings."""
        return [f"- {fact['content']}" for fact in facts]
    
    def _format_memories(self, memories: List[CognitiveMemory]) -> List[str]:
        """
        Format cognitive memories into human-readable strings.
        
        Args:
            memories: List of CognitiveMemory objects to format
            
        Returns:
            List[str]: Formatted memory strings
        """
        formatted = []
        for memory in memories:
            try:
                created_at = memory.get('created_at')
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()
                elif isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_at = datetime.utcnow()

                formatted.append(
                    f"- {memory.get('content', 'No content')} "
                    f"(Source: {memory.get('source', 'Unknown')}, "
                    f"Importance: {memory.get('importance', 1)}, "
                    f"Score: {memory.get('score', 0.0):.2f}, "
                    f"Created: {created_at.strftime('%Y-%m-%d %H:%M')})"
                )
            except Exception as e:
                logger.error(f"Error formatting memory: {str(e)}")
                continue
        return formatted
    
    def _format_summaries(self, summaries: List[ConversationSummary]) -> List[str]:
        """
        Format conversation summaries into human-readable strings.
        Shows the actual conversation time period instead of creation time for better context.
        
        Args:
            summaries: List of ConversationSummary objects to format
            
        Returns:
            List[str]: Formatted summary strings with conversation time period
        """
        formatted = []
        for summary in summaries:
            topics_str = ", ".join(summary.topics) if summary.topics else "No topics"
            time_period = ""
            if summary.start_time and summary.end_time:
                start_str = summary.start_time.strftime('%Y-%m-%d %H:%M')
                end_str = summary.end_time.strftime('%Y-%m-%d %H:%M')
                if start_str[:10] == end_str[:10]:  # Same day
                    time_period = f"Conversation on {start_str[:10]} from {start_str[11:]} to {end_str[11:]}"
                else:
                    time_period = f"Conversation from {start_str} to {end_str}"
            else:
                time_period = "Time period unknown"
                
            formatted.append(
                f"- {summary.content}\n"
                f"  Topics: {topics_str}\n"
                f"  {time_period}"
            )
        return formatted
    
    def _neo4j_to_preference(self, neo4j_node: Any) -> Preference:
        """
        Convert a Neo4j node to a Preference object with proper type conversions.
        
        Args:
            neo4j_node: Raw Neo4j node data
            
        Returns:
            Preference: Properly formatted preference object
            
        Logs:
            DEBUG: Node conversion details
            ERROR: Conversion failures
        """
        try:
            logger.debug(f"Converting Neo4j node: {neo4j_node}")
            node_dict = dict(neo4j_node)
            
            # Convert details from string to dict if needed
            details = node_dict.get('details', {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
                
            # Convert datetime objects
            created_at = node_dict.get('created_at')
            if hasattr(created_at, 'to_native'):  # Neo4j DateTime conversion
                created_at = created_at.to_native()
            else:
                created_at = datetime.now()  # Default to current time if not set
                
            updated_at = node_dict.get('updated_at')
            if hasattr(updated_at, 'to_native'):  # Neo4j DateTime conversion
                updated_at = updated_at.to_native()
            else:
                updated_at = created_at  # Default to created_at if not set
                
            # Generate a random UUID if person_id is None
            import uuid
            person_id = str(node_dict.get('person_id')) if node_dict.get('person_id') else str(uuid.uuid4())
            
            return Preference(
                id=node_dict.get('id'),
                person_id=person_id,
                preference_type=node_dict.get('preference_type'),
                importance=node_dict.get('importance', 1),
                details=details,
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Error converting Neo4j node to Preference: {str(e)}")
            raise
            
    def _neo4j_to_cognitive_memory(self, neo4j_node: Any) -> CognitiveMemory:
        """
        Convert a Neo4j node to a CognitiveMemory object.
        
        Args:
            neo4j_node: Raw Neo4j node data
            
        Returns:
            CognitiveMemory: Properly formatted memory object
        """
        try:
            logger.debug(f"Converting Neo4j node to CognitiveMemory: {neo4j_node}")
            node_dict = dict(neo4j_node)
            
            # Convert datetime objects
            created_at = node_dict.get('created_at')
            if hasattr(created_at, 'to_native'):
                created_at = created_at.to_native()
            else:
                created_at = datetime.utcnow()
                
            return CognitiveMemory(
                id=node_dict.get('id'),
                content=node_dict.get('content'),
                source=node_dict.get('source'),
                importance=node_dict.get('importance', 1),
                context=node_dict.get('context', {}),
                created_at=created_at,
                embedding=node_dict.get('embedding')
            )
        except Exception as e:
            logger.error(f"Error converting Neo4j node to CognitiveMemory: {str(e)}")
            raise
            
    def _neo4j_to_conversation_summary(self, neo4j_node: Any) -> ConversationSummary:
        """
        Convert a Neo4j node to a ConversationSummary object.
        
        Args:
            neo4j_node: Raw Neo4j node data
            
        Returns:
            ConversationSummary: Properly formatted summary object
        """
        try:
            logger.debug(f"Converting Neo4j node to ConversationSummary: {neo4j_node}")
            node_dict = dict(neo4j_node)
            
            # Convert datetime objects
            created_at = node_dict.get('created_at')
            if hasattr(created_at, 'to_native'):
                created_at = created_at.to_native()
            else:
                created_at = datetime.utcnow()
                
            start_time = node_dict.get('start_time')
            if hasattr(start_time, 'to_native'):
                start_time = start_time.to_native()
                
            end_time = node_dict.get('end_time')
            if hasattr(end_time, 'to_native'):
                end_time = end_time.to_native()
                
            return ConversationSummary(
                id=node_dict.get('id'),
                content=node_dict.get('content'),
                topics=node_dict.get('topics', []),
                action_items=node_dict.get('action_items', []),
                participants=node_dict.get('participants', []),
                sentiments=node_dict.get('sentiments', {}),
                location=node_dict.get('location'),
                events=node_dict.get('events', []),
                message_ids=node_dict.get('message_ids', []),
                created_at=created_at,
                start_time=start_time,
                end_time=end_time,
                embedding=node_dict.get('embedding')
            )
        except Exception as e:
            logger.error(f"Error converting Neo4j node to ConversationSummary: {str(e)}")
            raise
    
    def _get_relevant_documents(self, user_input: str) -> List[Dict]:
        """
        Get documents relevant to the user's input using semantic search.
        
        Args:
            user_input: The user's current question or comment
            
        Returns:
            List[Dict]: List of relevant documents with their properties
            
        Note:
            Documents are ordered by semantic similarity and include:
            - Basic metadata (name, type, summary)
            - Topics and entities
            - File path for reference
        """
        try:
            # Generate embedding for user input
            query_embedding = self.embeddings_service.generate_embedding(user_input)
            
            # Use similarity search from embeddings service
            results = self.embeddings_service.similarity_search(
                query_embedding=query_embedding,
                node_label="Document",
                limit=3,  # Limit to top 3 most relevant documents
                min_score=0.7
            )
            
            # Convert results to proper format
            documents = []
            for result in results:
                # Parse metadata from JSON string
                metadata = json.loads(result.get('metadata', '{}'))
                
                doc = {
                    'name': result.get('name'),
                    'doc_type': result.get('doc_type'),
                    'summary': result.get('summary'),
                    'topics': result.get('topics', []),
                    'entities': result.get('entities', []),
                    'file_path': result.get('file_path'),
                    'metadata': metadata
                }
                documents.append(doc)
            
            return documents
                
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            # Fallback to getting recent documents if semantic search fails
            fallback_query = """
            MATCH (d:Document)
            RETURN d {
                .*, 
                embedding: null
            } as d
            ORDER BY d.created_at DESC
            LIMIT 3
            """
            with self.driver.session() as session:
                result = session.run(fallback_query)
                documents = []
                for record in result:
                    doc = record["d"]
                    metadata = json.loads(doc.get('metadata', '{}'))
                    documents.append({
                        'name': doc.get('name'),
                        'doc_type': doc.get('doc_type'),
                        'summary': doc.get('summary'),
                        'topics': doc.get('topics', []),
                        'entities': doc.get('entities', []),
                        'file_path': doc.get('file_path'),
                        'metadata': metadata
                    })
                return documents

    def _format_documents(self, documents: List[Dict]) -> List[str]:
        """Format documents for context inclusion."""
        formatted = []
        for doc in documents:
            doc_info = [
                f"- {doc['name']} ({doc['doc_type']})",
                f"  Summary: {doc['summary'] if doc['summary'] else 'No summary available'}",
                f"  Topics: {', '.join(doc['topics']) if doc['topics'] else 'None'}",
                f"  Entities: {', '.join(doc['entities']) if doc['entities'] else 'None'}",
                f"  Path: {doc['file_path']}"
            ]
            formatted.append("\n".join(doc_info))
        return formatted

            
if __name__ == "__main__":
    """Test script for the ContextManager"""
    import logging
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    async def main():
        try:
            logger.info("Starting context manager test...")
            context_manager = ContextManager(driver)
            logger.info("ContextManager initialized successfully")
            
            # Test inputs
            test_inputs = [
                "When I talk about Thinkmax, what comes to mind?",
                "Do you remember what's next on our Smiles project?"
            ]
            
            # Test each input
            for test_input in test_inputs:
                logger.info(f"\nTesting with input: {test_input}")
                try:
                    logger.info("Getting formatted context...")
                    formatted_context = await context_manager.get_formatted_context(test_input)
                    logger.info("Successfully got formatted context")
                    logger.info("Formatted Context:")
                    logger.info("-" * 50)
                    logger.info(formatted_context)
                    logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error processing input '{test_input}': {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during testing: {str(e)}")
            raise
        finally:
            logger.info("Closing Neo4j driver...")
            driver.close()
            logger.info("Neo4j driver closed")
    
    # Run the async main function
    logger.info("Starting test script...")
    asyncio.run(main())
