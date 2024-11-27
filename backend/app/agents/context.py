"""
Context management module for SMILE agent.

This module is responsible for gathering, organizing, and formatting different types of contextual 
information to be injected into the prompt template. It acts as a central place for context 
management with methods to handle different types of context (preferences, entities, facts, 
summaries) and format them in a way that's optimal for LLM consumption.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID
from neo4j import GraphDatabase, Session
from app.services.neo4j import get_similar_memories

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
    """
    
    def __init__(self, driver: GraphDatabase.driver):
        """
        Initialize the ContextManager.
        
        Args:
            driver: Neo4j driver instance for database operations
        """
        self.driver = driver
        
    def get_formatted_context(self, user_input: str) -> str:
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
            
            # Gather different types of context
            important_preferences = self._get_important_preferences()
            relevant_preferences = self._get_relevant_preferences(user_input)
            relevant_entities = self._get_relevant_entities(user_input)
            relevant_summaries = self._get_relevant_summaries(user_input)
            relevant_memories = self._get_relevant_memories(user_input)
            recent_summaries = self._get_recent_summaries()
            
            # Format the context
            context_parts = []
            
            if important_preferences:
                context_parts.append("IMPORTANT PREFERENCES:")
                context_parts.extend(self._format_preferences(important_preferences))
            
            if relevant_preferences:
                context_parts.append("\nRELEVANT PREFERENCES:")
                context_parts.extend(self._format_preferences(relevant_preferences))
            
            if relevant_entities:
                context_parts.append("\nRELEVANT ENTITIES:")
                context_parts.extend(self._format_entities(relevant_entities))

            
            if relevant_memories:
                context_parts.append("\nRELEVANT MEMORIES:")
                context_parts.extend(relevant_memories)
            
            if recent_summaries:
                context_parts.append("\nRECENT CONVERSATION SUMMARIES:")
                context_parts.extend(self._format_summaries(recent_summaries))
            
            if relevant_summaries:
                context_parts.append("\nRELEVANT SUMMARIES:")
                context_parts.extend(self._format_summaries(relevant_summaries))
            
            formatted_context = "\n".join(context_parts)
            logger.debug(f"Formatted context: {formatted_context}")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error gathering context: {str(e)}")
            return ""
    
    def _get_important_preferences(self) -> List[Preference]:
        """Get all preferences with importance = 5."""
        query = """
        MATCH (p:Preference)
        WHERE p.importance = 5
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [self._neo4j_to_preference(record["p"]) for record in result]
    
    def _get_relevant_preferences(self, user_input: str) -> List[Preference]:
        """Get preferences relevant to the user's input using semantic search."""
        # TODO: Implement semantic search for preferences
        return []
    
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
            with self.driver.session() as session:
                # Use execute_read for read-only operations
                return session.execute_read(
                    lambda tx: get_similar_memories(tx, text=user_input)
                )
        except Exception as e:
            logger.error(f"Error getting relevant memories: {str(e)}")
            # Fallback to getting recent important memories if semantic search fails
            query = """
            MATCH (m:CognitiveMemory)
            WHERE m.importance >= 3
            RETURN m
            ORDER BY m.created_at DESC
            LIMIT 5
            """
            with self.driver.session() as session:
                result = session.run(query)
                return [self._neo4j_to_cognitive_memory(record["m"]) for record in result]
    
    def _get_recent_summaries(self) -> List[ConversationSummary]:
        """
        Get the most recent conversation summaries.
        
        Returns:
            List[ConversationSummary]: List of recent conversation summaries
        """
        query = """
        MATCH (s:ConversationSummary)
        WHERE datetime() - s.created_at <= duration('P1D')  // Within last 24 hours
        RETURN s
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
        # TODO: Implement semantic search for summaries
        query = """
        MATCH (s:ConversationSummary)
        RETURN s
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
            details_str = ", ".join(f"{k}: {v}" for k, v in pref.details.items())
            formatted.append(f"- {pref.preference_type}: {details_str} (Importance: {pref.importance})")
        return formatted
    
    def _format_entities(self, entities: List[Dict]) -> List[str]:
        """Format entities into human-readable strings."""
        formatted = []
        for entity in entities:
            formatted.append(f"- {entity['name']} ({entity['type']})")
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
            formatted.append(
                f"- {memory.content} "
                f"(Source: {memory.source}, "
                f"Importance: {memory.importance}, "
                f"Created: {memory.created_at.strftime('%Y-%m-%d %H:%M')})"
            )
        return formatted
    
    def _format_summaries(self, summaries: List[ConversationSummary]) -> List[str]:
        """
        Format conversation summaries into human-readable strings.
        
        Args:
            summaries: List of ConversationSummary objects to format
            
        Returns:
            List[str]: Formatted summary strings
        """
        formatted = []
        for summary in summaries:
            topics_str = ", ".join(summary.topics) if summary.topics else "No topics"
            formatted.append(
                f"- {summary.content}\n"
                f"  Topics: {topics_str}\n"
                f"  Time: {summary.created_at.strftime('%Y-%m-%d %H:%M')}"
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