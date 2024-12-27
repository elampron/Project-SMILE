"""Tool for searching memories in SMILE's knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.services.neo4j import driver, vector_similarity_search, get_related_entities
from app.services.embeddings import EmbeddingsService

# Configure logging
logger = logging.getLogger(__name__)

class SearchMemoriesInput(BaseModel):
    """Input for searching memories."""
    query: str = Field(
        description="The search query to find relevant memories. This can be a question, topic, or description."
    )
    importance: Optional[int] = Field(
        None,
        description="Optional filter for minimum importance level (1-5)",
        ge=1,
        le=5
    )
    limit: Optional[int] = Field(
        3,
        description="Maximum number of memories to return. Defaults to 3."
    )

class SearchMemoriesTool(BaseTool):
    """Tool for searching cognitive memories in SMILE's knowledge base."""
    
    name: ClassVar[str] = "search_memories"
    description: ClassVar[str] = """
    Search for relevant memories in SMILE's knowledge base.
    Use this tool when you need to:
    - Find memories related to a specific topic
    - Search for past experiences or learnings
    - Gather historical context about a subject
    - Find important memories about a topic
    
    The tool will return the most relevant memories based on semantic search.
    """
    
    args_schema: ClassVar[type[BaseModel]] = SearchMemoriesInput
    
    # Add model field for service
    embeddings_service: EmbeddingsService = Field(default_factory=lambda: EmbeddingsService())
    
    def __init__(self, **data):
        """Initialize the tool with necessary services."""
        super().__init__(**data)
    
    def _format_memory(self, memory: Dict) -> str:
        """Format a single memory result."""
        # Parse metadata from JSON string if it exists
        metadata = memory.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        # Format the memory information
        memory_info = [
            f"Memory: {memory.get('content')}",
            f"Importance: {'⭐' * memory.get('importance', 1)}",
            f"Created: {memory.get('created_at', 'Unknown')}",
            f"Context: {memory.get('context', 'No context available')}"
        ]
        
        # Add related entities if available
        related_entities = memory.get('related_entities', [])
        if related_entities:
            memory_info.append("Related Entities:")
            for entity in related_entities:
                memory_info.append(f"  - {entity}")
        
        # Add relevant metadata if it exists
        if metadata:
            memory_info.append("Metadata:")
            for key, value in metadata.items():
                memory_info.append(f"  - {key}: {value}")
        
        return "\n".join(memory_info)
    
    def _run(self, query: str, importance: Optional[int] = None, limit: int = 3) -> str:
        """
        Execute the memory search.
        
        Args:
            query: Search query text
            importance: Optional minimum importance level filter (1-5)
            limit: Maximum number of results to return
            
        Returns:
            str: Formatted string of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_service.generate_embedding(query)
            
            # Build importance filter if specified
            additional_filters = None
            if importance is not None:
                additional_filters = f"node.importance >= {importance}"
            
            # Execute the search
            with driver.session() as session:
                results = vector_similarity_search(
                    session=session,
                    index_name="memory_embeddings",
                    query_vector=query_embedding,
                    k=limit,
                    min_score=0.7,
                    additional_filters=additional_filters
                )
                
                if not results:
                    return "No relevant memories found."
                
                # Get related entities for each memory
                for result in results:
                    memory_id = result["node"].get("id")
                    if memory_id:
                        related = get_related_entities(
                            session=session,
                            node_id=memory_id
                        )
                        result["node"]["related_entities"] = [
                            entity.get("name") for entity in related
                        ]
                
                # Format results
                formatted_results = ["Here are the most relevant memories:"]
                for i, result in enumerate(results, 1):
                    memory = result["node"]
                    score = result["score"]
                    formatted_memory = self._format_memory(memory)
                    formatted_results.append(f"\n{i}. Relevance Score: {score:.2f}")
                    formatted_results.append(formatted_memory)
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return f"An error occurred while searching memories: {str(e)}"

if __name__ == "__main__":
    """
    Test script for SearchMemoriesTool
    """
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize tool
        search_tool = SearchMemoriesTool()
        
        # Create test data
        with driver.session() as session:
            # Create test memory
            session.run("""
                CREATE (m:CognitiveMemory)
                SET m = $properties
                """,
                properties={
                    "id": "test-memory-1",
                    "content": "Client meeting discussed AI project requirements",
                    "importance": 4,
                    "created_at": "2024-01-01",
                    "context": "Project Planning",
                    "is_test": True,
                    "embedding": search_tool.embeddings_service.generate_embedding(
                        "Client meeting discussed AI project requirements"
                    )
                }
            )
            
            # Create test entity
            session.run("""
                CREATE (e:Person)
                SET e = $properties
                """,
                properties={
                    "id": "test-person-1",
                    "name": "John Smith",
                    "role": "Client",
                    "is_test": True
                }
            )
            
            # Create relationship
            session.run("""
                MATCH (m:CognitiveMemory {id: $memory_id})
                MATCH (e:Person {id: $entity_id})
                CREATE (m)-[:INVOLVES]->(e)
                """,
                memory_id="test-memory-1",
                entity_id="test-person-1"
            )
            
            logger.info("Created test data")
            
            # Test queries
            test_queries = [
                {
                    "query": "What do we know about AI projects?",
                    "importance": None
                },
                {
                    "query": "Find important client discussions",
                    "importance": 4
                }
            ]
            
            # Run test queries
            for test in test_queries:
                logger.info("\n" + "=" * 50)
                logger.info(f"Testing query: {test['query']}")
                logger.info(f"Importance filter: {test['importance']}")
                
                try:
                    result = search_tool._run(
                        query=test["query"],
                        importance=test["importance"]
                    )
                    logger.info("\nResults:")
                    logger.info(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{test['query']}': {str(e)}")
            
            # Clean up test data
            session.run("MATCH (n) WHERE n.is_test = true DETACH DELETE n")
            logger.info("\nCleaned up test data")
            
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
    finally:
        logger.info("Tests completed") 