"""Tool for searching memories in SMILE's knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.services.neo4j import driver
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
    embeddings_service: EmbeddingsService = Field(default_factory=lambda: EmbeddingsService(driver))
    
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
            
            # Build the search query
            cypher_query = """
            MATCH (m:CognitiveMemory)
            WHERE ($importance IS NULL OR m.importance >= $importance)
            WITH m, vector.similarity(m.embedding, $query_embedding) AS score
            WHERE score >= 0.7
            
            // Get related entities
            OPTIONAL MATCH (m)-[r]-(related)
            WITH m, score, collect(DISTINCT related.name) as related_entities
            
            RETURN m {
                .*, 
                related_entities: related_entities
            } as memory, score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            # Execute the search
            with driver.session() as session:
                result = session.run(
                    cypher_query,
                    query_embedding=query_embedding,
                    importance=importance,
                    limit=limit
                )
                
                memories = list(result)
                
                if not memories:
                    return "No relevant memories found."
                
                # Format results
                formatted_results = ["Here are the most relevant memories:"]
                for i, record in enumerate(memories, 1):
                    memory = record["memory"]
                    score = record["score"]
                    formatted_memory = self._format_memory(memory)
                    formatted_results.append(f"\n{i}. Relevance Score: {score:.2f}")
                    formatted_results.append(formatted_memory)
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return f"An error occurred while searching memories: {str(e)}" 