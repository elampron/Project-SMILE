"""Tool for searching cognitive memories in the knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class SearchMemoriesInput(BaseModel):
    """Schema for searching memories in the knowledge base."""
    
    query: str = Field(description="The search query to find relevant memories")
    importance: Optional[int] = Field(None, description="Optional filter for memory importance (1-5)")
    limit: Optional[int] = Field(3, description="Maximum number of memories to return")

@tool
def search_memories(
    query: str,
    importance: Optional[int] = None,
    limit: Optional[int] = 3
) -> str:
    """Search for relevant memories in the knowledge base.
    
    Use this tool when you need to:
    - Find historical context about a topic
    - Search for important memories
    - Gather information about past events
    - Find connections between memories
    
    Parameters:
        query (str): The search query to find relevant memories
        importance (Optional[int]): Optional filter for memory importance (1-5)
        limit (Optional[int]): Maximum number of memories to return
        
    Return Value:
        str: Formatted string containing the search results
    """
    try:
        # Generate embedding for the query
        embeddings_service = EmbeddingsService()
        query_embedding = embeddings_service.generate_embedding(query)
        
        # Build the search query
        cypher_query = """
        MATCH (m:Memory)
        WHERE ($importance IS NULL OR m.importance = $importance)
        WITH m, vector.similarity(m.embedding, $query_embedding) AS score
        WHERE score >= 0.7
        
        // Get related entities
        OPTIONAL MATCH (m)-[r:MENTIONS]->(e)
        WITH m, score, collect(DISTINCT e.name) as entities
        
        RETURN m {
            .*, 
            entities: entities
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
                
                # Parse metadata from JSON string if it exists
                metadata = memory.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Format the memory information
                memory_info = [
                    f"\n{i}. Relevance Score: {score:.2f}",
                    f"Content: {memory.get('content')}",
                    f"Importance: {memory.get('importance', 'Unknown')}",
                    f"Created: {memory.get('created_at', 'Unknown')}"
                ]
                
                # Add entities if available
                entities = memory.get('entities', [])
                if entities:
                    memory_info.append("Related Entities:")
                    for entity in entities:
                        memory_info.append(f"  - {entity}")
                
                # Add relevant metadata if it exists
                if metadata:
                    memory_info.append("Metadata:")
                    for key, value in metadata.items():
                        memory_info.append(f"  - {key}: {value}")
                
                formatted_results.append("\n".join(memory_info))
            
            return "\n".join(formatted_results)
            
    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        return f"An error occurred while searching memories: {str(e)}" 