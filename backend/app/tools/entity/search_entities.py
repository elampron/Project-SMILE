"""Tool for searching entities in the knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class SearchEntitiesInput(BaseModel):
    """Schema for searching entities in the knowledge base."""
    
    query: str = Field(description="The search query to find relevant entities")
    entity_type: Optional[str] = Field(None, description="Optional filter for entity type ('Person' or 'Organization')")
    limit: Optional[int] = Field(3, description="Maximum number of entities to return")

@tool
def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: Optional[int] = 3
) -> str:
    """Search for relevant entities in the knowledge base.
    
    Use this tool when you need to:
    - Find people or organizations related to a topic
    - Search for specific individuals or companies
    - Gather information about entities
    - Find connections between people and organizations
    
    Parameters:
        query (str): The search query to find relevant entities
        entity_type (Optional[str]): Optional filter for entity type ('Person' or 'Organization')
        limit (Optional[int]): Maximum number of entities to return
        
    Return Value:
        str: Formatted string containing the search results
    """
    try:
        # Generate embedding for the query
        embeddings_service = EmbeddingsService()
        query_embedding = embeddings_service.generate_embedding(query)
        
        # Build the search query
        cypher_query = """
        MATCH (e)
        WHERE (e:Person OR e:Organization)
        AND ($entity_type IS NULL OR e:$entity_type)
        WITH e, vector.similarity(e.embedding, $query_embedding) AS score
        WHERE score >= 0.7
        
        // Get relationships
        OPTIONAL MATCH (e)-[r]-(related)
        WITH e, score, collect(DISTINCT type(r) + ' -> ' + related.name) as relationships
        
        RETURN e {
            .*, 
            relationships: relationships
        } as entity, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        # Execute the search
        with driver.session() as session:
            result = session.run(
                cypher_query,
                query_embedding=query_embedding,
                entity_type=entity_type,
                limit=limit
            )
            
            entities = list(result)
            
            if not entities:
                return "No relevant entities found."
            
            # Format results
            formatted_results = ["Here are the most relevant entities:"]
            for i, record in enumerate(entities, 1):
                entity = record["entity"]
                score = record["score"]
                
                # Parse metadata from JSON string if it exists
                metadata = entity.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Format the entity information based on type
                entity_type = entity.get('type', 'unknown')
                
                if entity_type == 'Person':
                    entity_info = [
                        f"\n{i}. Relevance Score: {score:.2f}",
                        f"Person: {entity.get('name')}",
                        f"Category: {entity.get('category', 'Unknown')}",
                        f"Role: {entity.get('role', 'Unknown')}",
                        f"Notes: {entity.get('notes', 'No notes available')}"
                    ]
                elif entity_type == 'Organization':
                    entity_info = [
                        f"\n{i}. Relevance Score: {score:.2f}",
                        f"Organization: {entity.get('name')}",
                        f"Type: {entity.get('org_type', 'Unknown')}",
                        f"Industry: {entity.get('industry', 'Unknown')}",
                        f"Description: {entity.get('description', 'No description available')}"
                    ]
                else:
                    entity_info = [
                        f"\n{i}. Relevance Score: {score:.2f}",
                        f"Entity: {entity.get('name')} (Type: {entity_type})",
                        f"Description: {entity.get('description', 'No description available')}"
                    ]
                
                # Add relationships if available
                relationships = entity.get('relationships', [])
                if relationships:
                    entity_info.append("Relationships:")
                    for rel in relationships:
                        entity_info.append(f"  - {rel}")
                
                # Add relevant metadata if it exists
                if metadata:
                    entity_info.append("Metadata:")
                    for key, value in metadata.items():
                        entity_info.append(f"  - {key}: {value}")
                
                formatted_results.append("\n".join(entity_info))
            
            return "\n".join(formatted_results)
            
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        return f"An error occurred while searching entities: {str(e)}" 