"""Tool for searching documents in the knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class SearchDocumentsInput(BaseModel):
    """Schema for searching documents in the knowledge base."""
    
    query: str = Field(description="The search query to find relevant documents")
    doc_type: Optional[str] = Field(None, description="Optional filter for document type")
    limit: Optional[int] = Field(3, description="Maximum number of documents to return")

@tool
def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    limit: Optional[int] = 3
) -> str:
    """Search for relevant documents in the knowledge base.
    
    Use this tool when you need to:
    - Find documents related to a topic
    - Search for specific files or content
    - Gather information from documents
    - Find patterns across documents
    
    Parameters:
        query (str): The search query to find relevant documents
        doc_type (Optional[str]): Optional filter for document type
        limit (Optional[int]): Maximum number of documents to return
        
    Return Value:
        str: Formatted string containing the search results
    """
    try:
        # Generate embedding for the query
        embeddings_service = EmbeddingsService()
        query_embedding = embeddings_service.generate_embedding(query)
        
        # Build the search query
        cypher_query = """
        MATCH (d:Document)
        WHERE $doc_type IS NULL OR d.doc_type = $doc_type
        WITH d, vector.similarity(d.embedding, $query_embedding) AS score
        WHERE score >= 0.7
        RETURN d {.*}, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        # Execute the search
        with driver.session() as session:
            result = session.run(
                cypher_query,
                query_embedding=query_embedding,
                doc_type=doc_type,
                limit=limit
            )
            
            documents = list(result)
            
            if not documents:
                return "No relevant documents found."
            
            # Format results
            formatted_results = ["Here are the most relevant documents:"]
            for i, record in enumerate(documents, 1):
                doc = record["d"]
                score = record["score"]
                
                # Parse metadata from JSON string if it exists
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Format the document information
                doc_info = [
                    f"\n{i}. Relevance Score: {score:.2f}",
                    f"Document: {doc.get('name')} (Type: {doc.get('doc_type', 'unknown')})",
                    f"Summary: {doc.get('summary', 'No summary available')}",
                    f"Topics: {', '.join(doc.get('topics', [])) or 'None'}",
                    f"Entities: {', '.join(doc.get('entities', [])) or 'None'}",
                    f"File Path: {doc.get('file_path', 'unknown')}"
                ]
                
                # Add relevant metadata if it exists
                if metadata:
                    doc_info.append("Metadata:")
                    for key, value in metadata.items():
                        doc_info.append(f"  - {key}: {value}")
                
                formatted_results.append("\n".join(doc_info))
            
            return "\n".join(formatted_results)
            
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return f"An error occurred while searching documents: {str(e)}" 