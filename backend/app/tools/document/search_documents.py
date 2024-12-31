"""Tool for searching documents in SMILE's knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService
from app.agents.context import ContextManager

# Configure logging
logger = logging.getLogger(__name__)

class SearchDocumentsInput(BaseModel):
    """Input for searching documents."""
    query: str = Field(
        description="The search query to find relevant documents. This can be a question, topic, or description."
    )
    doc_type: Optional[str] = Field(
        None,
        description="Optional filter for document type (e.g., 'meeting_notes', 'report', 'email')"
    )
    limit: Optional[int] = Field(
        3,
        description="Maximum number of documents to return. Defaults to 3."
    )

class SearchDocumentsTool(BaseTool):
    """Tool for searching documents in SMILE's knowledge base."""
    
    name: ClassVar[str] = "search_documents"
    description: ClassVar[str] = """
    Search for relevant documents in SMILE's knowledge base.
    Use this tool when you need to:
    - Find documents related to a specific topic
    - Search for information in stored documents
    - Look up specific document types
    - Gather context about a subject from stored documents
    
    The tool will return the most relevant documents based on semantic search.
    """
    
    args_schema: ClassVar[type[BaseModel]] = SearchDocumentsInput
    
    # Add model fields for services
    embeddings_service: EmbeddingsService = Field(default_factory=lambda: EmbeddingsService())
    context_manager: ContextManager = Field(default_factory=lambda: ContextManager(driver))
    
    def __init__(self, **data):
        """Initialize the tool with necessary services."""
        super().__init__(**data)
    
    def _format_document(self, doc: Dict) -> str:
        """Format a single document result."""
        # Parse metadata from JSON string if it exists
        metadata = doc.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        # Format the document information
        doc_info = [
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
        
        return "\n".join(doc_info)
    
    def _run(self, query: str, doc_type: Optional[str] = None, limit: int = 3) -> str:
        """
        Execute the document search.
        
        Args:
            query: Search query text
            doc_type: Optional document type filter
            limit: Maximum number of results to return
            
        Returns:
            str: Formatted string of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_service.generate_embedding(query)
            
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
                    formatted_doc = self._format_document(doc)
                    formatted_results.append(f"\n{i}. Relevance Score: {score:.2f}")
                    formatted_results.append(formatted_doc)
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"An error occurred while searching documents: {str(e)}" 