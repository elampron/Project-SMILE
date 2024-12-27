"""Tool for searching knowledge in SMILE's knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService
from app.services.knowledge_search import KnowledgeSearchService

# Configure logging
logger = logging.getLogger(__name__)

class SearchKnowledgeInput(BaseModel):
    """Input for searching knowledge."""
    query: str = Field(
        description="The search query to find relevant knowledge. This can be a question, topic, or description."
    )
    node_types: Optional[List[str]] = Field(
        None,
        description="Optional list of node types to search (e.g., ['Preference', 'Summary', 'Person', 'Organization', 'Document', 'CognitiveMemory'])"
    )
    limit: Optional[int] = Field(
        5,
        description="Maximum number of knowledge items to return. Defaults to 5."
    )

class SearchKnowledgeTool(BaseTool):
    """Tool for searching knowledge in SMILE's knowledge base."""
    
    name: ClassVar[str] = "search_knowledge"
    description: ClassVar[str] = """
    Search for relevant knowledge in SMILE's knowledge base.
    Use this tool when you need to:
    - Find information across different types of nodes
    - Search for preferences, summaries, people, organizations, documents, or memories
    - Gather comprehensive context about a subject
    - Find semantically similar information
    
    The tool will return the most relevant knowledge items based on semantic search.
    """
    
    args_schema: ClassVar[type[BaseModel]] = SearchKnowledgeInput
    embeddings_service: EmbeddingsService = Field(exclude=True)
    knowledge_service: KnowledgeSearchService = Field(exclude=True)
    
    def __init__(self, embeddings_service: EmbeddingsService, **data):
        """Initialize the tool with necessary services."""
        knowledge_service = KnowledgeSearchService(driver, embeddings_service)
        super().__init__(embeddings_service=embeddings_service, knowledge_service=knowledge_service, **data)
    
    def _format_knowledge(self, item: Dict) -> str:
        """Format a single knowledge result."""
        node = item.get('node', {})
        
        # Format the knowledge information
        knowledge_info = [
            f"Type: {item.get('type', 'Unknown')}",
            f"Content: {node.get('content') or node.get('details') or str(node)}",
            f"Score: {item.get('score', 0):.3f}"
        ]
        
        # Add created_at if available
        if created_at := node.get('created_at'):
            knowledge_info.append(f"Created: {created_at}")
        
        # Add properties if available
        for key, value in node.items():
            if key not in ['content', 'details', 'created_at', 'embedding', 'vector']:
                knowledge_info.append(f"{key}: {value}")
        
        return "\n".join(knowledge_info)
    
    def _run(self, query: str, node_types: Optional[List[str]] = None, limit: int = 5) -> str:
        """
        Execute the knowledge search.
        
        Args:
            query: Search query text
            node_types: Optional list of node types to search
            limit: Maximum number of results to return
            
        Returns:
            str: Formatted string of search results
        """
        try:
            # Execute the search using KnowledgeSearchService
            results = self.knowledge_service.search_knowledge(
                query=query,
                filters=node_types,
                limit=limit
            )
            
            if not results:
                return "No relevant knowledge found."
            
            # Format results
            formatted_results = ["Here are the most relevant knowledge items:"]
            for i, item in enumerate(results, 1):
                formatted_knowledge = self._format_knowledge(item)
                formatted_results.append(f"\n{i}. {formatted_knowledge}")
            
            return "\n".join(formatted_results)
                
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return f"An error occurred while searching knowledge: {str(e)}" 