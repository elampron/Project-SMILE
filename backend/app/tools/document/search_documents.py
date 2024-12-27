"""Tool for searching documents in SMILE's knowledge base."""

import json
import logging
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.services.neo4j import driver, vector_similarity_search, get_related_entities
from app.services.embeddings import EmbeddingsService

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
    
    # Add model field for service
    embeddings_service: EmbeddingsService = Field(default_factory=lambda: EmbeddingsService())
    
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
            f"Created: {doc.get('created_at', 'Unknown')}",
            f"File Path: {doc.get('file_path', 'unknown')}"
        ]
        
        # Add related entities if available
        related_entities = doc.get('related_entities', [])
        if related_entities:
            doc_info.append("Related Entities:")
            for entity in related_entities:
                doc_info.append(f"  - {entity}")
        
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
            
            # Build document type filter if specified
            additional_filters = None
            if doc_type:
                additional_filters = f"node.doc_type = '{doc_type}'"
            
            # Execute the search
            with driver.session() as session:
                results = vector_similarity_search(
                    session=session,
                    index_name="document_embeddings",
                    query_vector=query_embedding,
                    k=limit,
                    min_score=0.7,
                    additional_filters=additional_filters
                )
                
                if not results:
                    return "No relevant documents found."
                
                # Get related entities for each document
                for result in results:
                    doc_id = result["node"].get("id")
                    if doc_id:
                        related = get_related_entities(
                            session=session,
                            node_id=doc_id
                        )
                        result["node"]["related_entities"] = [
                            entity.get("name") for entity in related
                        ]
                
                # Format results
                formatted_results = ["Here are the most relevant documents:"]
                for i, result in enumerate(results, 1):
                    doc = result["node"]
                    score = result["score"]
                    formatted_doc = self._format_document(doc)
                    formatted_results.append(f"\n{i}. Relevance Score: {score:.2f}")
                    formatted_results.append(formatted_doc)
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"An error occurred while searching documents: {str(e)}"

if __name__ == "__main__":
    """
    Test script for SearchDocumentsTool
    """
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize tool
        search_tool = SearchDocumentsTool()
        
        # Create test data
        with driver.session() as session:
            # Create test document
            session.run("""
                CREATE (d:Document)
                SET d = $properties
                """,
                properties={
                    "id": "test-doc-1",
                    "name": "Project Requirements",
                    "doc_type": "meeting_notes",
                    "summary": "Discussion of AI project requirements and timeline",
                    "content": "Meeting covered key project requirements including ML models and deployment strategy",
                    "topics": ["AI", "Project Planning", "Requirements"],
                    "created_at": "2024-01-01",
                    "file_path": "/docs/project/requirements.md",
                    "is_test": True,
                    "embedding": search_tool.embeddings_service.generate_embedding(
                        "Meeting covered key project requirements including ML models and deployment strategy"
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
                    "role": "Project Manager",
                    "is_test": True
                }
            )
            
            # Create relationship
            session.run("""
                MATCH (d:Document {id: $doc_id})
                MATCH (e:Person {id: $entity_id})
                CREATE (d)-[:MENTIONS]->(e)
                """,
                doc_id="test-doc-1",
                entity_id="test-person-1"
            )
            
            logger.info("Created test data")
            
            # Test queries
            test_queries = [
                {
                    "query": "Find documents about AI project requirements",
                    "doc_type": None
                },
                {
                    "query": "Show me meeting notes",
                    "doc_type": "meeting_notes"
                }
            ]
            
            # Run test queries
            for test in test_queries:
                logger.info("\n" + "=" * 50)
                logger.info(f"Testing query: {test['query']}")
                logger.info(f"Document type filter: {test['doc_type']}")
                
                try:
                    result = search_tool._run(
                        query=test["query"],
                        doc_type=test["doc_type"]
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