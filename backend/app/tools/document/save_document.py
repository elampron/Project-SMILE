"""Tool for saving documents in SMILE's knowledge base."""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.models.memory import SmileDocument
from app.services.neo4j import driver, create_entity_node, create_entity_relationship
from app.services.embeddings import EmbeddingsService

# Configure logging
logger = logging.getLogger(__name__)

# Define schema for document saving
class DocumentSaveSchema(BaseModel):
    """Schema for document saving parameters."""
    name: str = Field(description="Name of the document (including extension)")
    content: str = Field(description="Content to save in the document")
    doc_type: str = Field(description="Type of document (e.g., Documentation, Web Summary, Task Guide)")
    topics: Optional[List[str]] = Field(
        default=None,
        description="List of topics covered in document",
        items={"type": "string", "description": "A topic covered in the document"}
    )
    entities: Optional[List[str]] = Field(
        default=None,
        description="Named entities mentioned in document",
        items={"type": "string", "description": "A named entity mentioned in the document"}
    )
    summary: Optional[str] = Field(default=None, description="Brief summary of content")
    tags: Optional[List[str]] = Field(
        default=None,
        description="Custom tags for categorization",
        items={"type": "string", "description": "A tag for categorizing the document"}
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the document",
        properties={
            "created_by": {"type": "string", "description": "Creator of the document"},
            "created_at": {"type": "string", "description": "Creation timestamp"},
            "version": {"type": "integer", "description": "Document version"},
            "status": {"type": "string", "description": "Document status"},
            "language": {"type": "string", "description": "Document language"},
            "additional": {"type": "object", "description": "Any additional metadata"}
        }
    )

@tool
def save_document(
    name: str,
    content: str,
    doc_type: str,
    topics: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
    summary: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save content to a document in the SMILE library and create corresponding nodes/relationships in Neo4j.
    
    Args:
        name (str): Name of the document (including extension)
        content (str): Content to save in the document
        doc_type (str): Type of document (e.g., Documentation, Web Summary, Task Guide)
        topics (Optional[List[str]]): List of topics covered in document
        entities (Optional[List[str]]): Named entities mentioned in document
        summary (Optional[str]): Brief summary of content
        tags (Optional[List[str]]): Custom tags for categorization
        metadata (Optional[Dict[str, Any]]): Optional metadata about the document
    
    Returns:
        str: URL/path to the saved document
    """
    try:
        # Create base library directory if it doesn't exist
        library_path = os.path.join(os.getcwd(), "library")
        os.makedirs(library_path, exist_ok=True)
        
        # Create document type directory
        doc_type_dir = doc_type.lower().replace(" ", "_")
        type_path = os.path.join(library_path, doc_type_dir)
        os.makedirs(type_path, exist_ok=True)
        
        # Create document model first to get the ID
        doc = SmileDocument(
            name=name,
            doc_type=doc_type,
            content=content,
            file_path="",  # Will be set after we create the filename with ID
            file_url="",   # Will be set after we create the filename with ID
            file_type=os.path.splitext(name)[1].lstrip('.') or 'txt',
            topics=topics or [],
            entities=entities or [],
            summary=summary,
            tags=tags or [],
            metadata=metadata or {},
            updated_at=datetime.utcnow(),
            created_by="SMILE"  # Or pass from context if available
        )
        
        # Add ID to filename
        name_base, ext = os.path.splitext(name)
        filename_with_id = f"{name_base}_{str(doc.id)}{ext}"
        
        # Create relative and absolute file paths with ID
        rel_file_path = os.path.join(doc_type_dir, filename_with_id)
        abs_file_path = os.path.join(library_path, rel_file_path)
        
        # Update document with final paths
        doc.file_path = rel_file_path
        doc.file_url = os.path.abspath(abs_file_path)
        
        # Generate embedding
        doc.embedding = EmbeddingsService(driver=driver).generate_embedding(doc.to_embedding_text())
        
        # Save content to file
        with open(abs_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create Neo4j nodes and relationships
        with driver.session() as session:
            # Create document node using existing method
            doc_db_id = session.execute_write(create_entity_node, doc)
            
            # Update document with database ID
            doc.db_id = doc_db_id
            
            # Create embedding vector
            if doc.embedding:
                session.execute_write(lambda tx: tx.run("""
                    MATCH (d:Document {id: $id})
                    CALL db.create.setNodeVectorProperty(d, 'embedding', $embedding)
                    """,
                    id=str(doc.id),
                    embedding=doc.embedding
                ))
            
            # Create relationships using existing method
            for relationship in doc.get_relationships():
                session.execute_write(create_entity_relationship, relationship)
        
        logger.info(f"Document saved successfully: {doc.file_url}")
        return doc.file_url
        
    except Exception as e:
        error_msg = f"Error saving document: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) 