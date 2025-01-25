"""Tool for saving documents in the knowledge base."""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.services.neo4j import driver, create_entity_node, create_relationship
from app.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class DocumentSaveSchema(BaseModel):
    """Schema for saving documents in the knowledge base."""
    
    name: str = Field(description="Name of the document")
    content: str = Field(description="Content of the document")
    doc_type: str = Field(description="Type of document (e.g. 'text', 'code', 'image')")
    topics: Optional[list[str]] = Field(None, description="List of topics covered in the document")
    entities: Optional[list[str]] = Field(None, description="List of entities mentioned in the document")
    summary: Optional[str] = Field(None, description="Brief summary of the document")
    tags: Optional[list[str]] = Field(None, description="List of tags for categorizing the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the document")

@tool
def save_document(
    name: str,
    content: str,
    doc_type: str,
    topics: Optional[list[str]] = None,
    entities: Optional[list[str]] = None,
    summary: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Save content to a document in the knowledge base.

    Parameters:
        name (str): Name of the document
        content (str): Content of the document
        doc_type (str): Type of document (e.g. 'text', 'code', 'image')
        topics (Optional[list[str]]): Optional list of topics covered in the document
        entities (Optional[list[str]]): Optional list of entities mentioned in the document
        summary (Optional[str]): Optional brief summary of the document
        tags (Optional[list[str]]): Optional list of tags for categorizing the document
        metadata (Optional[Dict[str, Any]]): Optional additional metadata about the document

    Return Value:
        Optional[Dict[str, Any]]: Document data if successful, None if failed
    """
    try:
        # Create base library directory if it doesn't exist
        library_path = os.path.join(os.getcwd(), "library")
        os.makedirs(library_path, exist_ok=True)
        
        # Create document type directory
        doc_type_dir = doc_type.lower().replace(" ", "_")
        type_path = os.path.join(library_path, doc_type_dir)
        os.makedirs(type_path, exist_ok=True)
        
        # Create document dictionary
        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "name": name,
            "doc_type": doc_type,
            "content": content,
            "file_path": "",  # Will be set after we create the filename with ID
            "file_url": "",   # Will be set after we create the filename with ID
            "file_type": os.path.splitext(name)[1].lstrip('.') or 'txt',
            "topics": topics or [],
            "entities": entities or [],
            "summary": summary,
            "tags": tags or [],
            "metadata": metadata or {},
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": "SMILE"  # Or pass from context if available
        }
        
        # Add ID to filename
        name_base, ext = os.path.splitext(name)
        filename_with_id = f"{name_base}_{doc_id}{ext}"
        
        # Create relative and absolute file paths with ID
        rel_file_path = os.path.join(doc_type_dir, filename_with_id)
        abs_file_path = os.path.join(library_path, rel_file_path)
        
        # Update document with final paths
        doc["file_path"] = rel_file_path
        doc["file_url"] = os.path.abspath(abs_file_path)
        
        # Generate embedding text
        embedding_text = f"{doc['name']}\n{doc['content']}\n{doc['summary'] or ''}"
        if doc['topics']:
            embedding_text += f"\nTopics: {', '.join(doc['topics'])}"
        if doc['entities']:
            embedding_text += f"\nEntities: {', '.join(doc['entities'])}"
        if doc['tags']:
            embedding_text += f"\nTags: {', '.join(doc['tags'])}"
        
        # Generate embedding
        doc["embedding"] = EmbeddingsService(driver=driver).generate_embedding(embedding_text)
        
        # Save content to file
        with open(abs_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create Neo4j nodes and relationships
        with driver.session() as session:
            # Create document node
            doc_db_id = session.execute_write(create_entity_node, doc)
            
            # Update document with database ID
            doc["db_id"] = doc_db_id
            
            # Create embedding vector
            if doc["embedding"]:
                session.execute_write(lambda tx: tx.run("""
                    MATCH (d:Document {id: $id})
                    CALL db.create.setNodeVectorProperty(d, 'embedding', $embedding)
                    """,
                    id=doc_id,
                    embedding=doc["embedding"]
                ))
            
            # Create relationships for topics and entities
            for topic in doc["topics"]:
                relationship = {
                    "source_id": doc_id,
                    "source_type": "Document",
                    "target_id": topic,
                    "target_type": "Topic",
                    "type": "COVERS"
                }
                session.execute_write(create_relationship, relationship)
                
            for entity in doc["entities"]:
                relationship = {
                    "source_id": doc_id,
                    "source_type": "Document",
                    "target_id": entity,
                    "target_type": "Entity",
                    "type": "MENTIONS"
                }
                session.execute_write(create_relationship, relationship)
        
        logger.info(f"Document saved successfully: {doc['file_url']}")
        return doc
        
    except Exception as e:
        error_msg = f"Error saving document: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) 