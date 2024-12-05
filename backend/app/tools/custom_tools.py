import subprocess
import logging
from langchain_core.tools import BaseTool, StructuredTool, tool
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import os
from datetime import datetime
from app.models.memory import SmileDocument, DocumentType
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService
import json

# Configure logging
logger = logging.getLogger(__name__)

# Define schema for document saving
class DocumentSaveSchema(BaseModel):
    name: str = Field(description="Name of the document (including extension)")
    content: str = Field(description="Content to save in the document")
    doc_type: str = Field(description="Type of document (e.g., Documentation, Web Summary, Task Guide)")
    topics: Optional[List[str]] = Field(default=None, description="List of topics covered in document")
    entities: Optional[List[str]] = Field(default=None, description="Named entities mentioned in document")
    summary: Optional[str] = Field(default=None, description="Brief summary of content")
    tags: Optional[List[str]] = Field(default=None, description="Custom tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the document")

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

# Define schema for Python execution tool
class PythonExecuteSchema(BaseModel):
    code: str = Field(description="Python code to execute")

# Define schema for Command execution tool
class CommandExecuteSchema(BaseModel):
    command: str = Field(description="Command to execute")

@tool
def execute_python(code: str) -> str:
    """Execute Python code and return the result.
    
    Use this tool when you need to execute Python code. The code will be evaluated
    and the result returned as a string.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Result of code execution or error message
    """
    logger.info(f"Executing Python code: {code}")
    try:
        # Add safety measures and execution logic here
        result = eval(code)  # Be careful with eval - you might want to use a safer execution method
        logger.info(f"Python execution successful: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error executing Python code: {str(e)}")
        return f"Error executing Python code: {str(e)}"

@tool
def execute_cmd(command: str) -> str:
    """Execute a system command and return the output.
    
    Use this tool when you need to run system commands. The command will be executed
    in a shell and the output returned as a string.
    
    Args:
        command (str): Command to execute
        
    Returns:
        str: Command execution output or error message
    """
    logger.info(f"Executing command: {command}")
    try:
        # Add safety measures and execution logic here
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout if result.stdout else result.stderr
        logger.info(f"Command execution successful: {output}")
        return output
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return f"Error executing command: {str(e)}"

if __name__ == "__main__":
    def test_save_document():
        """Test function to demonstrate save_document usage."""
        try:
            # Test document content
            test_content = """# Project SMILE Architecture Overview

This document provides a high-level overview of the Project SMILE architecture.

## Components
1. Core AI Agent (SMILE)
2. Memory System
3. Document Management
4. Knowledge Graph

## Key Features
- Natural language interaction
- Long-term memory storage
- Document organization
- Relationship tracking"""

            # Test saving a documentation file
            doc_url = save_document.invoke({
                "name": "smile_architecture.md",
                "content": test_content,
                "doc_type": "Documentation",
                "topics": ["architecture", "system design", "project structure"],
                "entities": ["SMILE", "Memory System", "Knowledge Graph"],
                "summary": "High-level architectural overview of Project SMILE",
                "tags": ["technical", "architecture", "documentation"],
                "metadata": {
                    "version": "1.0",
                    "status": "draft",
                    "author": "SMILE System",
                    "department": "Engineering"
                }
            })
            
            print(f"\nDocument saved successfully!")
            print(f"File URL: {doc_url}")
            print("\nDocument properties:")
            print("- Type: Documentation")
            print("- Topics: architecture, system design, project structure")
            print("- Entities: SMILE, Memory System, Knowledge Graph")
            print("- Tags: technical, architecture, documentation")
            
            # Test saving a task guide
            guide_content = """# How to Use SMILE's Document Management

A quick guide on using SMILE's document management features.

1. Save documents using natural language
2. Organize by type and topic
3. Search using semantic queries
4. Track relationships between documents"""

            guide_url = save_document.invoke({
                "name": "document_management_guide.md",
                "content": guide_content,
                "doc_type": "Task Guide",
                "topics": ["documentation", "user guide", "document management"],
                "entities": ["SMILE", "Document Management"],
                "summary": "Guide for using SMILE's document management features",
                "tags": ["guide", "how-to", "user documentation"],
                "metadata": {
                    "difficulty": "beginner",
                    "estimated_time": "10 minutes",
                    "prerequisites": ["Basic SMILE knowledge"]
                }
            })
            
            print(f"\nGuide saved successfully!")
            print(f"File URL: {guide_url}")
            print("\nGuide properties:")
            print("- Type: Task Guide")
            print("- Topics: documentation, user guide, document management")
            print("- Entities: SMILE, Document Management")
            print("- Tags: guide, how-to, user documentation")

        except Exception as e:
            print(f"Error testing save_document: {str(e)}")
            raise

    # Run the test
    print("Testing SMILE Document Management System...")
    test_save_document()
    print("\nTest completed successfully!")

