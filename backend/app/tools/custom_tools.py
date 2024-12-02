import subprocess
import logging
from langchain_core.tools import BaseTool, StructuredTool, tool
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from datetime import datetime
from app.models.memory import Document
from app.services.neo4j import driver
from app.services.embeddings import EmbeddingsService

# Configure logging
logger = logging.getLogger(__name__)

# Define schema for document saving
class DocumentSaveSchema(BaseModel):
    name: str = Field(description="Name of the document (including extension)")
    content: str = Field(description="Content to save in the document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the document")

@tool
def save_document(name: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save content to a document in the library and create a corresponding node in Neo4j.
    
    Args:
        name (str): Name of the document (including extension)
        content (str): Content to save in the document
        metadata (Optional[Dict[str, Any]]): Optional metadata about the document
    
    Returns:
        str: URL/path to the saved document
    """
    try:
        # Create library directory if it doesn't exist
        library_path = os.path.join(os.getcwd(), "library")
        os.makedirs(library_path, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(library_path, name)
        file_url = os.path.abspath(file_path)
        
        # Create document model
        doc = Document(
            name=name,
            content=content,
            file_url=file_url,
            metadata=metadata or {},
            updated_at=datetime.utcnow()
        )
        
        # Generate embedding
        doc.embedding = EmbeddingsService().generate_embedding(doc.to_embedding_text())
        
        # Save content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create Neo4j node
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("""
                CREATE (d:Document {
                    id: $id,
                    name: $name,
                    file_url: $file_url,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    metadata: $metadata
                })
                """,
                id=str(doc.id),
                name=doc.name,
                file_url=doc.file_url,
                created_at=doc.created_at.isoformat(),
                updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
                metadata=doc.metadata
            ))
            
            # Create embedding vector
            if doc.embedding:
                session.execute_write(lambda tx: tx.run("""
                    MATCH (d:Document {id: $id})
                    CALL db.create.setVectorProperty(d, 'embedding', $embedding)
                    RETURN d
                    """,
                    id=str(doc.id),
                    embedding=doc.embedding
                ))
        
        logger.info(f"Document saved successfully: {file_url}")
        return file_url
        
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

