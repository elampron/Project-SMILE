"""
Attachments module for the SMILE agent.

This module contains functions for handling file attachments and document processing.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, AsyncGenerator
from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import aiofiles
from app.utils.logger import logger
from app.models.memory import SmileDocument

if TYPE_CHECKING:
    from app.agents.smile.agent import Smile

async def process_attachments(agent: Any, attachments: List[Dict[str, Any]]) -> List[str]:
    """
    Process a list of attachments.
    
    Args:
        agent (Any): The agent instance
        attachments (List[Dict[str, Any]]): List of attachment dictionaries
        
    Returns:
        List[str]: List of processed attachment names
        
    Raises:
        Exception: If processing fails
    """
    processed = []
    
    for attachment in attachments:
        try:
            # Save document and get metadata
            doc = await save_document(agent, attachment)
            if doc:
                processed.append(doc.get('filename'))
                logger.info(f"Successfully processed attachment: {doc.get('filename')}")
                
        except Exception as e:
            logger.error(f"Failed to process attachment: {str(e)}")
            continue
            
    return processed

async def save_document(agent: Any, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Save a document and its metadata.
    
    Args:
        agent (Any): The agent instance
        document (Dict[str, Any]): Document dictionary containing file data and metadata
        
    Returns:
        Optional[Dict[str, Any]]: Saved document data or None if save fails
        
    Raises:
        Exception: If save fails
    """
    try:
        # Extract document info
        filename = document.get('filename')
        content = document.get('content')
        mime_type = document.get('mime_type', 'application/octet-stream')
        
        if not filename or not content:
            raise ValueError("Missing required document fields")
            
        # Generate unique ID
        doc_id = hashlib.sha256(
            f"{filename}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        # Create document object
        doc = {
            'id': doc_id,
            'filename': filename,
            'mime_type': mime_type,
            'created_at': datetime.utcnow(),
            'metadata': {
                'size': len(content),
                'source': 'user_upload'
            }
        }
        
        # Save file to disk asynchronously
        file_path = Path(agent) / doc_id
        if isinstance(content, str):
            content = content.encode('utf-8')
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
            
        logger.info(f"Saved document {filename} to {file_path}")
        return doc
        
    except Exception as e:
        logger.error(f"Failed to save document: {str(e)}")
        raise

async def get_document(agent: Any, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by ID.
    
    Args:
        agent (Any): The agent instance
        doc_id (str): The document ID
        
    Returns:
        Optional[Dict[str, Any]]: Retrieved document or None if not found
        
    Raises:
        Exception: If retrieval fails
    """
    try:
        file_path = Path(agent) / doc_id
        if not file_path.exists():
            logger.warning(f"Document {doc_id} not found")
            return None
            
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            
        return {
            'id': doc_id,
            'content': content,
            'created_at': datetime.fromtimestamp(file_path.stat().st_ctime)
        }
        
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise

async def delete_document(agent: Any, doc_id: str) -> bool:
    """
    Delete a document by ID.
    
    Args:
        agent (Any): The agent instance
        doc_id (str): The document ID
        
    Returns:
        bool: True if deleted successfully, False otherwise
        
    Raises:
        Exception: If deletion fails
    """
    try:
        file_path = Path(agent) / doc_id
        if not file_path.exists():
            logger.warning(f"Document {doc_id} not found for deletion")
            return False
            
        file_path.unlink()
        logger.info(f"Deleted document {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise 