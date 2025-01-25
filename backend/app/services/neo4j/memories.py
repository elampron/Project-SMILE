"""
Memory operations for Neo4j.

This module handles the creation and management of memory nodes
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any, List
from neo4j import ManagedTransaction
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_cognitive_memory_node(tx: ManagedTransaction, memory: Any) -> str:
    """
    Create a cognitive memory node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory: CognitiveMemory object containing the details
        
    Returns:
        str: ID of the created memory node
    """
    # Import here to avoid circular dependency
    from app.models.memory import CognitiveMemory
    
    if not isinstance(memory, CognitiveMemory):
        raise ValueError("memory must be a CognitiveMemory instance")
    
    # Convert memory to dictionary
    properties = memory.model_dump()
    properties = convert_properties_for_neo4j(properties)
    
    # Build query
    query = """
    CREATE (m:Memory {
        id: $id,
        type: $type,
        content: $content,
        created_at: datetime($created_at),
        validation_status: $validation_status,
        confidence: $confidence
    })
    RETURN m.id as id
    """
    
    try:
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            logger.info(f"Created memory node: {memory.type}")
            return record["id"]
        else:
            raise Exception("Failed to create memory node")
    except Exception as e:
        logger.error(f"Error creating memory node: {str(e)}")
        raise

def get_similar_memories(tx: ManagedTransaction, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get memories similar to the query embedding.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        query_embedding (List[float]): Query embedding to compare against
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of similar memories
    """
    query = """
    CALL db.index.vector.queryNodes('memory_embedding', $limit, $embedding)
    YIELD node, score
    RETURN node {.*, score: score} as memory
    ORDER BY score DESC
    """
    
    try:
        result = tx.run(query, embedding=query_embedding, limit=limit)
        return [record["memory"] for record in result]
    except Exception as e:
        logger.error(f"Error getting similar memories: {str(e)}")
        raise

def get_memory_by_id(tx: ManagedTransaction, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a memory by ID.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_id (str): ID of the memory to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: Memory data if found
    """
    query = """
    MATCH (m:Memory {id: $memory_id})
    RETURN m {.*, embedding: null} as memory
    """
    
    try:
        result = tx.run(query, memory_id=memory_id)
        record = result.single()
        return record["memory"] if record else None
    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}")
        raise

def update_memory_validation(tx: ManagedTransaction, memory_id: str, validation_status: str) -> bool:
    """
    Update the validation status of a memory.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_id (str): ID of the memory to update
        validation_status (str): New validation status
        
    Returns:
        bool: True if updated successfully
    """
    # Import here to avoid circular dependency
    from app.models.memory import ValidationStatus
    
    if validation_status not in ValidationStatus.__members__:
        raise ValueError(f"Invalid validation status: {validation_status}")
    
    query = """
    MATCH (m:Memory {id: $memory_id})
    SET m.validation_status = $validation_status
    RETURN m.id as id
    """
    
    try:
        result = tx.run(query, memory_id=memory_id, validation_status=validation_status)
        record = result.single()
        if record:
            logger.info(f"Updated memory validation status: {memory_id} -> {validation_status}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating memory validation: {str(e)}")
        raise

def get_memories_by_type(tx: ManagedTransaction, memory_type: str) -> List[Dict[str, Any]]:
    """
    Get all memories of a specific type.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_type (str): Type of memories to retrieve
        
    Returns:
        List[Dict[str, Any]]: List of memories
    """
    query = """
    MATCH (m:Memory {type: $type})
    RETURN m {.*, embedding: null} as memory
    ORDER BY m.created_at DESC
    """
    
    try:
        result = tx.run(query, type=memory_type)
        return [record["memory"] for record in result]
    except Exception as e:
        logger.error(f"Error getting memories by type: {str(e)}")
        raise 