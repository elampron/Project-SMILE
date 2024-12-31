"""
Memory operations for Neo4j.

This module handles the creation and management of memory nodes
in the Neo4j database.
"""

import logging
from typing import List, Optional
from uuid import UUID
from neo4j import ManagedTransaction
from app.models.memory import CognitiveMemory, ValidationStatus
from app.services.embeddings import EmbeddingsService
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize embeddings service
embeddings_service = EmbeddingsService()

def create_cognitive_memory_node(tx: ManagedTransaction, memory: CognitiveMemory) -> None:
    """
    Create a cognitive memory node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory (CognitiveMemory): Memory object to store
    """
    # Generate embedding if not provided
    if memory.embedding is None:
        # Create text representation for embedding
        text_for_embedding = memory.to_embedding_text()
        memory.embedding = embeddings_service.generate_embedding(text_for_embedding)

    # Convert memory to dictionary and prepare properties
    properties = memory.model_dump()
    properties['id'] = str(properties['id'])
    
    # Ensure required fields are set with default values if not provided
    properties['memory_type'] = properties.get('type', 'general')  # Use 'type' field as memory_type
    properties['context'] = properties.get('context', 'general')
    properties['validation_status'] = properties.get('validation', {}).get('is_valid', True)
    properties['source_type'] = properties.get('source', 'direct_observation')
    properties['source_id'] = str(properties.get('id'))  # Use memory's own ID if no source_id
    properties['confidence'] = properties.get('confidence', 1.0)
    
    # Convert properties for Neo4j
    properties = convert_properties_for_neo4j(properties)

    # Build the query
    query = """
    CREATE (m:CognitiveMemory {
        id: $id,
        content: $content,
        memory_type: $memory_type,
        context: $context,
        created_at: datetime($created_at),
        validation_status: $validation_status,
        embedding: $embedding,
        source_type: $source_type,
        source_id: $source_id,
        confidence: $confidence
    })
    """

    try:
        # Execute the query
        tx.run(query, **properties)
        logger.info(f"Successfully created memory node with ID: {properties['id']}")

        # Create relationship to source if provided
        if memory.source_id:
            source_relationship_query = """
            MATCH (m:CognitiveMemory {id: $memory_id}), (s {id: $source_id})
            CREATE (m)-[:DERIVED_FROM]->(s)
            """
            tx.run(
                source_relationship_query,
                memory_id=str(memory.id),
                source_id=memory.source_id
            )
            logger.info(f"Created DERIVED_FROM relationship for memory {memory.id}")

    except Exception as e:
        logger.error(f"Error creating memory node: {str(e)}")
        raise

def get_similar_memories(tx: ManagedTransaction, text: str, limit: int = 5) -> List[CognitiveMemory]:
    """
    Get memories similar to the provided text using vector similarity search.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        text (str): Text to find similar memories for
        limit (int): Maximum number of memories to return
        
    Returns:
        List[CognitiveMemory]: List of similar memories
    """
    # Generate embedding for the query text
    query_embedding = embeddings_service.generate_embedding(text)

    # Build the query
    query = """
    CALL db.index.vector.queryNodes('memory_embedding_index', $limit, $embedding)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    """

    try:
        result = tx.run(query, limit=limit, embedding=query_embedding)
        memories = []
        for record in result:
            node = record["node"]
            properties = dict(node.items())
            memory = CognitiveMemory(**properties)
            memories.append(memory)
        return memories
    except Exception as e:
        logger.error(f"Error getting similar memories: {str(e)}")
        raise

def get_memory_by_id(tx: ManagedTransaction, memory_id: UUID) -> Optional[CognitiveMemory]:
    """
    Get a memory by its ID.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_id (UUID): ID of the memory to retrieve
        
    Returns:
        Optional[CognitiveMemory]: The memory if found, None otherwise
    """
    query = """
    MATCH (m:CognitiveMemory {id: $id})
    RETURN m {.*, embedding: null} as m
    """

    try:
        result = tx.run(query, id=str(memory_id))
        record = result.single()
        if record:
            properties = dict(record["m"].items())
            return CognitiveMemory(**properties)
        return None
    except Exception as e:
        logger.error(f"Error getting memory by ID: {str(e)}")
        raise

def update_memory_validation(tx: ManagedTransaction, memory_id: UUID, validation: ValidationStatus) -> None:
    """
    Update the validation status of a memory.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_id (UUID): ID of the memory to update
        validation (ValidationStatus): New validation status
    """
    query = """
    MATCH (m:CognitiveMemory {id: $id})
    SET m.validation_status = $validation_status
    """

    try:
        tx.run(query, id=str(memory_id), validation_status=validation.value)
        logger.info(f"Updated validation status for memory {memory_id} to {validation.value}")
    except Exception as e:
        logger.error(f"Error updating memory validation: {str(e)}")
        raise

def get_memories_by_type(tx: ManagedTransaction, memory_type: str, limit: int = 10) -> List[CognitiveMemory]:
    """
    Get memories of a specific type.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory_type (str): Type of memories to retrieve
        limit (int): Maximum number of memories to return
        
    Returns:
        List[CognitiveMemory]: List of memories of the specified type
    """
    query = """
    MATCH (m:CognitiveMemory)
    WHERE m.memory_type = $memory_type
    RETURN m {.*, embedding: null} as m
    ORDER BY m.created_at DESC
    LIMIT $limit
    """

    try:
        result = tx.run(query, memory_type=memory_type, limit=limit)
        memories = []
        for record in result:
            properties = dict(record["m"].items())
            memory = CognitiveMemory(**properties)
            memories.append(memory)
        return memories
    except Exception as e:
        logger.error(f"Error getting memories by type: {str(e)}")
        raise 