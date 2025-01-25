"""
Summary operations for Neo4j.

This module handles the creation and management of summary nodes
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any, List
from neo4j import ManagedTransaction
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_summary_node(tx: ManagedTransaction, summary: Any) -> str:
    """
    Create a summary node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        summary: Summary object containing the details
        
    Returns:
        str: ID of the created summary node
    """
    # Import here to avoid circular dependency
    from app.models.memory import ConversationSummary
    
    if not isinstance(summary, ConversationSummary):
        raise ValueError("summary must be a ConversationSummary instance")
    
    # Convert summary to dictionary
    properties = summary.model_dump()
    properties = convert_properties_for_neo4j(properties)
    
    # Build query
    query = """
    CREATE (s:Summary {
        id: $id,
        thread_id: $thread_id,
        summary: $summary,
        last_updated: datetime($last_updated)
    })
    RETURN s.id as id
    """
    
    try:
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            logger.info(f"Created summary node for thread: {summary.thread_id}")
            return record["id"]
        else:
            raise Exception("Failed to create summary node")
    except Exception as e:
        logger.error(f"Error creating summary node: {str(e)}")
        raise

def create_summary_relationships(tx: ManagedTransaction, summary_id: str, entity_ids: List[str]) -> None:
    """
    Create relationships between a summary and mentioned entities.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        summary_id (str): ID of the summary node
        entity_ids (List[str]): List of entity IDs mentioned in the summary
    """
    query = """
    MATCH (s:Summary {id: $summary_id})
    MATCH (e) WHERE e.id IN $entity_ids
    CREATE (s)-[r:MENTIONS]->(e)
    """
    
    try:
        tx.run(query, summary_id=summary_id, entity_ids=entity_ids)
        logger.info(f"Created relationships for summary: {summary_id}")
    except Exception as e:
        logger.error(f"Error creating summary relationships: {str(e)}")
        raise

def get_summary(tx: ManagedTransaction, thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest summary for a thread.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        thread_id (str): ID of the conversation thread
        
    Returns:
        Optional[Dict[str, Any]]: Summary data if found
    """
    query = """
    MATCH (s:Summary {thread_id: $thread_id})
    RETURN s {.*, embedding: null} as summary
    ORDER BY s.last_updated DESC
    LIMIT 1
    """
    
    try:
        result = tx.run(query, thread_id=thread_id)
        record = result.single()
        return record["summary"] if record else None
    except Exception as e:
        logger.error(f"Error getting summary: {str(e)}")
        raise 