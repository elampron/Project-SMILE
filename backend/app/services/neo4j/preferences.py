"""
Preference operations for Neo4j.

This module handles the creation and management of preference nodes
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any, List
from neo4j import ManagedTransaction
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_preference_node(tx: ManagedTransaction, preference: Any) -> str:
    """
    Create a preference node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        preference: Preference object containing the details
        
    Returns:
        str: ID of the created preference node
    """
    # Import here to avoid circular dependency
    from app.models.memory import Preference
    
    if not isinstance(preference, Preference):
        raise ValueError("preference must be a Preference instance")
    
    # Convert preference to dictionary
    properties = preference.model_dump()
    properties = convert_properties_for_neo4j(properties)
    
    # Build query
    query = """
    CREATE (p:Preference {
        id: $id,
        type: $type,
        value: $value,
        created_at: datetime($created_at),
        updated_at: datetime($updated_at)
    })
    RETURN p.id as id
    """
    
    try:
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            logger.info(f"Created preference node: {preference.type} = {preference.value}")
            return record["id"]
        else:
            raise Exception("Failed to create preference node")
    except Exception as e:
        logger.error(f"Error creating preference node: {str(e)}")
        raise

def fetch_existing_preference_types(tx: ManagedTransaction) -> List[str]:
    """
    Get all existing preference types.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        
    Returns:
        List[str]: List of preference types
    """
    query = """
    MATCH (p:Preference)
    RETURN DISTINCT p.type as type
    """
    
    try:
        result = tx.run(query)
        return [record["type"] for record in result]
    except Exception as e:
        logger.error(f"Error fetching preference types: {str(e)}")
        raise

def get_preferences(tx: ManagedTransaction, preference_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get preferences, optionally filtered by type.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        preference_type (Optional[str]): Type to filter by
        
    Returns:
        List[Dict[str, Any]]: List of preferences
    """
    if preference_type:
        query = """
        MATCH (p:Preference {type: $type})
        RETURN p {.*, embedding: null} as preference
        ORDER BY p.created_at DESC
        """
        params = {"type": preference_type}
    else:
        query = """
        MATCH (p:Preference)
        RETURN p {.*, embedding: null} as preference
        ORDER BY p.created_at DESC
        """
        params = {}
    
    try:
        result = tx.run(query, **params)
        return [record["preference"] for record in result]
    except Exception as e:
        logger.error(f"Error getting preferences: {str(e)}")
        raise

def update_preference(tx: ManagedTransaction, preference_id: str, new_value: Any) -> bool:
    """
    Update a preference value.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        preference_id (str): ID of the preference to update
        new_value: New value to set
        
    Returns:
        bool: True if updated successfully
    """
    query = """
    MATCH (p:Preference {id: $id})
    SET p.value = $value,
        p.updated_at = datetime()
    RETURN p.id as id
    """
    
    try:
        result = tx.run(query, id=preference_id, value=new_value)
        record = result.single()
        if record:
            logger.info(f"Updated preference: {preference_id} = {new_value}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating preference: {str(e)}")
        raise 