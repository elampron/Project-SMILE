"""
Preference operations for Neo4j.

This module handles the creation and management of preference nodes
in the Neo4j database.
"""

import logging
from typing import List
from neo4j import ManagedTransaction
from app.models.memory import Preference
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def fetch_existing_preference_types(tx: ManagedTransaction) -> List[str]:
    """
    Fetch all existing preference types from the database.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        
    Returns:
        List[str]: List of existing preference types
    """
    query = """
    MATCH (p:Preference)
    RETURN DISTINCT p.preference_type as type, p {.*, embedding: null} as p
    """
    result = tx.run(query)
    return [record["type"] for record in result if record["type"]]

def create_preference_node(tx: ManagedTransaction, preference: Preference) -> None:
    """
    Create a preference node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        preference (Preference): Preference object to store
    """
    # Convert preference to dictionary and handle property conversion
    properties = preference.model_dump()
    properties['id'] = str(properties['id'])
    properties = convert_properties_for_neo4j(properties)
    
    # Add required fields
    properties['value'] = str(properties.get('details', {}))  # Convert details to string for value
    properties['source'] = 'direct_observation'  # Default source
    properties['confidence'] = 1.0  # Default confidence
    
    # Build the query
    query = """
    MERGE (p:Preference {id: $id})
    ON CREATE SET
        p.preference_type = $preference_type,
        p.value = $value,
        p.created_at = datetime($created_at),
        p.source = $source,
        p.confidence = $confidence
    ON MATCH SET
        p.value = $value,
        p.source = $source,
        p.confidence = $confidence
    """
    
    try:
        # Execute the query
        tx.run(query, **properties)
        logger.info(f"Successfully created/updated preference node with ID: {properties['id']}")
    except Exception as e:
        logger.error(f"Error creating preference node: {str(e)}")
        raise 