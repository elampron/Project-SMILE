"""
Relationship operations for Neo4j.

This module handles the creation and management of relationships
between nodes in the Neo4j database.
"""

import logging
from neo4j import ManagedTransaction
from app.models.memory import Relationship
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_entity_relationship(tx: ManagedTransaction, relationship: Relationship) -> None:
    """
    Create or update a relationship between two entities in Neo4j.

    Args:
        tx (ManagedTransaction): The Neo4j transaction
        relationship (Relationship): The relationship object to be created or updated
    """
    properties = {
        'id': str(relationship.id),
        'from_entity_id': relationship.from_entity_db_id,
        'to_entity_id': relationship.to_entity_db_id,
        'since': relationship.since,
        'until': relationship.until,
        'notes': relationship.notes
    }

    # Remove None values and convert properties
    properties = convert_properties_for_neo4j({k: v for k, v in properties.items() if v is not None})

    # Build the Cypher query
    merge_query = f"""
    MATCH (from), (to)
    WHERE from.id = $from_entity_id AND to.id = $to_entity_id
    MERGE (from)-[r:{relationship.type.upper().replace(' ', '_')}]-(to)
    ON CREATE SET {', '.join([f'r.{k} = ${k}' for k in properties.keys() if k != 'from_entity_id' and k != 'to_entity_id'])}
    ON MATCH SET {', '.join([f'r.{k} = ${k}' for k in properties.keys() if k != 'from_entity_id' and k != 'to_entity_id'])}
    """

    try:
        # Log the query for debugging
        logger.debug(f"Executing query: {merge_query} with parameters: {properties}")

        # Execute the query
        tx.run(merge_query, **properties)
        logger.info(f"Successfully created/updated relationship of type {relationship.type}")
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        raise 