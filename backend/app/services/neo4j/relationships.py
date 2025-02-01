"""
Relationship operations for Neo4j.

This module handles the creation and management of relationship edges
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any, List
from neo4j import ManagedTransaction
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_relationship(tx: ManagedTransaction, relationship: Any) -> str:
    """
    Create a relationship between two entities in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        relationship: Relationship object containing the details
        
    Returns:
        str: ID of the created relationship
    """
    # Import here to avoid circular dependency
    from app.models.memory import Relationship
    
    if not isinstance(relationship, Relationship):
        raise ValueError("relationship must be a Relationship instance")
    
    # Convert relationship to dictionary
    properties = relationship.model_dump()
    properties = convert_properties_for_neo4j(properties)
    
    # Build query
    query = """
    MATCH (from) WHERE from.id = $from_entity_id
    MATCH (to) WHERE to.id = $to_entity_id
    CREATE (from)-[r:RELATES {
        id: $id,
        type: $type,
        since: $since,
        until: $until,
        notes: $notes
    }]->(to)
    RETURN r.id as id
    """
    
    try:
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            logger.info(f"Created relationship: {relationship.type} from {relationship.from_entity_id} to {relationship.to_entity_id}")
            return record["id"]
        else:
            raise Exception("Failed to create relationship")
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        raise

def get_relationships(tx: ManagedTransaction, entity_id: str) -> List[Dict[str, Any]]:
    """
    Get all relationships for an entity.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        entity_id (str): ID of the entity
        
    Returns:
        List[Dict[str, Any]]: List of relationships
    """
    query = """
    MATCH (n)-[r:RELATES]-(m)
    WHERE n.id = $entity_id
    RETURN r {
        .*,
        from_name: CASE WHEN startNode(r).id = $entity_id 
                       THEN startNode(r).name 
                       ELSE endNode(r).name 
                  END,
        to_name: CASE WHEN startNode(r).id = $entity_id 
                     THEN endNode(r).name 
                     ELSE startNode(r).name 
                END
    } as relationship
    """
    
    try:
        result = tx.run(query, entity_id=entity_id)
        return [record["relationship"] for record in result]
    except Exception as e:
        logger.error(f"Error getting relationships: {str(e)}")
        raise

def delete_relationship(tx: ManagedTransaction, relationship_id: str) -> bool:
    """
    Delete a relationship by ID.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        relationship_id (str): ID of the relationship to delete
        
    Returns:
        bool: True if deleted successfully
    """
    query = """
    MATCH ()-[r:RELATES]->()
    WHERE r.id = $relationship_id
    DELETE r
    RETURN count(r) as deleted
    """
    
    try:
        result = tx.run(query, relationship_id=relationship_id)
        record = result.single()
        if record and record["deleted"] > 0:
            logger.info(f"Deleted relationship: {relationship_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting relationship: {str(e)}")
        raise 

def get_relationships_between_nodes(tx: ManagedTransaction,
                                node_ids: List[str],
                                relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get relationships between specified nodes with optional relationship type filtering.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        node_ids (List[str]): List of node IDs to find relationships between
        relationship_type (Optional[str]): Filter relationships by type
        
    Returns:
        List[Dict[str, Any]]: List of relationships with connected nodes details
    """
    # Base query with node filtering
    query = """
    MATCH (n)-[r:RELATES]-(m)
    WHERE n.id IN $node_ids AND m.id IN $node_ids
    """
    
    # Add relationship type filter if specified
    if relationship_type:
        query += "AND r.type = $relationship_type\n"
    
    # Complete the query with return statement
    query += """
    RETURN r {
        .*,
        from_node: {
            id: startNode(r).id,
            name: startNode(r).name,
            type: startNode(r).type
        },
        to_node: {
            id: endNode(r).id,
            name: endNode(r).name,
            type: endNode(r).type
        }
    } as relationship
    """
    
    try:
        params = {"node_ids": node_ids}
        if relationship_type:
            params["relationship_type"] = relationship_type
            
        result = tx.run(query, **params)
        relationships = [record["relationship"] for record in result]
        
        logger.info(f"Retrieved {len(relationships)} relationships between specified nodes")
        return relationships
        
    except Exception as e:
        logger.error(f"Error getting relationships between nodes: {str(e)}")
        raise