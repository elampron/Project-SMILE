"""
Graph operations for Neo4j.

This module handles graph-related operations in Neo4j, including:
1. Fetching nodes and relationships
2. Graph exploration
3. Graph visualization data
"""

from typing import List, Dict, Any, Optional
from neo4j import ManagedTransaction
from app.utils.logger import logger
from .driver import driver
from .utils import convert_datetime_fields, Neo4jEncoder
import json

def get_nodes_by_type_neo4j(
    tx: ManagedTransaction,
    node_type: str,
    skip: int = 0,
    limit: int = 10,
    filter_by: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get nodes of a specific type with optional filtering.
    
    Args:
        tx: Neo4j transaction
        node_type: Type of nodes to fetch
        skip: Number of nodes to skip
        limit: Maximum number of nodes to return
        filter_by: Optional filter string to match against node properties
        
    Returns:
        List of nodes with their properties
    """
    if filter_by:
        query = """
        MATCH (n)
        WHERE labels(n)[0] = $node_type AND any(prop in keys(n) where n[prop] =~ $filter)
        RETURN n {.*, id: elementId(n)} as node
        SKIP $skip
        LIMIT $limit
        """
        result = tx.run(query, node_type=node_type, filter=filter_by, skip=skip, limit=limit)
    else:
        query = """
        MATCH (n)
        WHERE labels(n)[0] = $node_type
        RETURN n {.*, id: elementId(n)} as node
        SKIP $skip
        LIMIT $limit
        """
        result = tx.run(query, node_type=node_type, skip=skip, limit=limit)

    try:
        nodes = [record["node"] for record in result]
        return [convert_datetime_fields(node) for node in nodes]
    except Exception as e:
        logger.error(f"Error fetching nodes: {str(e)}")
        raise e

def get_node_relationships_neo4j(
    tx: ManagedTransaction,
    node_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get relationships for a specific node.
    
    Args:
        tx: Neo4j transaction
        node_id: ID of the node
        limit: Maximum number of relationships to return
        
    Returns:
        List of relationships with their properties
    """
    query = """
    MATCH (n)-[r]-(m)
    WHERE elementId(n) = $node_id
    RETURN {
        id: elementId(r),
        type: type(r),
        properties: properties(r),
        source: elementId(startNode(r)),
        target: elementId(endNode(r)),
        source_node: properties(startNode(r)),
        target_node: properties(endNode(r))
    } as relationship
    LIMIT $limit
    """
    
    try:
        result = tx.run(query, node_id=node_id, limit=limit)
        return [record["relationship"] for record in result]
    except Exception as e:
        logger.error(f"Error fetching relationships: {str(e)}")
        raise e

def get_graph_visualization_data_neo4j(tx, node_types: List[str], limit: int = 100) -> Dict[str, List[Dict]]:
    """
    Get graph data for visualization, including nodes and their relationships.
    
    Args:
        tx: Neo4j transaction
        node_types: List of node types to include
        limit: Maximum number of nodes to return
        
    Returns:
        Dictionary containing nodes and relationships
    """
    # First get nodes
    nodes_query = """
    MATCH (n)
    WHERE any(label in labels(n) WHERE label in $node_types)
    WITH collect(n) as nodes
    UNWIND nodes as n
    RETURN collect(distinct {
        id: elementId(n),
        labels: labels(n),
        properties: properties(n)
    }) as nodes
    """
    nodes_result = tx.run(nodes_query, node_types=node_types)
    nodes = nodes_result.single()["nodes"]
    
    # Get relationships between these nodes
    rels_query = """
    MATCH (n)-[r]-(m)
    WHERE any(label in labels(n) WHERE label in $node_types)
    AND any(label in labels(m) WHERE label in $node_types)
    WITH DISTINCT r, startNode(r) as s, endNode(r) as e
    WHERE elementId(s) in $node_ids AND elementId(e) in $node_ids
    RETURN collect({
        id: elementId(r),
        type: type(r),
        source: elementId(s),
        target: elementId(e),
        properties: properties(r)
    }) as relationships
    """
    node_ids = [node["id"] for node in nodes]
    rels_result = tx.run(rels_query, node_types=node_types, node_ids=node_ids)
    result = rels_result.single()
    relationships = result["relationships"] if result else []
    
    return {
        "nodes": nodes,
        "relationships": relationships
    }

def explore_graph_neo4j(
    tx,
    search: Optional[str] = None,
    node_type: Optional[str] = None,
    limit: int = 100
) -> Dict[str, List[Dict]]:
    """
    Explore the graph with optional search and filtering.
    
    Args:
        tx: Neo4j transaction
        search: Optional search term to filter nodes
        node_type: Optional node type to filter by
        limit: Maximum number of nodes to return
        
    Returns:
        Dictionary containing nodes and relationships
    """
    params = {"limit": limit}
    type_filter = ""
    search_filter = ""
    
    if node_type:
        type_filter = "AND n:" + node_type
        params["node_type"] = node_type
        
    if search:
        search_filter = "WHERE any(prop in keys(n) where n[prop] =~ $search)"
        params["search"] = f"(?i).*{search}.*"
    
    # Get nodes
    nodes_query = f"""
    MATCH (n)
    {search_filter} {type_filter}
    WITH n LIMIT $limit
    RETURN collect(distinct {{
        id: elementId(n),
        labels: labels(n),
        properties: properties(n)
    }}) as nodes
    """
    nodes_result = tx.run(nodes_query, params)
    nodes = nodes_result.single()["nodes"]
    
    # Convert datetime fields in node properties
    for node in nodes:
        if "properties" in node:
            node["properties"] = convert_datetime_fields(node["properties"])
    
    # Get relationships between these nodes
    rels_query = f"""
    MATCH (n)
    {search_filter} {type_filter}
    WITH n LIMIT $limit
    MATCH (n)-[r]-(m)
    RETURN collect(distinct {{
        id: elementId(r),
        type: type(r),
        source: elementId(startNode(r)),
        target: elementId(endNode(r)),
        properties: properties(r)
    }}) as relationships
    """
    rels_result = tx.run(rels_query, params)
    relationships = rels_result.single()["relationships"]
    
    # Convert datetime fields in relationship properties
    for rel in relationships:
        if "properties" in rel:
            rel["properties"] = convert_datetime_fields(rel["properties"])
    
    # Convert the result to JSON and back to handle any remaining datetime objects
    result = {
        "nodes": nodes,
        "relationships": relationships
    }
    result_json = json.dumps(result, cls=Neo4jEncoder)
    return json.loads(result_json) 
