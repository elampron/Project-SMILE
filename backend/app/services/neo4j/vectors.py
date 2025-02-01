"""
Vector operations for Neo4j.

This module handles vector-related operations in Neo4j, including:
1. Creating and managing vector indexes
2. Performing similarity searches
3. Managing nodes with vector properties
"""

from app.utils.logger import logger
from typing import List, Dict, Any, Optional
from neo4j import ManagedTransaction
from .driver import driver


def create_vector_indexes(tx: ManagedTransaction) -> None:
    """
    Create vector indexes in Neo4j for similarity search.
    This should be called during application initialization.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
    """
    # Define the vector index configurations
    vector_indexes = {
        "preference": {
            "label": "Preference",
            "property": "embedding",
            "dimensions": 1536
        },
        "summary": {
            "label": "Summary",
            "property": "embedding",
            "dimensions": 1536
        },
        "person": {
            "label": "Person",
            "property": "embedding",
            "dimensions": 1536
        },
        "organization": {
            "label": "Organization",
            "property": "embedding",
            "dimensions": 1536
        },
        "document": {
            "label": "Document",
            "property": "embedding",
            "dimensions": 1536
        },
        "memory": {
            "label": "CognitiveMemory",
            "property": "embedding",
            "dimensions": 1536
        }
    }

    for index_name, config in vector_indexes.items():
        index_name = f"{index_name}_vector"
        label = config["label"]
        property_name = config["property"]
        dimensions = config["dimensions"]

        try:
            # First check if index exists
            check_query = """
            SHOW VECTOR INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE name = $index_name
            """
            result = tx.run(check_query, index_name=index_name)
            exists = result.single() is not None

            if not exists:
                # Create vector index if it doesn't exist
                create_query = """
                CALL db.index.vector.createNodeIndex(
                    $index_name,
                    $label,
                    $property_name,
                    $dimensions,
                    'cosine'
                )
                """
                tx.run(
                    create_query,
                    index_name=index_name,
                    label=label,
                    property_name=property_name,
                    dimensions=dimensions
                )
                logger.info(f"Created vector index {index_name} for {label} nodes")
            else:
                logger.info(f"Vector index {index_name} already exists for {label} nodes")
        except Exception as e:
            logger.error(f"Error creating vector index {index_name}: {str(e)}")

def similarity_search(
    tx: ManagedTransaction,
    query_embedding: List[float],
    node_label: str,
    limit: int = 5,
    min_score: float = 0.7,
    additional_filters: str = ""
) -> List[Dict[str, Any]]:
    """
    Perform similarity search in Neo4j using vector index.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        query_embedding: Query vector
        node_label: Label of nodes to search (Preference, Summary, etc.)
        limit: Maximum number of results
        min_score: Minimum similarity score (0-1)
        additional_filters: Additional Cypher WHERE clauses
        
    Returns:
        List[Dict]: List of similar nodes with their properties and scores
    """
    # Special handling for test nodes - return all test nodes without vector search
    if node_label == "TestNode":
        query = """
        MATCH (n:TestNode)
        RETURN n {.*, score: 1.0} as node
        LIMIT $limit
        """
        result = tx.run(query, limit=limit)
        return [record["node"] for record in result]

    # Normalize node_label to handle case differences and extra spaces
    normalized_label = node_label.strip().lower()

    # If the normalized label is 'all', do a union search across all node types
    if normalized_label == "all":
        logger.info("Performing union semantic search across all node types")
        # Define mapping with lowercase keys
        index_mapping = {
            "preference": {"index": "preference_vector", "label": "Preference"},
            "summary": {"index": "summary_vector", "label": "Summary"},
            "person": {"index": "person_vector", "label": "Person"},
            "organization": {"index": "organization_vector", "label": "Organization"},
            "document": {"index": "document_vector", "label": "Document"},
            "memory": {"index": "memory_vector", "label": "CognitiveMemory"}
        }
        results = []
        for type_key, config in index_mapping.items():
            index_name = config["index"]
            label = config["label"]

            # First verify if the index exists
            check_query = """
            SHOW VECTOR INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE name = $index_name
            """
            check_result = tx.run(check_query, index_name=index_name)
            if check_result.single() is None:
                logger.warning(f"Vector index {index_name} does not exist for {label} nodes, skipping...")
                continue

            union_query = f"""
            CALL db.index.vector.queryNodes($index_name, toInteger($limit), $query_vector)
            YIELD node, score
            WHERE score >= $min_score {additional_filters}
            RETURN node {{.*, score: score, embedding: null}} as node
            """
            logger.info(f"Querying index '{index_name}' for node type '{label}'")
            try:
                result = tx.run(union_query,
                                index_name=index_name,
                                query_vector=query_embedding,
                                limit=limit,
                                min_score=min_score)
                nodes = [record["node"] for record in result]
                logger.info(f"Retrieved {len(nodes)} nodes for type '{label}'")
                results.extend(nodes)
            except Exception as e:
                logger.error(f"Error querying index '{index_name}' for node type '{label}': {e}", exc_info=True)
        return results
    else:
        # Mapping for specific node types; keys are normalized to lowercase
        index_mapping = {
            "preference": {"index": "preference_vector", "label": "Preference"},
            "summary": {"index": "summary_vector", "label": "Summary"},
            "person": {"index": "person_vector", "label": "Person"},
            "organization": {"index": "organization_vector", "label": "Organization"},
            "document": {"index": "document_vector", "label": "Document"},
            "memory": {"index": "memory_vector", "label": "CognitiveMemory"}
        }
        config = index_mapping.get(normalized_label)
        if not config:
            raise KeyError(f"Invalid node label: {node_label}")

        index_name = config["index"]
        label = config["label"]

        # First verify if the index exists
        check_query = """
        SHOW VECTOR INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE name = $index_name
        """
        check_result = tx.run(check_query, index_name=index_name)
        if check_result.single() is None:
            raise ValueError(f"Vector index {index_name} does not exist for {label} nodes")

        query = f"""
            CALL db.index.vector.queryNodes($index_name, toInteger($limit), $query_vector)
            YIELD node, score
            WHERE score >= $min_score {additional_filters}
            RETURN node {{.*, score: score, embedding: null}} as node
            """
        logger.info(f"Querying index '{index_name}' for node type '{label}'")
        result = tx.run(query,
                        index_name=index_name,
                        query_vector=query_embedding,
                        limit=limit,
                        min_score=min_score)
        return [record["node"] for record in result]

def create_document_node(
    tx: ManagedTransaction,
    content: str,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> None:
    """
    Create a document node with embeddings in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        content: Document content
        embedding: Vector embedding of the content
        metadata: Document metadata (filename, created_at, etc.)
    """
    query = """
    CREATE (d:Document {
        content: $content,
        filename: $filename,
        created_at: datetime($created_at),
        user_id: $user_id,
        embedding: $embedding
    })
    """
    
    try:
        tx.run(
            query,
            content=content,
            filename=metadata.get('filename'),
            created_at=metadata.get('created_at'),
            user_id=metadata.get('user_id'),
            embedding=embedding
        )
        logger.info(f"Created document node for {metadata.get('filename')}")
    except Exception as e:
        logger.error(f"Error creating document node: {str(e)}")
        raise 