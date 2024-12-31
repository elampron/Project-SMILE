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
    indexes = [
        ("Preference", "preference_vector", "embedding", 1536),  # OpenAI embedding dimension
        ("Summary", "summary_vector", "embedding", 1536),
        ("Person", "person_vector", "embedding", 1536),
        ("Organization", "org_vector", "embedding", 1536),
        ("Document", "document_vector", "embedding", 1536),
        ("CognitiveMemory", "memory_vector", "embedding", 1536)
    ]
    
    for label, index_name, property_name, dimensions in indexes:
        try:
            # Create vector index
            query = f"""
            CALL db.index.vector.createNodeIndex(
                $index_name,
                $label,
                $property_name,
                $dimensions,
                'cosine'
            )
            """
            tx.run(
                query,
                index_name=index_name,
                label=label,
                property_name=property_name,
                dimensions=dimensions
            )
            logger.info(f"Created vector index {index_name} for {label} nodes")
        except Exception as e:
            # Check for specific error messages indicating index already exists
            if any(msg in str(e) for msg in [
                "already exists an index",
                "AlreadyIndexedException",
                "An equivalent index already exists"
            ]):
                logger.info(f"Vector index {index_name} already exists for {label} nodes")
            else:
                logger.error(f"Error creating vector index {index_name}: {str(e)}")
                # Don't raise the error, just log it and continue
                # This allows other indexes to be created even if one fails

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
    index_name = {
        "Preference": "preference_vector",
        "Summary": "summary_vector",
        "Person": "person_vector",
        "Organization": "org_vector",
        "Document": "document_vector",
        "CognitiveMemory": "memory_vector"
    }[node_label]
    
    # Build the query with optional filters
    query = """
    CALL db.index.vector.queryNodes(
        $index_name,
        $k,
        $query_vector
    ) YIELD node, score
    WHERE score >= $min_score """ + additional_filters + """
    WITH node, score
    RETURN node {.*, embedding: null} as node, score
    ORDER BY score DESC
    """
    
    try:
        result = tx.run(
            query,
            index_name=index_name,
            k=limit,
            query_vector=query_embedding,
            min_score=min_score
        )
        
        return [
            {
                **dict(record["node"]),
                "similarity_score": record["score"]
            }
            for record in result
        ]
    except Exception as e:
        logger.error(f"Error performing similarity search: {str(e)}")
        raise

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