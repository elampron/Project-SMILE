"""
Conversation summary operations for Neo4j.

This module handles the creation and management of conversation summaries
and their relationships in the Neo4j database.
"""

import logging
from neo4j import ManagedTransaction
from app.models.memory import ConversationSummary
from app.services.embeddings import EmbeddingsService
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize embeddings service
embeddings_service = EmbeddingsService()

def create_summary_node(tx: ManagedTransaction, summary: ConversationSummary) -> None:
    """
    Create a summary node in Neo4j with embedding.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction object
        summary (ConversationSummary): ConversationSummary object to store
    """
    properties = summary.model_dump()
    properties['id'] = str(properties['id'])
    
    # Generate embedding if not provided
    if summary.embedding is None:
        # Create text representation for embedding
        text_for_embedding = f"{summary.content} {' '.join(summary.topics)}"
        summary.embedding = embeddings_service.generate_embedding(text_for_embedding)
    
    # Remove nested objects for node properties
    del properties['action_items']
    del properties['participants']
    del properties['sentiments']
    
    # Add embedding to properties
    properties['embedding'] = summary.embedding
    
    # Convert properties for Neo4j
    properties = convert_properties_for_neo4j(properties)
    
    query = """
    CREATE (s:Summary {
        id: $id,
        content: $content,
        topics: $topics,
        location: $location,
        events: $events,
        created_at: datetime($created_at),
        message_ids: $message_ids,
        embedding: $embedding
    })
    RETURN s {.*, embedding: null} as s
    """
    
    try:
        tx.run(query, **properties)
        logger.info(f"Successfully created summary node with ID: {properties['id']}")
    except Exception as e:
        logger.error(f"Error creating summary node: {str(e)}")
        raise

def create_summary_relationships(tx: ManagedTransaction, summary: ConversationSummary) -> None:
    """
    Create relationships between Summary node and its related nodes in Neo4j.
    Handles participants, action items, sentiments, topics, and entities.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction object
        summary (ConversationSummary): ConversationSummary object containing relationship data
    """
    try:
        # Create relationships to participants
        for participant in summary.participants:
            participant_name = participant.name
            # Merge participant node
            participant_query = """
            MERGE (p:Person {name: $name})
            RETURN p {.*, embedding: null} as p
            """
            tx.run(participant_query, name=participant_name)
            # Create relationship
            relationship_query = """
            MATCH (s:Summary {id: $summary_id}), (p:Person {name: $name})
            CREATE (s)-[:INVOLVES]->(p)
            """
            tx.run(relationship_query, summary_id=str(summary.id), name=participant_name)
        
        # Create relationships for topics
        for topic in summary.topics:
            # Merge topic node
            topic_query = """
            MERGE (t:Topic {name: $name})
            RETURN t {.*, embedding: null} as t
            """
            tx.run(topic_query, name=topic)
            # Create relationship
            relationship_query = """
            MATCH (s:Summary {id: $summary_id}), (t:Topic {name: $name})
            MERGE (s)-[:COVERS_TOPIC]->(t)
            """
            tx.run(relationship_query, summary_id=str(summary.id), name=topic)
            
        logger.info(f"Successfully created relationships for summary with ID: {summary.id}")
    except Exception as e:
        logger.error(f"Error creating summary relationships: {str(e)}")
        raise 