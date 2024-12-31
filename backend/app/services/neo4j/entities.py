"""
Entities operations for Neo4j.

This module handles the creation and management of entity nodes
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any
from uuid import UUID
from neo4j import ManagedTransaction
from app.models.memory import BaseEntity, PersonEntity, OrganizationEntity, SmileDocument
from app.services.embeddings import EmbeddingsService
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize embeddings service
embeddings_service = EmbeddingsService()

def create_entity_node(tx: ManagedTransaction, entity: BaseEntity) -> str:
    """
    Create or update an entity node in Neo4j with embedding.
    Uses MERGE to avoid duplicates, matching on name and type.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction object
        entity (BaseEntity): Entity object to store (PersonEntity, OrganizationEntity, or SmileDocument)
        
    Returns:
        str: The ID of the created/updated node
    """
    # Generate embedding if not provided
    if entity.embedding is None:
        # Create text representation for embedding
        if isinstance(entity, SmileDocument):
            text_for_embedding = entity.to_embedding_text()
        else:
            text_for_embedding = f"{entity.name} {entity.type} {entity.notes if hasattr(entity, 'notes') else ''}"
        entity.embedding = embeddings_service.generate_embedding(text_for_embedding)
    
    # Convert entity to dictionary and prepare properties
    properties = entity.model_dump()
    properties['id'] = str(properties['id'])
    properties = convert_properties_for_neo4j(properties)
    
    # Build the query using MERGE to avoid duplicates
    # Match on name and type for uniqueness
    query = """
    MERGE (e:%s {name: $name, type: $type})
    ON CREATE SET
        e.id = $id,
        e.created_at = datetime($created_at),
        e.embedding = $embedding
    """ % entity.type
    
    # Add category for Person entities
    if isinstance(entity, PersonEntity):
        query += ", e.category = $category"
    
    # Add additional fields that should be updated on match
    query += """
    ON MATCH SET
        e.embedding = $embedding,
        e.updated_at = datetime($created_at)
    """
    
    # Add optional fields if they exist and are not null
    optional_fields = ['notes', 'nickname', 'birth_date', 'email', 'phone', 'address', 
                      'industry', 'website', 'metadata']
    for field in optional_fields:
        if field in properties and properties[field]:
            query += f", e.{field} = ${field}"
    
    query += "\nRETURN e {.*, embedding: null} as e"
    
    try:
        # Execute the query
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            entity.db_id = record["e"]["id"]  # Set the db_id on the entity
            logger.info(f"Successfully created/updated entity node with name: {properties['name']}")
            return record["e"]["id"]
        else:
            raise Exception("Failed to create/update entity node")
    except Exception as e:
        logger.error(f"Error creating/updating entity node: {str(e)}")
        raise

def get_or_create_person_entity(tx: ManagedTransaction, person_details: Dict[str, Any]) -> PersonEntity:
    """
    Get or create a person entity in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        person_details (Dict[str, Any]): Dictionary containing person details
        
    Returns:
        PersonEntity: The created or retrieved person entity
    """
    # Ensure required fields are present
    required_fields = ['name', 'category']
    for field in required_fields:
        if field not in person_details:
            raise ValueError(f"Missing required field: {field}")
    
    # Create PersonEntity instance
    person = PersonEntity(
        name=person_details['name'],
        category=person_details['category'],
        notes=person_details.get('notes', ''),
        type='Person'
    )
    
    # Create or update the person node and get the ID
    db_id = create_entity_node(tx, person)
    person.db_id = db_id
    
    return person 