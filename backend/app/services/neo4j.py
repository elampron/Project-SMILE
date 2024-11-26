# app/services/neo4j.py

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    from neo4j import GraphDatabase
except ImportError:
    raise ImportError("Please install neo4j-driver: pip install neo4j")

from app.configs.settings import settings
from app.models.memory import (
    PersonEntity, OrganizationEntity, Relationship, 
    ConversationSummary, Preference
)
from app.models.agents import User
from app.services.embeddings import EmbeddingsService

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize the driver
driver = GraphDatabase.driver(
    settings.app_config.get("neo4j_config").get("uri"),
    auth=(
        settings.app_config.get("neo4j_config").get("username"), 
        settings.app_config.get("neo4j_config").get("password")
    )
)

# Initialize embeddings service
embeddings_service = EmbeddingsService(driver)

def create_entity_node(tx, entity):
    """
    Create or update an entity node in Neo4j with embedding.
    
    Args:
        tx: Neo4j transaction object
        entity: Entity object to store (PersonEntity or OrganizationEntity)
    """
    # Generate embedding if not provided
    if entity.embedding is None:
        # Create text representation for embedding
        text_for_embedding = f"{entity.name} {entity.type} {entity.notes if entity.notes else ''}"
        entity.embedding = embeddings_service.generate_embedding(text_for_embedding)
    
    labels = [entity.type]
    properties = entity.dict()

    # Convert UUID fields to strings
    properties['id'] = str(properties['id'])

    # Ensure datetime fields are strings in ISO format
    properties['created_at'] = properties['created_at'].isoformat()
    if properties.get('updated_at'):
        properties['updated_at'] = properties['updated_at'].isoformat()
    else:
        properties['updated_at'] = None

    # Prepare unique properties based on entity type
    if entity.type == 'Person':
        unique_props = {
            'name': properties['name'],
            'category': properties['category']
        }
    elif entity.type == 'Organization':
        unique_props = {
            'name': properties['name']
        }

    # Remove properties that are not needed for the merge
    merge_properties = {k: v for k, v in unique_props.items() if v is not None}

    # Remaining properties to set or update
    on_match_set = {k: v for k, v in properties.items() if k not in unique_props and v is not None}

    # Build the Cypher query
    merge_query = f"""
    MERGE (e:{':'.join(labels)} {{{', '.join([f'{k}: ${k}' for k in merge_properties.keys()])}}})
    ON CREATE SET {', '.join([f'e.{k} = ${k}' for k in on_match_set.keys()])},
                  e.embedding = $embedding
    ON MATCH SET {', '.join([f'e.{k} = ${k}' for k in on_match_set.keys()])},
                 e.embedding = $embedding
    RETURN e.id as id
    """

    # Combine all parameters
    parameters = {**merge_properties, **on_match_set, 'embedding': entity.embedding}

    try:
        result = tx.run(merge_query, **parameters)
        record = result.single()
        if record:
            return record["id"]
        else:
            return None
    except Exception as e:
        logger.error(f"Error creating entity node: {str(e)}")
        raise

def create_entity_relationship(tx, relationship):
    """
    Create or update a relationship between two entities in Neo4j.

    Args:
        tx: The Neo4j transaction.
        relationship: The relationship object to be created or updated.
    """
    # Import here to avoid circular imports
    from app.models.memory import Relationship
    
    properties = {
        'id': str(relationship.id),
        'from_entity_id': relationship.from_entity_db_id,
        'to_entity_id': relationship.to_entity_db_id,
        'since': relationship.since,
        'until': relationship.until,
        'notes': relationship.notes
    }

    # Remove None values
    properties = {k: v for k, v in properties.items() if v is not None}

    # Build the Cypher query
    merge_query = f"""
    MATCH (from), (to)
    WHERE from.id = $from_entity_id AND to.id = $to_entity_id
    MERGE (from)-[r:{relationship.type.upper().replace(' ', '_')}]-(to)
    ON CREATE SET {', '.join([f'r.{k} = ${k}' for k in properties.keys() if k != 'from_entity_id' and k != 'to_entity_id'])}
    ON MATCH SET {', '.join([f'r.{k} = ${k}' for k in properties.keys() if k != 'from_entity_id' and k != 'to_entity_id'])}
    """

    # Log the query
    logger.debug(f"Executing query: {merge_query} with parameters: {properties}")

    # Execute the query
    tx.run(merge_query, **properties)

def create_summary_node(tx, summary: ConversationSummary):
    """
    Create a summary node in Neo4j with embedding.
    
    Args:
        tx: Neo4j transaction object
        summary: ConversationSummary object to store
    """
    properties = summary.model_dump()
    properties['id'] = str(properties['id'])
    properties['created_at'] = properties['created_at'].isoformat()
    
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
    RETURN s
    """
    tx.run(query, **properties)

def create_summary_relationships(tx, summary: ConversationSummary):
    """
    Create relationships between Summary node and its related nodes in Neo4j.
    Handles participants, action items, and sentiments.
    """
    # Create relationships to participants
    for participant in summary.participants:
        participant_name = participant.name
        # Merge participant node
        participant_query = """
        MERGE (p:Person {name: $name})
        RETURN p
        """
        tx.run(participant_query, name=participant_name)
        # Create relationship
        relationship_query = """
        MATCH (s:Summary {id: $summary_id}), (p:Person {name: $name})
        CREATE (s)-[:INVOLVES]->(p)
        """
        tx.run(relationship_query, summary_id=str(summary.id), name=participant_name)
    
    # Create relationships for action items
    for action_item in summary.action_items:
        action_query = """
        CREATE (a:ActionItem {
            description: $description,
            assignee: $assignee,
            due_date: date($due_date)
        })
        """
        tx.run(action_query, 
            description=action_item.description, 
            assignee=action_item.assignee, 
            due_date=action_item.due_date.isoformat() if action_item.due_date else None
        )
        # Create relationship
        relationship_query = """
        MATCH (s:Summary {id: $summary_id}), (a:ActionItem {description: $description})
        CREATE (s)-[:HAS_ACTION_ITEM]->(a)
        """
        tx.run(relationship_query, summary_id=str(summary.id), description=action_item.description)

def fetch_existing_preference_types(tx):
    """Get all existing preference types from Neo4j."""
    query = """
    MATCH (p:Preference)
    RETURN DISTINCT p.preference_type AS preference_type
    """
    result = tx.run(query)
    types = [record["preference_type"] for record in result]
    return types

def create_preference_node(tx, preference: Preference):
    """
    Create a preference node in Neo4j with embedding.
    
    Args:
        tx: Neo4j transaction object
        preference: Preference object to store
        
    Logs:
        DEBUG: Query execution details
        ERROR: When preference creation fails
    """
    # Serialize complex details object to JSON string if it's a dict
    details = json.dumps(preference.details) if isinstance(preference.details, dict) else preference.details
    
    # Generate embedding if not provided
    if preference.embedding is None:
        # Create text representation for embedding
        text_for_embedding = f"{preference.preference_type} {json.dumps(preference.details)}"
        preference.embedding = embeddings_service.generate_embedding(text_for_embedding)
    
    query = """
    MATCH (person:Person {id: $person_id})
    MERGE (pref:Preference {
        preference_type: $preference_type,
        details: $details
    })
    ON CREATE SET pref.id = $id,
                  pref.importance = $importance,
                  pref.created_at = datetime($created_at),
                  pref.embedding = $embedding
    CREATE (person)-[:HAS_PREFERENCE]->(pref)
    """
    
    try:
        tx.run(query,
               person_id=str(preference.person_id),
               id=str(preference.id),
               preference_type=preference.preference_type,
               importance=preference.importance,
               details=details,
               created_at=preference.created_at.isoformat(),
               embedding=preference.embedding
               )
        logger.debug(f"Created preference node for person {preference.person_id}")
    except Exception as e:
        logger.error(f"Failed to create preference node: {e}")
        raise

def get_person_id_by_name(tx, name):
    """
    Get person ID by their name from Neo4j database.
    
    Args:
        tx: Neo4j transaction object
        name (str): Name of the person to search for
        
    Returns:
        str: Person's ID if found, None otherwise
        
    Logs:
        DEBUG: Query execution details
        WARNING: When person is not found
    """
    logger.debug(f"Searching for person with name: {name}")
    
    query = """
    MATCH (p:Person {name: $name})
    RETURN p.id AS id
    """
    
    result = tx.run(query, name=name)
    record = result.single()
    
    if record is None:
        logger.warning(f"No person found with name: {name}")
        return None
        
    return record["id"]

def create_or_update_user(tx, user: User) -> User:
    """
    Create or update a user record in Neo4j and establish relationship with Person node if exists.
    
    Args:
        tx: Neo4j transaction
        user (User): User model instance
    
    Returns:
        User: Updated user instance
        
    Logs:
        DEBUG: Creation/update details
        INFO: When relationship is established
        WARNING: When person node is not found
    """
    logger.debug(f"Creating/updating user: {user.name}")
    
    # Create or update User node
    query = """
    MERGE (u:User {email: $email})
    ON CREATE SET 
        u.id = $id,
        u.name = $name,
        u.created_at = datetime($created_at)
    ON MATCH SET 
        u.name = $name,
        u.updated_at = datetime()
    """
    
    if user.person_id:
        # If person_id exists, create relationship with Person node
        query += """
        WITH u
        MATCH (p:Person {id: $person_id})
        MERGE (u)-[r:IS_PERSON]->(p)
        """
        
    query += "RETURN u {.*,u.embedding=null}"
    
    try:
        result = tx.run(
            query,
            email=user.main_email,
            id=str(user.id),
            name=user.name,
            created_at=user.created_at.isoformat(),
            person_id=str(user.person_id) if user.person_id else None
        )
        
        record = result.single()
        if not record:
            raise ValueError("Failed to create/update user")
            
        if user.person_id:
            logger.info(f"Established relationship between User {user.name} and Person node {user.person_id}")
        
        return user
        
    except Exception as e:
        logger.error(f"Error creating/updating user: {str(e)}")
        raise

def get_or_create_person_entity(tx, person_details: dict) -> PersonEntity:
    """
    Create or update a PersonEntity for the main user.
    
    Args:
        tx: Neo4j transaction
        person_details (dict): Person details from config
        
    Returns:
        PersonEntity: Created/updated person entity
        
    Logs:
        DEBUG: Creation details
        INFO: When person entity is created/updated
    """
    logger.debug(f"Creating/updating person entity with details: {person_details}")
    
    # Create PersonEntity instance
    person = PersonEntity(
        name=person_details["name"],
        type="Person",
        category=person_details["category"],
        nickname=person_details.get("nickname"),
        birth_date=person_details.get("birth_date"),
        email=person_details.get("email"),
        phone=person_details.get("phone"),
        address=person_details.get("address"),
        notes=person_details.get("notes")
    )
    
    try:
        # Use existing create_entity_node function
        db_id = create_entity_node(tx, person)
        person.db_id = db_id
        
        logger.info(f"Successfully created/updated person entity for {person.name}")
        return person
        
    except Exception as e:
        logger.error(f"Error creating/updating person entity: {str(e)}")
        raise

# Create vector indexes during application startup
def create_indexes():
    """Create necessary indexes in Neo4j."""
    embeddings_service.create_vector_indexes()
