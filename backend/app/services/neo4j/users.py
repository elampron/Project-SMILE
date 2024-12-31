"""
User-related operations for Neo4j.

This module handles the creation and management of user nodes
and their relationships in the Neo4j database.
"""

import logging
from typing import Optional
from neo4j import ManagedTransaction
from app.models.agents import User
from .utils import convert_properties_for_neo4j
from datetime import datetime
from uuid import UUID

# Initialize logger
logger = logging.getLogger(__name__)

def get_person_id_by_name(tx: ManagedTransaction, name: str) -> Optional[str]:
    """
    Get the ID of a person node by their name.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        name (str): Name of the person to find
        
    Returns:
        Optional[str]: The ID of the person if found, None otherwise
    """
    query = """
    MATCH (p:Person {name: $name})
    RETURN p {.*, embedding: null} as p, p.id as id
    """
    
    try:
        result = tx.run(query, name=name)
        record = result.single()
        return record["id"] if record else None
    except Exception as e:
        logger.error(f"Error getting person ID by name: {str(e)}")
        raise

def get_user_by_email(tx: ManagedTransaction, email: str) -> Optional[User]:
    """
    Get a user by their email address.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        email (str): Email address to search for
        
    Returns:
        Optional[User]: The user if found, None otherwise
    """
    query = """
    MATCH (u:User {main_email: $email})
    OPTIONAL MATCH (u)-[:IS]->(p:Person)
    RETURN u {.*, embedding: null} as u, p.id as person_id
    """
    
    try:
        result = tx.run(query, email=email)
        record = result.single()
        if record:
            user_data = dict(record["u"].items())
            
            # Convert Neo4j DateTime objects to Python datetime
            for field in ['created_at', 'updated_at']:
                if field in user_data:
                    if hasattr(user_data[field], 'to_native'):
                        # Neo4j DateTime object
                        user_data[field] = user_data[field].to_native()
                    elif isinstance(user_data[field], str):
                        # ISO format string
                        user_data[field] = datetime.fromisoformat(user_data[field].replace('Z', '+00:00'))
                    # If it's already a datetime object, leave it as is
            
            # Convert UUID strings to UUID objects
            if 'id' in user_data and isinstance(user_data['id'], str):
                user_data['id'] = UUID(user_data['id'])
            if 'person_id' in user_data and isinstance(user_data['person_id'], str):
                user_data['person_id'] = UUID(user_data['person_id'])
                
            user = User(**user_data)
            if record["person_id"]:
                user.person_id = record["person_id"]
            return user
        return None
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        raise

def create_or_update_user(tx: ManagedTransaction, user: User) -> User:
    """
    Create or update a user in Neo4j.
    Also creates a corresponding Person node and relationship if they don't exist.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        user (User): User object to store
        
    Returns:
        User: Updated user object with person_id if available
    """
    # Convert user to dictionary and prepare properties
    properties = user.model_dump()
    properties['id'] = str(properties['id'])
    
    # Build the query using MERGE to avoid duplicates
    query = """
    MERGE (u:User {main_email: $main_email})
    ON CREATE SET
        u.id = $id,
        u.name = $name,
        u.created_at = datetime($created_at)
    ON MATCH SET
        u.name = $name
    RETURN u {.*, embedding: null} as u
    """
    
    try:
        # Execute the query
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            # Create corresponding Person node if it doesn't exist
            # Generate embedding for the person
            from app.services.embeddings import EmbeddingsService
            embeddings_service = EmbeddingsService()
            text_for_embedding = f"{user.name} User account for {user.name}"
            embedding = embeddings_service.generate_embedding(text_for_embedding)
            
            person_query = """
            MERGE (p:Person {name: $name, category: 'user'})
            ON CREATE SET
                p.id = $id,
                p.notes = $notes,
                p.type = 'Person',
                p.created_at = datetime($created_at),
                p.embedding = $embedding
            ON MATCH SET
                p.notes = $notes,
                p.embedding = $embedding
            RETURN p {.*, embedding: null} as p, p.id as person_id
            """
            person_result = tx.run(
                person_query,
                name=user.name,
                id=str(user.id),
                notes=f"User account for {user.name}",
                created_at=datetime.utcnow().isoformat(),
                embedding=embedding
            )
            person_record = person_result.single()
            if person_record:
                user.person_id = person_record["person_id"]
            
            # Create relationship between User and Person nodes
            relationship_query = """
            MATCH (u:User {main_email: $main_email}), (p:Person {name: $name})
            MERGE (u)-[:IS]->(p)
            """
            tx.run(relationship_query, main_email=user.main_email, name=user.name)
            
            logger.info(f"Successfully created/updated user {user.name} with ID: {user.id}")
            return user
        else:
            raise Exception("Failed to create/update user")
    except Exception as e:
        logger.error(f"Error creating/updating user: {str(e)}")
        raise 