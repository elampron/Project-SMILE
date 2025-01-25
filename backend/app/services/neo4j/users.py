"""
User operations for Neo4j.

This module handles the creation and management of user nodes
in the Neo4j database.
"""

import logging
from typing import Optional, Dict, Any
from neo4j import ManagedTransaction
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

def create_or_update_user(tx: ManagedTransaction, user: Any) -> Any:
    """
    Create or update a user node in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        user: User object containing the details
        
    Returns:
        User: The created/updated user instance
    """
    # Import here to avoid circular dependency
    from app.models.agents import User
    
    if not isinstance(user, User):
        raise ValueError("user must be a User instance")
    
    # Convert user to dictionary
    properties = user.model_dump()
    properties = convert_properties_for_neo4j(properties)
    
    # Build query
    query = """
    MERGE (u:User {main_email: $main_email})
    ON CREATE SET
        u.id = $id,
        u.name = $name,
        u.created_at = datetime($created_at)
    ON MATCH SET
        u.name = $name,
        u.updated_at = datetime()
    RETURN u {
        .id,
        .name,
        .main_email,
        created_at: toString(u.created_at),
        updated_at: CASE WHEN u.updated_at IS NOT NULL 
                   THEN toString(u.updated_at)
                   ELSE null END
    } as user
    """
    
    try:
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            user_data = record["user"]
            logger.info(f"Created/updated user: {user.main_email}")
            return User(**user_data)
        else:
            raise Exception("Failed to create/update user")
    except Exception as e:
        logger.error(f"Error creating/updating user: {str(e)}")
        raise

def get_user_by_email(tx: ManagedTransaction, email: str) -> Optional[Dict[str, Any]]:
    """
    Get a user by email.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        email (str): Email of the user to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: User data if found
    """
    query = """
    MATCH (u:User {main_email: $email})
    RETURN u {
        .id,
        .name,
        .main_email,
        created_at: toString(u.created_at),
        updated_at: CASE WHEN u.updated_at IS NOT NULL 
                   THEN toString(u.updated_at)
                   ELSE null END
    } as user
    """
    
    try:
        result = tx.run(query, email=email)
        record = result.single()
        return record["user"] if record else None
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        raise

def get_person_id_by_name(tx: ManagedTransaction, name: str) -> Optional[str]:
    """
    Get a person's ID by their name.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        name (str): Name of the person
        
    Returns:
        Optional[str]: Person's ID if found
    """
    query = """
    MATCH (p:Person {name: $name})
    RETURN p.id as id
    """
    
    try:
        result = tx.run(query, name=name)
        record = result.single()
        return record["id"] if record else None
    except Exception as e:
        logger.error(f"Error getting person ID: {str(e)}")
        raise 