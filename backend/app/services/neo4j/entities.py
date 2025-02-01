"""
Entities operations for Neo4j.

This module handles the creation and management of entity nodes
in the Neo4j database.
"""
import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from neo4j import ManagedTransaction
from app.services.embeddings import EmbeddingsService
from .utils import convert_properties_for_neo4j

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize embeddings service
embeddings_service = EmbeddingsService()

def create_entity_node(tx: ManagedTransaction, entity: Any) -> str:
    """
    Create or update an entity node in Neo4j with embedding.
    Uses MERGE to avoid duplicates, matching on name and type.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction object
        entity: Entity object or dictionary to store
        
    Returns:
        str: The ID of the created/updated node
    """
    # Import here to avoid circular dependency
    from app.models.memory import PersonEntity
    
    # Handle dictionary input
    if isinstance(entity, dict):
        properties = entity.copy()
        entity_type = properties.get('type', 'Document')
        
        # Generate embedding if not provided
        if 'embedding' not in properties or properties['embedding'] is None:
            # Create text representation for embedding
            if entity_type == 'Document':
                text_for_embedding = f"{properties.get('name', '')} {properties.get('content', '')} {properties.get('summary', '')}"
                if properties.get('topics'):
                    text_for_embedding += f"\nTopics: {', '.join(properties['topics'])}"
                if properties.get('entities'):
                    text_for_embedding += f"\nEntities: {', '.join(properties['entities'])}"
                if properties.get('tags'):
                    text_for_embedding += f"\nTags: {', '.join(properties['tags'])}"
            else:
                text_for_embedding = f"{properties.get('name', '')} {entity_type} {properties.get('notes', '')}"
            properties['embedding'] = embeddings_service.generate_embedding(text_for_embedding)
    else:
        # Generate embedding if not provided
        if entity.embedding is None:
            text_for_embedding = f"{entity.name} {entity.type} {entity.notes if hasattr(entity, 'notes') else ''}"
            entity.embedding = embeddings_service.generate_embedding(text_for_embedding)
        
        # Convert entity to dictionary and prepare properties
        properties = entity.model_dump()
        entity_type = entity.type
    
    # Ensure ID is string
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
    """ % entity_type
    
    # Add category for Person entities
    if isinstance(entity, PersonEntity) or (isinstance(entity, dict) and entity_type == 'Person'):
        query += ", e.category = $category"
    
    # Add additional fields that should be updated on match
    query += """
    ON MATCH SET
        e.embedding = $embedding,
        e.updated_at = datetime($created_at)
    """
    
    # Add optional fields if they exist and are not null
    optional_fields = ['notes', 'nickname', 'birth_date', 'email', 'phone', 'address', 
                      'industry', 'website', 'metadata', 'content', 'summary', 'topics',
                      'entities', 'tags', 'file_path', 'file_url', 'file_type', 'doc_type']
    for field in optional_fields:
        if field in properties and properties[field]:
            query += f", e.{field} = ${field}"
    
    query += "\nRETURN e {.*, embedding: null} as e"
    
    try:
        # Execute the query
        result = tx.run(query, **properties)
        record = result.single()
        if record:
            if not isinstance(entity, dict):
                entity.db_id = record["e"]["id"]  # Set the db_id on the entity
            logger.info(f"Successfully created/updated entity node with name: {properties['name']}")
            return record["e"]["id"]
        else:
            raise Exception("Failed to create/update entity node")
    except Exception as e:
        logger.error(f"Error creating/updating entity node: {str(e)}")
        raise

def get_or_create_person_entity(tx: ManagedTransaction, person_details: Dict[str, Any]) -> Any:
    """
    Get or create a person entity in Neo4j.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        person_details (Dict[str, Any]): Dictionary containing person details
        
    Returns:
        PersonEntity: The created or retrieved person entity
    """
    # Import here to avoid circular dependency
    from app.models.memory import PersonEntity
    
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

def get_nodes(tx: ManagedTransaction,
            label: Optional[str] = None,
            search_term: Optional[str] = None,
            limit: int = 50,
            offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve nodes from Neo4j with optional filtering and pagination.

    Args:
        tx (ManagedTransaction): Neo4j transaction object
        label (Optional[str]): Node label to filter by (e.g., 'Person', 'Document')
        search_term (Optional[str]): Term to search in node properties
        limit (int): Maximum number of nodes to return (default: 50)
        offset (int): Number of nodes to skip (default: 0)

    Returns:
        List[Dict[str, Any]]: List of matching nodes with their properties
    """
    # Start building the query
    query = "MATCH (n"
    
    # Add label filter if provided
    if label:
        query += f":{label}"
    query += ")"

    # Add search term filter if provided
    if search_term:
        query += """
        WHERE n.name =~ $search_regex
        OR n.notes =~ $search_regex
        OR n.content =~ $search_regex
        """

    # Add return statement with pagination
    query += """
    RETURN n {.*, embedding: null} as node
    SKIP $offset
    LIMIT $limit
    """

    # Prepare parameters
    params = {
        "offset": offset,
        "limit": limit,
        "search_regex": f"(?i).*{search_term}.*" if search_term else None
    }

    # Execute query and return results
    result = tx.run(query, params)
    return [record["node"] for record in result]