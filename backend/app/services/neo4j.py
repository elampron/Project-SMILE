# app/services/neo4j.py

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.models.memory import CognitiveMemory, MemoryAssociation, ValidationStatus
from app.services.embeddings import EmbeddingsService
from uuid import UUID
try:
    from neo4j import GraphDatabase, Session, ManagedTransaction
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

def convert_datetime_fields(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert datetime objects to ISO format strings in a dictionary.
    
    Args:
        properties (Dict[str, Any]): Dictionary containing potential datetime fields
        
    Returns:
        Dict[str, Any]: Dictionary with datetime fields converted to ISO strings
    """
    datetime_fields = ['created_at', 'start_time', 'end_time']
    converted = properties.copy()
    
    for field in datetime_fields:
        if field in converted:
            value = converted[field]
            if isinstance(value, datetime):
                converted[field] = value.isoformat()
            elif isinstance(value, str):
                try:
                    # Verify it's a valid ISO format
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    converted[field] = value
                except ValueError:
                    # If invalid, use current time
                    converted[field] = datetime.utcnow().isoformat()
            else:
                converted[field] = datetime.utcnow().isoformat()
    
    return converted

def create_summary_node(tx, summary: ConversationSummary):
    """
    Create a summary node in Neo4j with embedding.
    
    Args:
        tx: Neo4j transaction object
        summary: ConversationSummary object to store
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
        
    query += "RETURN u"
    
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

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def convert_datetime_to_iso(obj):
    """Convert datetime objects in a dictionary to ISO format strings."""
    if isinstance(obj, dict):
        return {key: convert_datetime_to_iso(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_iso(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

class Neo4jEncoder(json.JSONEncoder):
    """Custom JSON encoder for Neo4j-compatible types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

def convert_properties_for_neo4j(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert property values to Neo4j-compatible formats.
    
    Args:
        properties: Dictionary of properties to convert
        
    Returns:
        Dict[str, Any]: Dictionary with converted values
        
    Converts:
    - UUID to str
    - datetime to ISO format string
    - Nested dictionaries to JSON strings
    - Complex objects to primitive types
    """
    converted = {}
    
    for key, value in properties.items():
        if value is None:
            continue
            
        if isinstance(value, UUID):
            converted[key] = str(value)
        elif isinstance(value, datetime):
            converted[key] = value.isoformat()
        elif isinstance(value, dict):
            # Convert nested dictionaries to JSON strings
            converted[key] = json.dumps(value, cls=Neo4jEncoder)
        elif isinstance(value, list):
            # Handle lists - keep primitive lists as is, convert complex ones
            if value and isinstance(value[0], (dict, UUID, datetime)):
                # List contains complex objects, convert each item
                converted_list = []
                for item in value:
                    if isinstance(item, dict):
                        converted_list.append(json.dumps(item, cls=Neo4jEncoder))
                    elif isinstance(item, UUID):
                        converted_list.append(str(item))
                    elif isinstance(item, datetime):
                        converted_list.append(item.isoformat())
                    else:
                        converted_list.append(item)
                converted[key] = converted_list
            else:
                # List of primitive values, keep as is
                converted[key] = value
        else:
            # Keep primitive values as is
            converted[key] = value
            
    return converted

def create_cognitive_memory_node(tx: ManagedTransaction, memory: CognitiveMemory) -> None:
    """
    Create a CognitiveMemory node in Neo4j and establish relationships with relevant entities.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
        memory (CognitiveMemory): Memory to create node for
        
    The function creates the following relationships:
    - (Memory)-[:BELONGS_TO]->(User)  # Memory ownership
    - (Memory)-[:MENTIONS]->(Entity)   # Entities mentioned in memory
    - (Memory)-[:INVOLVES]->(Person)   # People involved in the memory
    - (Memory)-[:RELATES_TO]->(Topic)  # Topics related to the memory
    """
    # Convert the memory to a dictionary and handle special fields
    properties = memory.model_dump()
    
    # Extract context before conversion for relationship creation
    context = properties.pop('context', {})
    
    # Convert remaining properties
    properties = convert_datetime_fields(properties)
    properties = convert_properties_for_neo4j(properties)
    
    # Create the memory node
    create_query = """
    CREATE (m:CognitiveMemory)
    SET m = $properties
    RETURN m {
        .*, 
        embedding: null
    } as m
    """
    memory_result = tx.run(create_query, properties=properties).single()
    
    if not memory_result:
        raise ValueError("Failed to create memory node")
        
    memory_node = memory_result['m']
    memory_id = memory_node.id  # This is Neo4j's internal node ID
    
    # Link memory to the user (assuming user's node exists)
    link_to_user_query = """
    MATCH (u:Person {type: 'Person', category: 'Human'})  // Find the user's node
    MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
    MERGE (m)-[:BELONGS_TO]->(u)
    """
    tx.run(link_to_user_query, memory_id=memory_id)
    
    # Extract and link to mentioned entities
    if 'entities' in context:
        for entity in context['entities']:
            # Convert entity properties to primitive types
            entity_props = {
                'name': entity.get('name'),
                'type': entity.get('type'),
                'source': entity.get('source', 'memory')
            }
            # Try to find existing entity or create new one
            entity_query = """
            MERGE (e:Entity {name: $name, type: $type})
            ON CREATE SET e += $properties
            WITH e
            MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
            MERGE (m)-[:MENTIONS]->(e)
            """
            tx.run(
                entity_query,
                name=entity_props['name'],
                type=entity_props['type'],
                properties=entity_props,
                memory_id=memory_id
            )
    
    # Link to people involved
    if 'participants' in context:
        for person in context['participants']:
            # Convert person properties to primitive types
            person_props = {
                'name': person.get('name'),
                'role': person.get('role'),
                'source': person.get('source', 'memory')
            }
            person_query = """
            MERGE (p:Person {name: $name})
            ON CREATE SET p += $properties
            WITH p
            MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
            MERGE (m)-[:INVOLVES]->(p)
            """
            tx.run(
                person_query,
                name=person_props['name'],
                properties=person_props,
                memory_id=memory_id
            )
    
    # Link to topics
    if 'topics' in context:
        for topic in context['topics']:
            topic_query = """
            MERGE (t:Topic {name: $topic})
            WITH t
            MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
            MERGE (m)-[:RELATES_TO]->(t)
            """
            tx.run(topic_query, topic=topic, memory_id=memory_id)
    
    # Link to location if present
    if 'location' in context:
        location_query = """
        MERGE (l:Location {name: $location})
        WITH l
        MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
        MERGE (m)-[:OCCURRED_AT]->(l)
        """
        tx.run(
            location_query,
            location=context['location'],
            memory_id=memory_id
        )
    
    # Link to related memories based on shared topics and entities
    if 'topics' in context or 'entities' in context:
        related_memories_query = """
        MATCH (m:CognitiveMemory) WHERE id(m) = $memory_id
        MATCH (other:CognitiveMemory) WHERE id(other) <> $memory_id
        WITH m, other
        OPTIONAL MATCH (m)-[:RELATES_TO]->(t:Topic)<-[:RELATES_TO]-(other)
        WITH m, other, COUNT(t) as shared_topics
        OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other)
        WITH m, other, shared_topics, COUNT(e) as shared_entities
        WITH m, other, shared_topics + shared_entities as similarity_score
        WHERE similarity_score > 0
        MERGE (m)-[r:RELATED_TO]->(other)
        SET r.similarity = toFloat(similarity_score)
        """
        tx.run(related_memories_query, memory_id=memory_id)

def get_similar_memories(tx: ManagedTransaction, text: str, limit: int = 5) -> List[CognitiveMemory]:
    """
    Get memories similar to the input text based on shared topics and entities.
    
    Args:
        tx: Neo4j transaction
        text: Input text to find similar memories for
        limit: Maximum number of memories to return
        
    Returns:
        List[CognitiveMemory]: List of similar memories
    """
    # For now, return recent memories with high importance
    query = """
    MATCH (m:CognitiveMemory)
    WHERE m.importance >= 0.7
    RETURN m {
        .*, 
        embedding: null
    } as m
    ORDER BY m.created_at DESC
    LIMIT $limit
    """
    result = tx.run(query, limit=limit)
    return [record["m"] for record in result]

def get_memory_by_id(tx, memory_id: UUID) -> Optional[CognitiveMemory]:
    """
    Retrieve a specific memory by ID.
    
    Args:
        tx: Neo4j transaction object
        memory_id: UUID of the memory to retrieve
        
    Returns:
        Optional[CognitiveMemory]: Retrieved memory or None if not found
    """
    query = """
    MATCH (m:CognitiveMemory {id: $memory_id})
    RETURN m {
        .*, 
        embedding: null
    } as m
    """
    
    result = tx.run(query, memory_id=str(memory_id))
    record = result.single()
    
    if not record:
        return None
        
    memory_dict = dict(record["m"])
    # Handle special fields
    memory_dict['semantic'] = json.loads(memory_dict['semantic'])
    memory_dict['temporal'] = json.loads(memory_dict['temporal'])
    memory_dict['validation'] = json.loads(memory_dict['validation'])
    memory_dict['structured_data'] = json.loads(memory_dict['structured_data']) if memory_dict['structured_data'] else None
    
    return CognitiveMemory(**memory_dict)

def update_memory_validation(tx, memory_id: UUID, validation: ValidationStatus):
    """
    Update the validation status of a memory.
    
    Args:
        tx: Neo4j transaction object
        memory_id: UUID of the memory to update
        validation: New validation status
    """
    query = """
    MATCH (m:CognitiveMemory {id: $memory_id})
    SET m.validation = $validation,
        m.updated_at = datetime()
    """
    
    tx.run(query,
           memory_id=str(memory_id),
           validation=json.dumps(validation.model_dump()))

def create_memory_indexes(tx):
    """
    Create necessary indexes for cognitive memory nodes.
    """
    # Create unique constraint on id
    tx.run("CREATE CONSTRAINT cognitive_memory_id IF NOT EXISTS FOR (m:CognitiveMemory) REQUIRE m.id IS UNIQUE")
    
    # Create index on type for faster type-based queries
    tx.run("CREATE INDEX cognitive_memory_type IF NOT EXISTS FOR (m:CognitiveMemory) ON (m.type)")
    
    # Create vector index for embeddings
    tx.run("""
    CALL db.index.vector.createNodeIndex(
        'memory_embeddings',
        'CognitiveMemory',
        'embedding',
        1536,
        'cosine'
    )
    """)

def get_memories_by_type(tx, memory_type: str, limit: int = 10) -> List[CognitiveMemory]:
    """
    Retrieve memories of a specific type.
    """
    query = """
    MATCH (m:CognitiveMemory {type: $type})
    RETURN m {
        .*, // all properties
        embedding: null  // explicitly set embedding to null
    } as m
    ORDER BY m.created_at DESC
    LIMIT $limit
    """
    
    results = tx.run(query, type=memory_type, limit=limit)
    
    memories = []
    for record in results:
        memory_dict = dict(record["m"])
        # Handle special fields
        memory_dict['semantic'] = json.loads(memory_dict['semantic'])
        memory_dict['temporal'] = json.loads(memory_dict['temporal'])
        memory_dict['validation'] = json.loads(memory_dict['validation'])
        memory_dict['structured_data'] = json.loads(memory_dict['structured_data']) if memory_dict['structured_data'] else None
        
        memory = CognitiveMemory(**memory_dict)
        memories.append(memory)
    
    return memories

def initialize_cognitive_memory_schema(tx):
    """
    Initialize Neo4j schema for cognitive memories.
    Creates necessary indexes and constraints.
    This function should not be called directly - use initialize_schema() instead.
    """
    # Only drop the constraint in this transaction
    tx.run("DROP CONSTRAINT cognitive_memory_id IF EXISTS")

def check_if_index_exists(tx, index_name):
    """
    Check if an index with the given name already exists.
    
    Args:
        tx: Neo4j transaction object
        index_name: Name of the index to check
        
    Returns:
        bool: True if index exists, False otherwise
    """
    query = """
    SHOW INDEXES
    YIELD name, type
    WHERE name = $index_name
    RETURN count(*) > 0 as exists
    """
    result = tx.run(query, index_name=index_name)
    return result.single()["exists"]

def initialize_schema_with_session(session: Session) -> None:
    """
    Initialize the complete schema using a single session but separate transactions.
    Checks for existing indexes before creating them.
    
    Args:
        session: Neo4j session object
    """
    logger.debug("Starting schema initialization")
    
    try:
        # Check and create regular index if it doesn't exist
        session.execute_write(lambda tx: tx.run(
            """
            CREATE INDEX cognitive_memory_type IF NOT EXISTS 
            FOR (m:CognitiveMemory) ON (m.type)
            """
        ))
        logger.debug("Regular index check/creation completed")
        
        # Check if vector index exists before trying to create it
        vector_index_exists = session.execute_read(
            lambda tx: check_if_index_exists(tx, "memory_embeddings")
        )
        
        if not vector_index_exists:
            logger.debug("Vector index does not exist, creating it...")
            try:
                session.execute_write(lambda tx: tx.run(
                    """
                    CALL db.index.vector.createNodeIndex(
                        'memory_embeddings',
                        'CognitiveMemory',
                        'embedding',
                        1536,
                        'cosine'
                    )
                    """
                ))
                logger.info("Vector index created successfully")
            except Exception as e:
                if "EquivalentSchemaRuleAlreadyExistsException" in str(e):
                    logger.debug("Vector index already exists (race condition)")
                else:
                    logger.error(f"Error creating vector index: {str(e)}")
                    raise
        else:
            logger.debug("Vector index already exists, skipping creation")
        
        logger.info("Successfully completed schema initialization")
        
    except Exception as e:
        logger.error(f"Error initializing schema: {str(e)}")
        raise

def initialize_schema():
    """
    Initialize the complete schema using a new session.
    This function should be called during application startup.
    """
    with driver.session() as session:
        initialize_schema_with_session(session)

# Initialize schema when the module is loaded
try:
    initialize_schema()
    logger.info("Schema initialized successfully on module load")
except Exception as e:
    logger.error(f"Failed to initialize schema on module load: {str(e)}")

def cleanup_and_create_constraint(tx):
    """
    Clean up duplicate nodes and create unique constraint.
    Must be run after schema initialization.
    
    Args:
        tx: Neo4j transaction object
    """
    logger.debug("Cleaning up duplicates and creating constraint")
    
    # Clean up duplicates
    result = tx.run("""
        MATCH (m:CognitiveMemory)
        WITH m.id as id, collect(m) as nodes
        WHERE size(nodes) > 1
        UNWIND tail(nodes) as duplicate
        DETACH DELETE duplicate
        RETURN count(duplicate) as deleted_count
    """)
    
    deleted_count = result.single()["deleted_count"]
    if deleted_count > 0:
        logger.info(f"Removed {deleted_count} duplicate CognitiveMemory nodes")
    
    # Now create the constraint
    tx.run("""
        CREATE CONSTRAINT cognitive_memory_id IF NOT EXISTS 
        FOR (m:CognitiveMemory) REQUIRE m.id IS UNIQUE
    """)

def exclude_embedding_from_properties(properties: dict) -> dict:
    """
    Remove embedding from node properties.
    
    Args:
        properties (dict): Node properties dictionary
        
    Returns:
        dict: Properties without embedding
    """
    if properties and 'embedding' in properties:
        del properties['embedding']
    return properties