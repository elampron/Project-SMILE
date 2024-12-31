"""
Neo4j service package.

This package provides functionality for interacting with the Neo4j database.
"""

from .driver import driver, get_driver, close_driver
from .schema import initialize_schema, initialize_schema_with_session
from .entities import (
    create_entity_node,
    get_or_create_person_entity,
)
from .relationships import (
    create_entity_relationship,
)
from .summaries import (
    create_summary_node,
    create_summary_relationships,
)
from .preferences import (
    create_preference_node,
    fetch_existing_preference_types,
)
from .memories import (
    create_cognitive_memory_node,
    get_similar_memories,
    get_memory_by_id,
    update_memory_validation,
    get_memories_by_type,
)
from .users import (
    create_or_update_user,
    get_user_by_email,
    get_person_id_by_name,
)
from .vectors import (
    create_vector_indexes,
    similarity_search,
)

__all__ = [
    'driver',
    'get_driver',
    'close_driver',
    'initialize_schema',
    'initialize_schema_with_session',
    'create_entity_node',
    'get_or_create_person_entity',
    'create_entity_relationship',
    'create_summary_node',
    'create_summary_relationships',
    'create_preference_node',
    'fetch_existing_preference_types',
    'create_cognitive_memory_node',
    'get_similar_memories',
    'get_memory_by_id',
    'update_memory_validation',
    'get_memories_by_type',
    'create_or_update_user',
    'get_user_by_email',
    'get_person_id_by_name',
    'create_vector_indexes',
    'similarity_search',
] 