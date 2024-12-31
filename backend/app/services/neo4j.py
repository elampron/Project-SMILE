"""
Neo4j service module.

This module re-exports the functionality from the modular Neo4j services.
It maintains backward compatibility while using the new modular structure.
"""

from .neo4j.driver import driver, get_driver, close_driver
from .neo4j.schema import (
    initialize_schema,
    initialize_schema_with_session,
    cleanup_and_create_constraint,
    create_memory_indexes,
)
from .neo4j.entities import (
    create_entity_node,
    get_or_create_person_entity,
)
from .neo4j.relationships import create_entity_relationship
from .neo4j.summaries import (
    create_summary_node,
    create_summary_relationships,
)
from .neo4j.preferences import (
    create_preference_node,
    fetch_existing_preference_types,
)
from .neo4j.memories import (
    create_cognitive_memory_node,
    get_similar_memories,
    get_memory_by_id,
    update_memory_validation,
    get_memories_by_type,
)
from .neo4j.users import (
    create_or_update_user,
    get_person_id_by_name,
)
from .neo4j.utils import (
    convert_datetime_fields,
    convert_properties_for_neo4j,
    exclude_embedding_from_properties,
    DateTimeEncoder,
    Neo4jEncoder,
)

__all__ = [
    'driver',
    'get_driver',
    'close_driver',
    'initialize_schema',
    'initialize_schema_with_session',
    'cleanup_and_create_constraint',
    'create_memory_indexes',
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
    'get_person_id_by_name',
    'convert_datetime_fields',
    'convert_properties_for_neo4j',
    'exclude_embedding_from_properties',
    'DateTimeEncoder',
    'Neo4jEncoder',
] 