"""
Neo4j schema initialization and management module.

This module handles the creation and maintenance of Neo4j database schema,
including constraints and indexes.
"""

from app.utils.logger import logger
from neo4j import Session, ManagedTransaction
from .driver import driver
from .vectors import create_vector_indexes


def create_constraints(tx: ManagedTransaction) -> None:
    """
    Create Neo4j constraints.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
    """
    try:
        # Create constraints for Person nodes
        tx.run("""
        CREATE CONSTRAINT person_name_category IF NOT EXISTS
        FOR (p:Person) REQUIRE (p.name, p.category) IS UNIQUE
        """)
        
        # Create constraints for Document nodes
        tx.run("""
        CREATE CONSTRAINT document_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE (d.name, d.doc_type, d.file_path) IS UNIQUE
        """)
        
        # Create constraints for Summary nodes
        tx.run("""
        CREATE CONSTRAINT summary_id IF NOT EXISTS
        FOR (s:Summary) REQUIRE s.id IS UNIQUE
        """)
        
        # Create constraints for CognitiveMemory nodes
        tx.run("""
        CREATE CONSTRAINT memory_id IF NOT EXISTS
        FOR (m:CognitiveMemory) REQUIRE m.id IS UNIQUE
        """)
        
        # Create constraints for Preference nodes
        tx.run("""
        CREATE CONSTRAINT preference_id IF NOT EXISTS
        FOR (p:Preference) REQUIRE p.id IS UNIQUE
        """)
        
        logger.info("Created Neo4j constraints successfully")
    except Exception as e:
        logger.warning(f"Error creating constraints: {str(e)}")

def create_indexes(tx: ManagedTransaction) -> None:
    """
    Create Neo4j indexes.
    
    Args:
        tx (ManagedTransaction): Neo4j transaction
    """
    try:
        # Create vector index for embeddings
        tx.run("""
        CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS
        FOR (n:CognitiveMemory)
        ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """)
        
        # Create index for memory type
        tx.run("""
        CREATE INDEX memory_type_index IF NOT EXISTS
        FOR (n:CognitiveMemory)
        ON (n.memory_type)
        """)
        
        logger.info("Created Neo4j indexes successfully")
    except Exception as e:
        logger.warning(f"Error creating indexes: {str(e)}")

def initialize_schema_with_session(session: Session) -> None:
    """
    Initialize Neo4j schema with the provided session.
    
    Args:
        session (Session): Neo4j session
    """
    try:
        # Create constraints in one transaction
        try:
            with session.begin_transaction() as tx:
                create_constraints(tx)
                tx.commit()  # Explicitly commit the transaction
                logger.info("Created constraints successfully")
        except Exception as e:
            logger.warning(f"Error during constraint creation: {str(e)}")
            # Continue with other operations even if constraints fail
            
        # Create regular indexes in another transaction
        try:
            with session.begin_transaction() as tx:
                create_indexes(tx)
                tx.commit()  # Explicitly commit the transaction
                logger.info("Created indexes successfully")
        except Exception as e:
            logger.warning(f"Error during index creation: {str(e)}")
            # Continue with other operations even if indexes fail
            
        # Create vector indexes in a final transaction
        try:
            with session.begin_transaction() as tx:
                create_vector_indexes(tx)
                tx.commit()  # Explicitly commit the transaction
                logger.info("Created vector indexes successfully")
        except Exception as e:
            logger.warning(f"Error during vector index creation: {str(e)}")
            # Continue with other operations even if vector indexes fail
            
        logger.info("Neo4j schema initialization completed")
    except Exception as e:
        logger.error(f"Error during schema initialization: {str(e)}")
        # Don't raise the error to allow the application to start
        # Schema elements that failed to create will be retried on next startup

def initialize_schema() -> None:
    """
    Initialize Neo4j schema using a new session.
    This is the main entry point for schema initialization.
    """
    try:
        with driver.session() as session:
            initialize_schema_with_session(session)
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j schema: {str(e)}")
        raise 