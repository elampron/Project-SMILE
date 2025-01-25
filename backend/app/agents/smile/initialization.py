"""
Initialization module for the SMILE agent.

This module contains all initialization and cleanup related functionality.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver
from app.services.neo4j import create_or_update_user, get_user_by_email
from app.models.agents import User
from app.utils.logger import logger

if TYPE_CHECKING:
    from app.agents.smile.agent import Smile

def initialize_main_user(agent: Any) -> None:
    """
    Initialize the main user from config and persist to Neo4j.
    Creates or updates both User and PersonEntity records.
    
    Args:
        agent: The Smile agent instance
        
    Raises:
        ValueError: If main user configuration is missing
    """
    try:
        # Get main user config
        user_config = agent.settings.app_config.get("main_user")
        langchain_config = agent.settings.app_config.get("langchain_config")
        if not user_config:
            raise ValueError("Main user configuration is missing in app_config.yaml")

        # Create User instance
        agent.main_user = User(
            name=user_config["name"],
            main_email=user_config["main_email"]
        )
        agent.thread_id = langchain_config["thread_id"]

        # Persist user to Neo4j and get updated user with person_id
        with agent.driver.session() as session:
            # First check if user exists
            existing_user = session.execute_read(
                get_user_by_email,
                agent.main_user.main_email
            )
            
            if existing_user:
                agent.main_user = User(**existing_user)
                logger.info(f"Using existing main user: {agent.main_user.name}")
                return

            # Create/update user record only if it doesn't exist
            # This will also create the corresponding Person node
            agent.main_user = session.execute_write(
                create_or_update_user, 
                agent.main_user
            )
            
            logger.info(f"Main user initialized: {agent.main_user.name}")
            
    except Exception as e:
        logger.error(f"Failed to initialize main user: {str(e)}")
        raise

def initialize_postgres(agent: Any) -> None:
    """
    Initialize PostgreSQL connection and checkpointer.
    
    Args:
        agent: The Smile agent instance
        
    Raises:
        Exception: If connection fails
    """
    try:
        logger.info(f"Connecting to PostgreSQL at {agent.postgres_url}...")
        
        # Create direct connection with optimized settings
        conn = Connection.connect(
            agent.postgres_url,
            autocommit=True,
            prepare_threshold=None,  # Disable prepared statements
            options="-c synchronous_commit=off"  # Optimize for performance
        )
        
        # Create saver with the connection and use it directly as checkpointer
        agent._checkpointer = PostgresSaver(conn)
        
        # Setup tables using the checkpointer
        agent._checkpointer.setup()
        logger.info("PostgreSQL tables created successfully")
        logger.info("Successfully connected to PostgreSQL")
        
    except Exception as pg_error:
        logger.error(
            "Failed to connect to PostgreSQL. Please ensure:\n"
            "1. PostgreSQL is running\n"
            "2. The connection string in app_config.yaml is correct\n"
            "3. If running locally, use 'localhost' instead of 'postgres' as the host\n"
            f"Error details: {str(pg_error)}"
        )
        raise

def initialize_directories(agent: Any) -> None:
    """
    Initialize required directories for the agent.
    
    Args:
        agent: The Smile agent instance
    """
    # Create attachments directory if it doesn't exist
    agent.attachments_dir = Path("library/attachments")
    agent.attachments_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized attachments directory at {agent.attachments_dir}")

def cleanup(agent: Any) -> None:
    """
    Clean up resources used by the agent.
    
    Args:
        agent: The Smile agent instance
    """
    try:
        if hasattr(agent, '_checkpointer') and hasattr(agent._checkpointer, 'conn'):
            agent._checkpointer.conn.close()
            logger.info("Successfully closed PostgreSQL connection")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise 