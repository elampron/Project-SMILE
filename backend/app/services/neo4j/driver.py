"""
Neo4j driver initialization and management module.

This module is responsible for creating and managing the Neo4j driver instance
that will be used across the application.
"""

import logging
from neo4j import GraphDatabase
from app.configs.settings import settings

from app.utils.logger import logger

driver = GraphDatabase.driver(
    settings.app_config.get("neo4j_config").get("uri"),
    auth=(
        settings.app_config.get("neo4j_config").get("username"), 
        settings.app_config.get("neo4j_config").get("password")
    )
)

def get_driver():
    """
    Get the Neo4j driver instance.
    
    Returns:
        neo4j.Driver: The Neo4j driver instance
    """
    return driver

def close_driver():
    """
    Close the Neo4j driver connection.
    Should be called when shutting down the application.
    """
    try:
        driver.close()
        logger.info("Neo4j driver connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver connection: {str(e)}")
        raise 