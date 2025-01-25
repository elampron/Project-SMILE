"""
Context management service for retrieving and formatting context.
"""

from typing import Any, Dict, List, Optional
from neo4j import Driver
from app.utils.logger import logger

class ContextManager:
    """
    Class for managing context retrieval and formatting.
    """
    
    def __init__(self, driver: Driver):
        """
        Initialize ContextManager with a Neo4j driver.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.logger = logger
        
    def get_formatted_context(self, user_input: str) -> str:
        """
        Get formatted context based on user input.
        
        Args:
            user_input: The user's current question or comment
            
        Returns:
            str: Formatted context string
            
        Raises:
            Exception: If context retrieval fails
        """
        try:
            # Get relevant nodes from Neo4j
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.content CONTAINS $input
                    RETURN n.content AS content
                    LIMIT 5
                """, input=user_input)
                
                # Format the context
                context_parts = []
                for record in result:
                    context_parts.append(record["content"])
                    
                if not context_parts:
                    return "No relevant context found."
                    
                formatted_context = "\n\n".join(context_parts)
                self.logger.info(f"Found {len(context_parts)} context parts")
                return formatted_context
                
        except Exception as e:
            self.logger.error(f"Error getting formatted context: {str(e)}")
            raise 