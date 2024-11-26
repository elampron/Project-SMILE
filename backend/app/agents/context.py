"""
Context management module for SMILE agent.

This module is responsible for gathering, organizing, and formatting different types of contextual 
information to be injected into the prompt template. It acts as a central place for context 
management with methods to handle different types of context (preferences, entities, facts, 
summaries) and format them in a way that's optimal for LLM consumption.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase, Session

from app.models.memory import Preference, PersonEntity, OrganizationEntity
from app.configs.settings import settings

# Initialize logger
logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context preparation and formatting for the SMILE agent.
    
    This class is responsible for:
    1. Gathering context from different sources (Neo4j, state, etc.)
    2. Filtering and organizing context based on relevance and importance
    3. Formatting context in a way that's optimal for LLM consumption
    
    Attributes:
        driver: Neo4j driver instance for database operations
    """
    
    def __init__(self, driver: GraphDatabase.driver):
        """
        Initialize the ContextManager.
        
        Args:
            driver: Neo4j driver instance for database operations
        """
        self.driver = driver
        
    def get_formatted_context(self, user_input: str) -> str:
        """
        Master method to gather and format all context for the LLM.
        
        Args:
            user_input: The user's current question or comment
            
        Returns:
            str: Formatted context string ready for LLM consumption
            
        Logs:
            DEBUG: Context gathering process details
            ERROR: When context gathering fails
        """
        try:
            logger.debug(f"Starting context gathering for input: {user_input}")
            
            # Gather different types of context
            important_preferences = self._get_important_preferences()
            relevant_preferences = self._get_relevant_preferences(user_input)
            relevant_entities = self._get_relevant_entities(user_input)
            important_facts = self._get_important_facts()
            relevant_facts = self._get_relevant_facts(user_input)
            relevant_summaries = self._get_relevant_summaries(user_input)
            
            # Format the context
            context_parts = []
            
            if important_preferences:
                context_parts.append("IMPORTANT PREFERENCES:")
                context_parts.extend(self._format_preferences(important_preferences))
            
            if relevant_preferences:
                context_parts.append("\nRELEVANT PREFERENCES:")
                context_parts.extend(self._format_preferences(relevant_preferences))
            
            if relevant_entities:
                context_parts.append("\nRELEVANT ENTITIES:")
                context_parts.extend(self._format_entities(relevant_entities))
            
            if important_facts:
                context_parts.append("\nIMPORTANT FACTS:")
                context_parts.extend(self._format_facts(important_facts))
            
            if relevant_facts:
                context_parts.append("\nRELEVANT FACTS:")
                context_parts.extend(self._format_facts(relevant_facts))
            
            if relevant_summaries:
                context_parts.append("\nRELEVANT SUMMARIES:")
                context_parts.extend(self._format_summaries(relevant_summaries))
            
            formatted_context = "\n".join(context_parts)
            logger.debug(f"Formatted context: {formatted_context}")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error gathering context: {str(e)}")
            return ""
    
    def _get_important_preferences(self) -> List[Preference]:
        """Get all preferences with importance = 5."""
        query = """
        MATCH (p:Preference)
        WHERE p.importance = 5
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [self._neo4j_to_preference(record["p"]) for record in result]
    
    def _get_relevant_preferences(self, user_input: str) -> List[Preference]:
        """Get preferences relevant to the user's input using semantic search."""
        # TODO: Implement semantic search for preferences
        return []
    
    def _get_relevant_entities(self, user_input: str) -> List[Dict]:
        """Get entities relevant to the user's input using semantic search."""
        # TODO: Implement semantic search for entities
        return []
    
    def _get_important_facts(self) -> List[Dict]:
        """Get important facts that should always be shown."""
        # TODO: Implement once fact extraction is added
        return []
    
    def _get_relevant_facts(self, user_input: str) -> List[Dict]:
        """Get facts relevant to the user's input using semantic search."""
        # TODO: Implement once fact extraction is added
        return []
    
    def _get_relevant_summaries(self, user_input: str) -> List[Dict]:
        """Get all summaries from Neo4j."""
        query = """
        MATCH (s:Summary)
        RETURN s
        ORDER BY s.created_at DESC
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record["s"]) for record in result]
    
    def _format_preferences(self, preferences: List[Preference]) -> List[str]:
        """Format preferences into human-readable strings."""
        formatted = []
        for pref in preferences:
            details_str = ", ".join(f"{k}: {v}" for k, v in pref.details.items())
            formatted.append(f"- {pref.preference_type}: {details_str} (Importance: {pref.importance})")
        return formatted
    
    def _format_entities(self, entities: List[Dict]) -> List[str]:
        """Format entities into human-readable strings."""
        formatted = []
        for entity in entities:
            formatted.append(f"- {entity['name']} ({entity['type']})")
        return formatted
    
    def _format_facts(self, facts: List[Dict]) -> List[str]:
        """Format facts into human-readable strings."""
        return [f"- {fact['content']}" for fact in facts]
    
    def _format_summaries(self, summaries: List[Dict]) -> List[str]:
        """Format summaries into human-readable strings."""
        return [f"- {summary['content']}" for summary in summaries]
    
    def _neo4j_to_preference(self, neo4j_node: Any) -> Preference:
        """
        Convert a Neo4j node to a Preference object with proper type conversions.
        
        Args:
            neo4j_node: Raw Neo4j node data
            
        Returns:
            Preference: Properly formatted preference object
            
        Logs:
            DEBUG: Node conversion details
            ERROR: Conversion failures
        """
        try:
            logger.debug(f"Converting Neo4j node: {neo4j_node}")
            node_dict = dict(neo4j_node)
            
            # Convert details from string to dict if needed
            details = node_dict.get('details', {})
            if isinstance(details, str):
                import json
                details = json.loads(details)
                
            # Convert datetime objects
            created_at = node_dict.get('created_at')
            if hasattr(created_at, 'to_native'):  # Neo4j DateTime conversion
                created_at = created_at.to_native()
            else:
                created_at = datetime.now()  # Default to current time if not set
                
            updated_at = node_dict.get('updated_at')
            if hasattr(updated_at, 'to_native'):  # Neo4j DateTime conversion
                updated_at = updated_at.to_native()
            else:
                updated_at = created_at  # Default to created_at if not set
                
            # Generate a random UUID if person_id is None
            import uuid
            person_id = str(node_dict.get('person_id')) if node_dict.get('person_id') else str(uuid.uuid4())
            
            return Preference(
                id=node_dict.get('id'),
                person_id=person_id,
                preference_type=node_dict.get('preference_type'),
                importance=node_dict.get('importance', 1),
                details=details,
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Error converting Neo4j node to Preference: {str(e)}")
            raise 