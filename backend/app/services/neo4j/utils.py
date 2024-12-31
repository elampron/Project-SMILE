"""
Utility functions for Neo4j operations.

This module provides common utility functions used across the Neo4j services,
including property conversion, JSON encoding, and other helper functions.
"""

import json
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class Neo4jEncoder(json.JSONEncoder):
    """JSON encoder for Neo4j-specific types."""
    def default(self, obj):
        if isinstance(obj, (datetime, UUID)):
            return str(obj)
        return super().default(obj)

def convert_datetime_fields(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert datetime objects to ISO format strings in a dictionary.
    
    Args:
        properties (Dict[str, Any]): Dictionary containing potential datetime fields
        
    Returns:
        Dict[str, Any]: Dictionary with datetime fields converted to ISO strings
    """
    datetime_fields = ['created_at', 'start_time', 'end_time', 'since', 'until']
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

def convert_properties_for_neo4j(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a dictionary of properties to be Neo4j-compatible.
    
    Args:
        properties (Dict[str, Any]): Dictionary of properties to convert
        
    Returns:
        Dict[str, Any]: Dictionary with Neo4j-compatible values
    """
    if not properties:
        return {}

    # Create a copy to avoid modifying the original
    converted = properties.copy()

    # Convert datetime fields
    converted = convert_datetime_fields(converted)

    # Convert UUID fields to strings
    for key, value in converted.items():
        if isinstance(value, UUID):
            converted[key] = str(value)
        elif isinstance(value, list):
            # Convert list items if they are UUIDs
            converted[key] = [str(item) if isinstance(item, UUID) else item for item in value]
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            converted[key] = convert_properties_for_neo4j(value)

    return converted

def exclude_embedding_from_properties(properties: dict) -> dict:
    """
    Create a copy of properties dictionary without the embedding field.
    
    Args:
        properties (dict): Original properties dictionary
        
    Returns:
        dict: Properties dictionary without embedding field
    """
    if not properties:
        return {}
    
    return {k: v for k, v in properties.items() if k != 'embedding'} 