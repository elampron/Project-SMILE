"""
Embeddings service for managing vector operations in Neo4j.

This module provides functionality for:
1. Generating embeddings using OpenAI's text-embedding model
2. Creating and managing vector indexes in Neo4j
3. Performing vector similarity searches
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from app.configs.settings import settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Service for managing embeddings and vector operations.
    
    This class handles:
    1. Generating embeddings using OpenAI's API
    2. Creating and managing vector indexes in Neo4j
    3. Performing similarity searches
    """
    
    def __init__(self, driver):
        """
        Initialize the EmbeddingsService.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.model = settings.llm_config.get("embeddings").get("params").get("model")
    
        
    def create_vector_indexes(self):
        """
        Create vector indexes in Neo4j for similarity search.
        This should be called during application initialization.
        """
        indexes = [
            ("Preference", "preference_vector", "embedding", 1536),  # OpenAI embedding dimension
            ("Summary", "summary_vector", "embedding", 1536),
            ("Person", "person_vector", "embedding", 1536),
            ("Organization", "org_vector", "embedding", 1536)
        ]
        
        with self.driver.session() as session:
            for label, index_name, property_name, dimensions in indexes:
                try:
                    # Create vector index
                    query = f"""
                    CALL db.index.vector.createNodeIndex(
                        $index_name,
                        $label,
                        $property_name,
                        $dimensions,
                        'cosine'
                    )
                    """
                    session.run(
                        query,
                        index_name=index_name,
                        label=label,
                        property_name=property_name,
                        dimensions=dimensions
                    )
                    logger.info(f"Created vector index {index_name} for {label} nodes")
                except Exception as e:
                    if "An equivalent index already exists" in str(e):
                        logger.debug(f"Vector index {index_name} already exists")
                    else:
                        logger.error(f"Error creating vector index {index_name}: {str(e)}")
                        raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using OpenAI's API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Vector embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query_embedding: List[float],
        node_label: str,
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Neo4j using vector index.
        
        Args:
            query_embedding: Query vector
            node_label: Label of nodes to search (Preference, Summary, etc.)
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List[Dict]: List of similar nodes with their properties
        """
        query = f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $k,
            $query_vector,
            $min_score
        ) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            try:
                index_name = {
                    "Preference": "preference_vector",
                    "Summary": "summary_vector",
                    "Person": "person_vector",
                    "Organization": "org_vector"
                }[node_label]
                
                result = session.run(
                    query,
                    index_name=index_name,
                    k=limit,
                    query_vector=query_embedding,
                    min_score=min_score
                )
                
                return [
                    {
                        **dict(record["node"]),
                        "similarity_score": record["score"]
                    }
                    for record in result
                ]
            except Exception as e:
                logger.error(f"Error performing similarity search: {str(e)}")
                raise 