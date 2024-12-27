"""
Embeddings service for managing text embeddings.

This module provides functionality for:
1. Generating embeddings using OpenAI's text-embedding model
2. Batch processing for multiple texts
3. Embedding caching (future enhancement)
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from app.configs.settings import settings
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Service for managing text embeddings.
    
    This class handles:
    1. Generating embeddings using OpenAI's API
    2. Batch processing for multiple texts
    3. Future: Embedding caching and optimization
    """
    
    def __init__(self, driver: Optional[GraphDatabase.driver] = None):
        """
        Initialize the EmbeddingsService.
        
        Args:
            driver (Optional[GraphDatabase.driver]): Neo4j driver instance for database operations
        """
        self.model = settings.llm_config.get("embeddings").get("params").get("model")
        self.driver = driver
    
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
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of vector embeddings
            
        Raises:
            Exception: If batch embedding generation fails
        """
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Test script for EmbeddingsService
    """
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize service
        embeddings_service = EmbeddingsService()
        
        # Test 1: Generate single embedding
        logger.info("\nTest 1: Generating single embedding...")
        text = "This is a test sentence for embedding generation."
        embedding = embeddings_service.generate_embedding(text)
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test 2: Generate batch embeddings
        logger.info("\nTest 2: Generating batch embeddings...")
        texts = [
            "First test sentence for batch processing.",
            "Second test sentence for batch processing.",
            "Third test sentence for batch processing."
        ]
        embeddings = embeddings_service.batch_generate_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            logger.info(f"Embedding {i+1} dimensions: {len(emb)}")
        
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
    finally:
        logger.info("Tests completed")