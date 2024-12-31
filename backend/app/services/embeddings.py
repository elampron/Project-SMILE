"""
Embeddings service for generating vector embeddings.

This module provides functionality for generating embeddings using OpenAI's text-embedding model.
"""

from app.utils.logger import logger
from typing import List
import openai
from app.configs.settings import settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmbeddingsService:
    """
    Service for generating embeddings using OpenAI's API.
    """
    
    def __init__(self):
        """Initialize the EmbeddingsService."""
        self.model = settings.llm_config.get("embeddings").get("params").get("model")
    
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