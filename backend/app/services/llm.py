"""
LLM service module for creating and configuring language models.
"""

from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from app.configs.settings import settings

def llm_factory(settings: Any, model_type: str) -> Any:
    """
    Factory function to create language model instances.
    
    Args:
        settings: Application settings
        model_type: Type of model to create ("chatbot_agent" or "embeddings")
        
    Returns:
        Language model instance
        
    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == "chatbot_agent":
        return ChatOpenAI(
            model=settings.llm_config["chatbot_agent"]["params"]["model"],
            temperature=settings.llm_config["chatbot_agent"]["params"]["temperature"],
            streaming=True
        )
    elif model_type == "embeddings":
        return OpenAIEmbeddings(
            model=settings.llm_config["embeddings"]["params"]["model"]
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}") 