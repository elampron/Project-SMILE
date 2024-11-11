import os
import yaml
from rich import print
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Define the settings class
class Settings(BaseSettings):
    # Define the project settings
    PROJECT_NAME: str = "Proeject-SMILE" # Project name
    PROJECT_VERSION: str = "0.1.0" # Project version
    PROJECT_DESCRIPTION: str = "Proeject-SMILE" # Project description
    DEBUG: bool = False # Debug mode
    LOG_LEVEL: str = "INFO" # Log level
    LOG_FILE: Optional[str] = None # Log file
    LANGCHAIN_TRACING_V2: Optional[bool] = False # Langchain tracing v2
    LANGCHAIN_ENDPOINT: Optional[str] = None # Langchain endpoint
    LANGCHAIN_API_KEY: Optional[str] = None # Langchain api key
    LANGCHAIN_PROJECT: Optional[str] = None # Langchain project
    # Define the LLM settings
    LLM_EMBEDDING: Optional[str] = None # Embedding model
    LLM_BIG: str="gtp-4os" # Big model for main interaction
    LLM_SMALL: str="gtp-4o-mini" # Small model for summarization, categorization and extraction
    
    # Define the LLM configuration
    @property
    def llm_config(self):
        with open("app\configs\settings.py", "r") as file:
            return yaml.safe_load(file)

# Example usage
settings = Settings()
print(settings.llm_config.chatbot_agent.name)