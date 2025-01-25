import os
import yaml
from rich import print
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Define the settings class
class Settings(BaseSettings):
    # API Keys
    LANGCHAIN_API_KEY: Optional[str] = None # Langchain api key
    OPENAI_API_KEY: Optional[str] = None # OpenAI API key
    ANTHROPIC_API_KEY: Optional[str] = None # Anthropic API key
    TAVILY_API_KEY: Optional[str] = None # Tavily API key
    TWILIO_ACCOUNT_SID: Optional[str] = None # Twilio account sid
    TWILIO_AUTH_TOKEN: Optional[str] = None # Twilio auth token
    NGROK_AUTHTOKEN: Optional[str] = None # Ngrok authtoken
    XAI_API_KEY: Optional[str] = None # XAI API key
    GOOGLE_API_KEY: Optional[str] = None # Google API key
    GROQ_API_KEY: Optional[str] = None # Groq API key
    
    # Neo4j Configuration
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "secretdb"
    
    # PostgreSQL Configuration
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "smile"
    
    # Config files
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    app_config_path: str = os.path.join("app", "configs", "app_config.yaml")
    llm_config_path: str = os.path.join("app", "configs", "llm_config.yaml")

    # Define the project settings
    @property
    def app_config(self):
        with open(self.app_config_path, "r") as file:
            return yaml.safe_load(file)
    
    # Define the LLM configuration
    @property
    def llm_config(self):
        # Change the path to point to your actual YAML config file
        with open(self.llm_config_path, "r") as file:
            return yaml.safe_load(file)
    
    # Helper methods for LLM clients
    def get_chatbot_agent_llm(self):
        """Get the chatbot agent LLM client."""
        # Mock implementation for testing
        return "chatbot_agent_llm"
    
    def get_embeddings_client(self):
        """Get the embeddings client."""
        # Mock implementation for testing
        return "embeddings_client"

# Initialize settings
settings = Settings()

# print("Settings have been loaded successfully")
