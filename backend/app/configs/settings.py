import os
import yaml
from rich import print
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Define the settings class
class Settings(BaseSettings):
    LANGCHAIN_API_KEY: Optional[str] = None # Langchain api key
    OPENAI_API_KEY: Optional[str] = None # OpenAI API key
    ANTHROPIC_API_KEY: Optional[str] = None # Anthropic API key
    TAVILY_API_KEY: Optional[str] = None # Tavily API key
    TWILIO_ACCOUNT_SID: Optional[str] = None # Twilio account sid
    TWILIO_AUTH_TOKEN: Optional[str] = None # Twilio auth token
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    # Define the project settings
    @property
    def app_config(self):
        config_path = os.path.join("app", "configs", "app_config.yaml")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    
    # Define the LLM configuration
    @property
    def llm_config(self):
        # Change the path to point to your actual YAML config file
        config_path = os.path.join("app", "configs", "llm_config.yaml")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
        
   


# Comment out or remove the example usage since it's causing errors
settings = Settings()

# print("Settings have been loaded successfully")
