from typing import List,Dict
from datetime import datetime
from uuid import uuid4
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

def llm_factory(settings, llm_name: str):
        llm_config = settings.llm_config.get(llm_name)
        if llm_config:
            params = {key: value for key, value in llm_config.get("params", {}).items()}

            provider = llm_config.get("provider")
            if provider == "openai":
                llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, **params)
            elif provider == "anthropic":
                llm = ChatAnthropic(api_key=settings.ANTHROPIC_API_KEY, **params)
            elif provider == "xai":
                llm = ChatXAI(api_key=settings.XAI_API_KEY, **params)
            elif provider == "google":
                llm = ChatGoogleGenerativeAI(api_key=settings.GOOGLE_API_KEY, **params)
            elif provider == "groq":
                llm = ChatGroq(api_key=settings.GROQ_API_KEY, **params)
            elif provider == "ollama":
                llm = ChatOllama(**params)
            else:
                raise ValueError(f"Invalid LLM provider: {provider}")
        else:
            raise ValueError(f"LLM config for {llm_name} not found")

        return llm  


def prepare_conversation_data(batch: List[BaseMessage]) -> List[Dict]:
        """
        Prepare conversation data for processing, ensuring all data is JSON serializable.
        
        Args:
            batch (List[SmileMessage]): List of messages to process
            
        Returns:
            List[Dict]: List of dictionaries containing message data
        """
        conversation_data = []
        for msg in batch:
            # Safely access 'id' field with a default value and convert UUID to string
            msg_id = str(getattr(msg, 'id', None) or uuid4())
            
            # Safely access 'content' field
            content = getattr(msg, 'content', "")
            
            # Safely access 'timestamp' field and convert to ISO format
            timestamp = getattr(msg, 'timestamp', None)
            if timestamp:
                timestamp = timestamp.isoformat()
            else:
                timestamp = datetime.utcnow().isoformat()
            
            # Safely access 'sender' field with a default value
            type = getattr(msg, 'type', 'human')
            
            # Handle ToolMessage content if applicable
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, 'name', 'Unknown Tool')
                content = f"[Tool Output: {tool_name}]"
                sender = 'tool'
            
            # Build the message dictionary with all values ensured to be JSON serializable
            message_dict = {
                "id": msg_id,  # Now a string instead of UUID
                "content": content,
                "timestamp": timestamp,
                "type": type,
            }
            conversation_data.append(message_dict)
        
        return conversation_data