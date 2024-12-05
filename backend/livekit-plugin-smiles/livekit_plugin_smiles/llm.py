from typing import AsyncIterator, Optional
from livekit.agents import llm
import aiohttp
import json
import logging

logger = logging.getLogger(__name__)

class LLM(llm.LLM):
    """SMILES LLM implementation for LiveKit's agent framework."""
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None):
        """Initialize the SMILES LLM.
        
        Args:
            api_url: The URL of the SMILES API endpoint
            api_key: Optional API key for authentication
        """
        super().__init__()
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(self, chat_ctx: llm.ChatContext) -> AsyncIterator[str]:
        """Process chat messages through SMILES API.
        
        Args:
            chat_ctx: LiveKit chat context containing message history
            
        Yields:
            Streamed response chunks from SMILES API
        """
        await self._ensure_session()
        
        # Convert LiveKit chat context to SMILES format
        messages = [
            {
                "role": msg.role,
                "content": msg.text
            } 
            for msg in chat_ctx.messages
        ]
        
        try:
            async with self.session.post(
                f"{self.api_url}/chat",
                headers=self._get_headers(),
                json={"messages": messages}
            ) as response:
                response.raise_for_status()
                
                # Handle streaming response
                async for chunk in response.content:
                    if chunk:
                        try:
                            text = chunk.decode().strip()
                            if text:
                                try:
                                    # Try to parse as JSON first
                                    data = json.loads(text)
                                    # Extract the relevant text content based on the response structure
                                    if "user_transcript" in data:
                                        yield data["user_transcript"]
                                    else:
                                        # If it's a different type of JSON response, yield the whole JSON
                                        yield json.dumps(data)
                                except json.JSONDecodeError:
                                    # If it's not JSON, yield the raw text
                                    yield text
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
                            continue
                            
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            yield "I apologize, but I'm having trouble connecting to the service right now."
            
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            yield "I encountered an unexpected error. Please try again later."

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
