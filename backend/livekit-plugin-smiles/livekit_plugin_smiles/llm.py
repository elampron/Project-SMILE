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
        
        # Get the last message from the context
        if not chat_ctx.messages:
            yield "No message provided"
            return
            
        last_message = chat_ctx.messages[-1]
        
        # Prepare request payload according to ChatRequest schema
        payload = {
            "message": last_message.text,
            "thread_id": getattr(chat_ctx, 'thread_id', None)  # Use thread_id if available
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/chat/json",
                headers=self._get_headers(),
                json=payload
            ) as response:
                response.raise_for_status()
                
                # Handle streaming response
                async for chunk in response.content:
                    if chunk:
                        try:
                            text = chunk.decode().strip()
                            if text:
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
