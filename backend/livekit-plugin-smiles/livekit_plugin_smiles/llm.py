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

    async def chat(self, chat_ctx: llm.ChatContext, fnc_ctx: Optional[dict] = None) -> AsyncIterator[str]:
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
            
        # Debug log the full message history
        logger.debug("Full message history:")
        for msg in chat_ctx.messages:
            logger.debug(f"Message: {msg.__dict__}")
            
        last_message = chat_ctx.messages[-1]
        
        # Debug log the last message content
        logger.debug(f"Processing last message: {last_message.__dict__}")
        
        # Prepare request payload according to ChatRequest schema
        payload = {
            "message": last_message.content,  # Changed from text to content
            "thread_id": getattr(chat_ctx, 'thread_id', None)  # Use thread_id if available
        }
        
        # Debug log the request payload
        logger.debug(f"Sending request to API: {payload}")
        
        try:
            async with self.session.post(
                f"{self.api_url}/chat/json",
                headers=self._get_headers(),
                json=payload
            ) as response:
                response.raise_for_status()
                
                # Debug log the response
                logger.debug(f"API response status: {response.status}")
                
                # Handle streaming response
                async for chunk in response.content:
                    if chunk:
                        try:
                            text = chunk.decode().strip()
                            logger.debug(f"Received chunk: {text}")
                            if text:
                                yield text
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
                            continue
                            
        except aiohttp.ClientError as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = await e.response.text()
                    logger.error(f"API error response body: {error_body}")
                except:
                    pass
            logger.error(f"API request failed: {error_msg}")
            yield "I apologize, but I'm having trouble connecting to the service right now."
            
        except Exception as e:
            logger.error(f"Unexpected error in chat: {str(e)}")
            yield "I encountered an unexpected error. Please try again later."

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
