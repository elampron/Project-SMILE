from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
from app.agents.smile import Smile

# Configure logging
logger = logging.getLogger(__name__)

smile = Smile()
# Create router instance
router = APIRouter()

@router.post("/chat")
async def chat_endpoint(
    message: str,
    thread_id: Optional[str] = "MainThread"
):
    """
    Chat endpoint that streams responses from the Smile agent.
    
    Args:
        message (str): The user's input message
        thread_id (str, optional): Thread identifier for conversation tracking. Defaults to "MainThread"
    
    Returns:
        StreamingResponse: A streaming response containing the agent's reply
    
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Initialize Smile agent
        logger.info(f"Initializing chat with message: {message[:50]}... (thread_id: {thread_id})")
        
        
        async def response_generator():
            """Generator function to stream the response"""
            try:
                response_content = ""
                for chunk in smile.stream(
                    message,
                    config={"thread_id": thread_id}
                ):
                    response_content += chunk
                    # Stream each chunk to the client
                    yield chunk
                
                logger.info(f"Successfully completed chat response (thread_id: {thread_id})")
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating response: {str(e)}"
                )
        
        return StreamingResponse(
            response_generator(),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing chat: {str(e)}"
        )
