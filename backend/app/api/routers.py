from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
from app.agents.smile import Smile
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

smile = Smile()
# Create router instance
router = APIRouter()

# Add this class for request validation
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "MainThread"

@router.post("/chat")
async def chat_endpoint(
    chat_request: ChatRequest = Body(...)
):
    """
    Chat endpoint that streams responses from the Smile agent.
    
    Args:
        chat_request (ChatRequest): The chat request containing message and thread_id
    
    Returns:
        StreamingResponse: A streaming response containing the agent's reply
    
    Raises:
        HTTPException: If there's an error processing the request
    """
    # Log the raw request parameters
    logger.info(f"Received chat request - Message: {chat_request.message[:100]}...")
    logger.info(f"Thread ID: {chat_request.thread_id}")
    
    try:
        # Validate input parameters
        if not chat_request.message:
            logger.error("Empty message received")
            raise HTTPException(
                status_code=422,
                detail="Message must be a non-empty string"
            )
            
        logger.info(f"Initializing chat with message: {chat_request.message[:50]}... (thread_id: {chat_request.thread_id})")
        
        # Define the response generator as an asynchronous generator
        async def response_generator():
            """Asynchronous generator function to stream the response"""
            try:
                response_content = ""
                # Log the configuration being passed to smile.stream
                logger.debug(f"Calling smile.stream with config: {{'thread_id': {chat_request.thread_id}}}")

                # Use 'async for' since smile.stream is an asynchronous generator
                async for chunk in smile.stream(
                    chat_request.message,
                    config={"thread_id": chat_request.thread_id}
                ):
                    response_content += chunk
                    logger.debug(f"Streaming chunk of size: {len(chunk)} bytes")
                    # Yield the chunk directly
                    yield chunk

                logger.info(f"Successfully completed chat response (thread_id: {chat_request.thread_id})")

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                # Handle exceptions within the generator
                yield f"Error: {str(e)}"

        # Return the StreamingResponse with the asynchronous generator
        return StreamingResponse(
            response_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "Content-Encoding": "identity",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing chat: {str(e)}"
        )
