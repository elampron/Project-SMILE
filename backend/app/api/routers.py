from fastapi import APIRouter, HTTPException, Body, WebSocket, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
from app.agents.smile import Smile
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize Smile as None - will be set during startup
smile = None

async def get_smile():
    """
    Dependency to get initialized Smile instance.
    """
    if not smile:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return smile

# Add this class for request validation
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "MainThread"

@router.post("/chat")
def chat_endpoint(
    chat_request: ChatRequest = Body(...),
    smile_agent: Smile = Depends(get_smile)
):
    """Synchronous chat endpoint."""
    try:
        if not chat_request.message:
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")
            
        def response_generator():
            try:
                for chunk in smile_agent.stream(
                    chat_request.message,
                    config={"configurable": {"thread_id": chat_request.thread_id}}
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}"

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
        raise HTTPException(status_code=500, detail=f"Error initializing chat: {str(e)}")

@router.get("/history")
def get_history_endpoint(
    thread_id: Optional[str] = "MainThread",
    num_messages: Optional[int] = 50,
    smile_agent: Smile = Depends(get_smile)
):
    """Synchronous history endpoint."""
    try:
        history = smile_agent.get_conversation_history(
            num_messages=num_messages, 
            thread_id=thread_id
        )
        return {"status": "success", "data": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@router.on_event("startup")
async def startup_event():
    """Initialize Smile agent on startup"""
    global smile
    try:
        smile = Smile()
        smile.initialize()
        logger.info("Smile agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Smile agent: {str(e)}", exc_info=True)
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup Smile agent on shutdown"""
    global smile
    try:
        if smile:
            await smile.cleanup()
            logger.info("Smile agent cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during Smile agent cleanup: {str(e)}", exc_info=True)
        raise
