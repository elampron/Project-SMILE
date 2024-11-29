from fastapi import APIRouter, HTTPException, Body, WebSocket, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
import logging
from app.agents.smile import Smile
from pydantic import BaseModel
from app.configs.settings import settings
import yaml
import os
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
    thread_id: Optional[str] = settings.app_config["langchain_config"]["thread_id"]

# Add these classes for request validation
class UpdateSettingsRequest(BaseModel):
    config_type: str  # Either "app_config" or "llm_config"
    settings_data: Dict[str, Any]

@router.post("/chat")
def chat_endpoint(
    chat_request: ChatRequest = Body(...),
    smile_agent: Smile = Depends(get_smile)
):
    """Synchronous chat endpoint."""
    try:
        if not chat_request.message:
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")

        # check if thread_id was provided, if not, use the one from the app_config
        if not chat_request.thread_id:
            thread_id = settings.app_config["langchain_config"]["thread_id"]
        else:
            thread_id = chat_request.thread_id

        def response_generator():
            try:
                for chunk in smile_agent.stream(
                    chat_request.message,
                    config={"configurable": {"thread_id": thread_id}}
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
    thread_id: Optional[str] = settings.app_config["langchain_config"]["thread_id"],
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

@router.get("/settings/{config_type}")
async def get_settings(config_type: str):
    """
    Get the current settings for the specified configuration type.
    
    Args:
        config_type (str): The type of configuration to retrieve ("app_config" or "llm_config")
        
    Returns:
        dict: The current settings for the specified configuration
    """
    try:
        if config_type not in ["app_config", "llm_config"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid config_type. Must be either 'app_config' or 'llm_config'"
            )
        
        config = getattr(settings, config_type, None)
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_type} not found"
            )
            
        return {"status": "success", "data": config}
    except Exception as e:
        logger.error(f"Error retrieving settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving settings: {str(e)}")

@router.put("/settings")
async def update_settings(update_request: UpdateSettingsRequest):
    """
    Update the settings for the specified configuration type.
    
    Args:
        update_request (UpdateSettingsRequest): The request containing the config type and new settings
        
    Returns:
        dict: A success message and the updated settings
    """
    try:
        if update_request.config_type not in ["app_config", "llm_config"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid config_type. Must be either 'app_config' or 'llm_config'"
            )

        # Get the file path based on config type
        config_paths = {
            "app_config": settings.app_config_path,
            "llm_config": settings.llm_config_path
        }
        
        file_path = config_paths.get(update_request.config_type)
        if not file_path:
            raise HTTPException(
                status_code=500,
                detail=f"Config path not found for {update_request.config_type}"
            )
        
        try:
            # Read the current config file
            with open(file_path, 'r') as f:
                current_config = yaml.safe_load(f) or {}
            
            # Update with new settings
            current_config.update(update_request.settings_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the updated settings to the YAML file
            with open(file_path, 'w') as f:
                yaml.safe_dump(current_config, f)
                
            # Update the settings object in memory
            setattr(settings, update_request.config_type, current_config)
            
            logger.info(f"Successfully updated {update_request.config_type} settings")
            
        except Exception as e:
            logger.error(f"Error writing to config file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error writing to config file: {str(e)}"
            )
        
        return {
            "status": "success",
            "message": f"Successfully updated {update_request.config_type}",
            "data": current_config
        }
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

@router.get("/settings")
async def get_all_settings():
    """
    Get all available settings including both app_config and llm_config.
    
    Returns:
        dict: A dictionary containing all configuration settings
    """
    try:
        return {
            "status": "success",
            "data": {
                "app_config": settings.app_config,
                "llm_config": settings.llm_config,
                # Add any additional configuration types here
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving all settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving all settings: {str(e)}")

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
            smile.cleanup()
            logger.info("Smile agent cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during Smile agent cleanup: {str(e)}", exc_info=True)
        raise
