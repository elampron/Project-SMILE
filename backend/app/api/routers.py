from fastapi import APIRouter, HTTPException, Body, WebSocket, Depends, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Dict, Any, List
import logging
from app.agents.smile import Smile
from pydantic import BaseModel
from app.configs.settings import settings
from app.services.embeddings import EmbeddingsService
from app.agents.context import ContextManager
from app.services.neo4j import driver
import yaml
import os
import json
from fastapi.encoders import jsonable_encoder

from app.utils.logger import logger

# Initialize ContextManager
context_manager = ContextManager(driver)

# Create router instance
router = APIRouter()

# Initialize Smile as None - will be set during startup
smile = None

@router.get("/context")
async def get_context(user_input: str):
    """
    Get formatted context based on user input.
    
    Args:
        user_input (str): The user's current question or comment
        
    Returns:
        dict: A dictionary containing the formatted context and status
        
    Raises:
        HTTPException: If there's an error getting the context
    """
    try:
        logger.info(f"Getting formatted context for input: {user_input}")
        formatted_context = context_manager.get_formatted_context(user_input)
        return {
            "status": "success",
            "data": formatted_context
        }
    except Exception as e:
        logger.error(f"Error getting formatted context: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting formatted context: {str(e)}"
        )

async def get_smile():
    """
    Dependency to get initialized Smile instance.
    """
    global smile
    if not smile:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if hasattr(smile, '__aiter__'):
        async for instance in smile:
            return instance
    return smile

# Add these classes for request validation
class UpdateSettingsRequest(BaseModel):
    config_type: str  # Either "app_config" or "llm_config"
    settings_data: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    attachments: Optional[List[Dict[str, str]]] = []

# Original chat endpoint for JSON requests
@router.post("/chat/json")
async def chat_json_endpoint(
    request: ChatRequest,
    smile_agent: Any = Depends(get_smile)
) -> Dict[str, Any]:
    """Chat endpoint that accepts JSON input."""
    try:
        message = request.message.strip()
        thread_id = request.thread_id or settings.app_config["langchain_config"]["thread_id"]
        print(f"Thread ID from request: {thread_id}")
        if not message:
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")

        async def response_generator():
            try:
                config = {"thread_id": thread_id}
                async for chunk in smile_agent.stream(message, config=config, attachments=request.attachments):
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
        logger.error(f"Error in chat_json_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

# Form-based chat endpoint for file uploads
@router.post("/chat/form")
async def chat_form_endpoint(
    message: str = Form(...),
    attachments: Optional[List[UploadFile]] = None,
    smile_agent: Any = Depends(get_smile)
) -> Dict[str, Any]:
    """Chat endpoint that accepts form data."""
    try:
        logger.info(f"Received form request - Message: {message}, Files: {[f.filename for f in (attachments or [])]}")
        
        if not message.strip():
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")
            
        # Use default thread_id if none provided
        thread_id = settings.app_config["langchain_config"]["thread_id"]
        logger.info(f"Using thread_id: {thread_id}")
        
        # Process any uploaded files
        processed_attachments = []
        if attachments:
            logger.info(f"Processing {len(attachments)} files")
            for file in attachments:
                try:
                    # Read file content
                    content = await file.read()
                    
                    # Try to decode as UTF-8 text
                    try:
                        text_content = content.decode('utf-8')
                        logger.info(f"Successfully decoded file {file.filename} as UTF-8")
                    except UnicodeDecodeError:
                        # If not text, store as binary
                        text_content = f"[Binary file: {file.filename}]"
                        logger.info(f"File {file.filename} appears to be binary")
                    
                    processed_attachments.append({
                        'filename': file.filename,
                        'content': text_content
                    })
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                    # Continue with other files if one fails
                    continue

        async def response_generator():
            try:
                config = {"thread_id": thread_id}
                async for chunk in smile_agent.stream(message, config=config, attachments=processed_attachments):
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
        logger.error(f"Error in chat_form_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@router.get("/history")
async def get_history_endpoint(
    thread_id: Optional[str] = settings.app_config["langchain_config"]["thread_id"],
    num_messages: Optional[int] = 50,
    smile_agent: Any = Depends(get_smile)
):
    """Get conversation history endpoint."""
    try:
        history = await smile_agent.get_conversation_history(
            thread_id=thread_id,
            num_messages=num_messages
        )
        return JSONResponse(
            status_code=200,
            content={"status": "success", "data": history}
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )

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
        # Initialize Smile agent
        smile = Smile()
        await smile.initialize()
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
