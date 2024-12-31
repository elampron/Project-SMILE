from fastapi import APIRouter, HTTPException, Body, WebSocket, Depends, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
import logging
from app.agents.smile import Smile
from pydantic import BaseModel
from app.configs.settings import settings
from app.services.embeddings import EmbeddingsService
import yaml
import os
import json
from fastapi.encoders import jsonable_encoder

from app.utils.logger import logger


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

# Add these classes for request validation
class UpdateSettingsRequest(BaseModel):
    config_type: str  # Either "app_config" or "llm_config"
    settings_data: Dict[str, Any]

# Original chat endpoint for JSON requests
@router.post("/chat/json")
async def chat_json_endpoint(
    chat_request: dict = Body(...),
    smile_agent: Smile = Depends(get_smile)
):
    """JSON-based chat endpoint for backward compatibility"""
    try:
        message = chat_request.get("message", "").strip()
        thread_id = chat_request.get("thread_id") or settings.app_config["langchain_config"]["thread_id"]
        
        if not message:
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")

        def response_generator():
            try:
                for chunk in smile_agent.stream(
                    message,
                    config={
                        "configurable": {
                            "thread_id": thread_id
                        }
                    }
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
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in chat endpoint: {str(e)}")

# New form-data chat endpoint
@router.post("/chat")
async def chat_form_endpoint(
    message: str = Form(...),
    thread_id: str = Form(None),
    files: List[UploadFile] = File([]),
    smile_agent: Smile = Depends(get_smile)
):
    """
    Form-data chat endpoint that supports file uploads.
    Files are processed and saved, then passed to the agent for this run only.
    """
    try:
        logger.info(f"Received form request - Message: {message}, Thread ID: {thread_id}, Files: {[f.filename for f in files]}")
        
        if not message.strip():
            raise HTTPException(status_code=422, detail="Message must be a non-empty string")

        # Use thread_id from settings if not provided
        effective_thread_id = thread_id or settings.app_config["langchain_config"]["thread_id"]
        logger.info(f"Using thread_id: {effective_thread_id}")

        # Process uploaded files for this run
        current_attachments = []
        if files:
            logger.info(f"Processing {len(files)} files")
            for file in files:
                try:
                    if not file.filename:
                        logger.warning("Skipping file with no filename")
                        continue
                        
                    content = await file.read()
                    try:
                        decoded_content = content.decode('utf-8')
                        logger.info(f"Successfully decoded file {file.filename} as UTF-8")
                    except UnicodeDecodeError:
                        logger.warning(f"File {file.filename} contains binary content")
                        decoded_content = str(content)
                    
                    # Save document and get attachment object for this run
                    try:
                        attachment = smile_agent.save_document(decoded_content, file.filename)
                        if attachment:
                            current_attachments.append(attachment)
                            logger.info(f"Successfully processed file: {file.filename}")
                        else:
                            logger.warning(f"Failed to create attachment for {file.filename}")
                    except Exception as e:
                        logger.error(f"Error saving document {file.filename}: {str(e)}")
                        continue
                    
                    # Reset file seek position
                    await file.seek(0)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    # Continue processing other files instead of failing completely
                    continue

        if not current_attachments and files:
            logger.warning("No valid attachments were created from uploaded files")

        async def response_generator():
            try:
                # Pass current attachments to the stream method
                for chunk in smile_agent.stream(
                    message,
                    config={
                        "configurable": {
                            "thread_id": effective_thread_id
                        }
                    },
                    attachments=current_attachments  # Pass attachments for this run
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
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in chat endpoint: {str(e)}")

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
        # Initialize Smile agent
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
