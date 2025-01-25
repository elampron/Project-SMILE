"""
Tests for the real Smile agent by directly calling router endpoint methods from 'backend/app/api/routers.py'.
This bypasses the FastAPI app layer and allows direct testing of each endpoint's logic with the actual Smile agent.

Detailed logging is used to trace behaviors and potential issues.
"""

import pytest
import logging
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, AsyncGenerator
import io
import json
from pathlib import Path

# Import router endpoint functions and the global Smile from routers
from app.api.routers import (
    get_context,
    chat_json_endpoint,
    chat_form_endpoint,
    get_history_endpoint,
    get_settings,
    update_settings,
    get_all_settings,
    startup_event,
    shutdown_event,
    get_smile,
    UpdateSettingsRequest,
    ChatRequest,
    smile as global_smile  # Import the global smile variable
)

from app.models.agents import Attachment, AttachmentType
from app.agents.smile import Smile  # Updated import from the new package structure
from app.utils.logger import logger

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture
async def real_smile():
    """
    Fixture that provides a real Smile agent instance for testing.
    Creates a new instance for each test to avoid coroutine reuse issues.
    """
    try:
        smile_instance = Smile()
        await smile_instance.initialize()
        logger.info("[TEST] Smile agent initialized successfully")
        yield smile_instance
    except Exception as e:
        logger.error(f"Error initializing Smile agent: {e}")
        raise
    finally:
        logger.info("[TEST] Shutting down Smile agent")

class TestSmileContext:
    """Tests for context-related functionality."""
    
    @pytest.mark.asyncio
    async def test_get_context(self, real_smile):
        """Test the get_context endpoint function with a mock user_input using the real agent."""
        user_input = "How can I set up a new project?"
        logger.info(f"[TEST] test_get_context with user_input='{user_input}'")

        async for smile_instance in real_smile:
            response = await get_context(user_input=user_input)
            assert response["status"] == "success"
            assert isinstance(response["data"], str)
            assert len(response["data"]) > 0
            break

class TestSmileChat:
    """Tests for chat-related functionality."""
    
    @pytest.mark.asyncio
    async def test_chat_json_endpoint(self, real_smile):
        """Test the chat_json_endpoint function."""
        async for smile_instance in real_smile:
            chat_request = ChatRequest(
                message="Hello, how are you?",
                thread_id="test_thread",
                attachments=[]
            )

            response = await chat_json_endpoint(chat_request, smile_instance)
            assert response.status_code == 200
            assert isinstance(response, StreamingResponse)

            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode())

            assert len(chunks) > 0
            assert any(len(chunk.strip()) > 0 for chunk in chunks)
            break

    @pytest.mark.asyncio
    async def test_chat_form_endpoint(self, real_smile):
        """Test the chat form endpoint."""
        async for smile_instance in real_smile:
            # Create a test file
            file = UploadFile(
                filename="test.txt",
                file=io.BytesIO(b"test content")
            )
            
            # Test the endpoint
            response = await chat_form_endpoint(
                message="test message",
                attachments=[file],
                smile_agent=smile_instance
            )
            
            assert isinstance(response, StreamingResponse)
            
            # Read the response content
            content = []
            async for chunk in response.body_iterator:
                content.append(chunk.decode())
            
            # Join the chunks and verify content
            response_text = "".join(content)
            assert len(response_text) > 0

class TestSmileHistory:
    """Tests for history-related functionality."""
    
    @pytest.mark.asyncio
    async def test_get_history_endpoint(self, real_smile):
        """Test get_history_endpoint."""
        test_thread_id = "test_thread_history"
        logger.info(f"[TEST] test_get_history_endpoint with thread_id='{test_thread_id}'")

        async for smile_instance in real_smile:
            response = await get_history_endpoint(thread_id=test_thread_id, num_messages=50, smile_agent=smile_instance)
            assert response.status_code == 200
            assert response.body is not None
            break

class TestSmileSettings:
    """Tests for settings-related functionality."""
    
    @pytest.mark.asyncio
    async def test_get_settings(self):
        """Test the get_settings endpoint for a valid config_type."""
        config_type = "app_config"
        logger.info(f"[TEST] test_get_settings with config_type='{config_type}'")

        response = await get_settings(config_type)
        logger.info(f"get_settings response: {response}")

        assert response["status"] == "success"
        assert isinstance(response["data"], dict), "Expected dictionary data for config"

    @pytest.mark.asyncio
    async def test_get_settings_invalid(self):
        """Test the get_settings endpoint for an invalid config_type."""
        config_type = "invalid_config_type"
        logger.info(f"[TEST] test_get_settings_invalid with config_type='{config_type}'")

        with pytest.raises(HTTPException) as exc_info:
            await get_settings(config_type)
        logger.info(f"Raised HTTPException detail: {exc_info.value.detail}")

        assert "Invalid config_type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_settings(self):
        """Test updating settings."""
        update_request = UpdateSettingsRequest(
            config_type="app_config",
            settings_data={"test_new_setting": "test_value"}
        )
        logger.info(f"[TEST] test_update_settings with update_request={update_request.model_dump_json()}")

        with pytest.raises(HTTPException) as exc_info:
            await update_settings(update_request)

        logger.info(f"HTTPException detail: {exc_info.value.detail}")

    @pytest.mark.asyncio
    async def test_update_settings_invalid(self):
        """Test that update_settings raises an HTTPException for invalid config_type."""
        update_request = UpdateSettingsRequest(
            config_type="totally_invalid_config",
            settings_data={"dummy_key": "dummy_value"}
        )
        logger.info("[TEST] test_update_settings_invalid with config_type='totally_invalid_config'")

        with pytest.raises(HTTPException) as exc_info:
            await update_settings(update_request)
        logger.info(f"Raised HTTPException detail: {exc_info.value.detail}")

        assert "Invalid config_type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_all_settings(self):
        """Test the get_all_settings endpoint."""
        logger.info("[TEST] test_get_all_settings")

        response = await get_all_settings()
        logger.info(f"get_all_settings response: {response}")

        assert response["status"] == "success"
        data = response["data"]
        assert "app_config" in data, "Expected 'app_config' in returned data"
        assert "llm_config" in data, "Expected 'llm_config' in returned data"

# @pytest.mark.asyncio
# async def test_process_attachments(real_smile, tmp_path):
#     """Test processing of file attachments."""
#     async for smile_instance in real_smile:
#         # Create a test file
#         test_file = tmp_path / "test.txt"
#         test_file.write_text("This is a test file")
        
#         # Create a mock file upload
#         with open(test_file, "rb") as f:
#             file = UploadFile(S
#                 filename="test.txt",
#                 file=io.BytesIO(f.read())
#             )
        
#         # Process the attachment
#         config = {"thread_id": "test_thread"}
#         async for chunk in smile_instance.stream(
#             "Here's a file",
#             config=config,
#             attachments=[{
#                 'filename': file.filename,
#                 'content': "This is a test file"
#             }]
#         ):
#             assert isinstance(chunk, str)
#             assert len(chunk) > 0
#             break 