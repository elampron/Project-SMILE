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

from app.agents.smile import Smile

# Configure logging
logger = logging.getLogger(__name__)

# This fixture ensures the real Smile agent is spun up once for all tests, then cleaned up.
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

# =====================
# Endpoint Tests
# =====================

@pytest.mark.asyncio
async def test_get_context(real_smile):
    """
    Test the get_context endpoint function with a mock user_input using the real agent.
    """
    user_input = "How can I set up a new project?"
    logger.info(f"[TEST] test_get_context with user_input='{user_input}'")

    # Get the Smile instance
    async for smile_instance in real_smile:
        response = await get_context(user_input=user_input)
        assert response["status"] == "success"
        assert isinstance(response["data"], str)
        assert len(response["data"]) > 0
        break

@pytest.mark.asyncio
async def test_chat_json_endpoint(real_smile):
    """
    Test the chat_json_endpoint function by providing a mock chat request body
    and verifying the streaming response from the real agent.
    """
    # Get the Smile instance
    async for smile_instance in real_smile:
        # Create a mock chat request
        chat_request = ChatRequest(
            message="Hello, how are you?",
            thread_id="test_thread",
            attachments=[]
        )

        # Call the endpoint directly
        response = await chat_json_endpoint(chat_request, smile_instance)
        assert response.status_code == 200
        assert isinstance(response, StreamingResponse)

        # Read the streamed response
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())

        # Verify we got some response
        assert len(chunks) > 0
        assert any(len(chunk.strip()) > 0 for chunk in chunks)
        break

@pytest.mark.asyncio
async def test_chat_form_endpoint(real_smile, tmp_path):
    """
    Test the chat_form_endpoint function by providing form-like parameters and mock file uploads
    using the real Smile agent.
    """
    # Get the Smile instance
    async for smile_instance in real_smile:
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test file")
        
        # Create a mock file upload
        with open(test_file, "rb") as f:
            file = UploadFile(
                filename="test.txt",
                file=io.BytesIO(f.read())
            )

        # Call the endpoint directly
        response = await chat_form_endpoint(
            message="Here's a file",
            thread_id="test_thread",
            files=[file],
            smile_agent=smile_instance
        )
        assert response.status_code == 200
        assert isinstance(response, StreamingResponse)

        # Read the streamed response
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())

        # Verify we got some response
        assert len(chunks) > 0
        assert any(len(chunk.strip()) > 0 for chunk in chunks)
        break

@pytest.mark.asyncio
async def test_get_history_endpoint(real_smile):
    """
    Test get_history_endpoint by calling it directly with the real agent and a test thread_id.
    """
    test_thread_id = "test_thread_history"
    logger.info(f"[TEST] test_get_history_endpoint with thread_id='{test_thread_id}'")

    # Get the Smile instance
    async for smile_instance in real_smile:
        response = await get_history_endpoint(thread_id=test_thread_id, num_messages=50, smile_agent=smile_instance)
        assert response.status_code == 200
        assert response.body is not None
        break

@pytest.mark.asyncio
async def test_get_settings(real_smile):
    """
    Test the get_settings endpoint for a valid config_type with the real agent.
    """
    config_type = "app_config"
    logger.info(f"[TEST] test_get_settings with config_type='{config_type}'")

    response = await get_settings(config_type)
    logger.info(f"get_settings response: {response}")

    assert response["status"] == "success"
    assert isinstance(response["data"], dict), "Expected dictionary data for config"

@pytest.mark.asyncio
async def test_get_settings_invalid(real_smile):
    """
    Test the get_settings endpoint for an invalid config_type, expecting an HTTPException.
    """
    config_type = "invalid_config_type"
    logger.info(f"[TEST] test_get_settings_invalid with config_type='{config_type}'")

    with pytest.raises(HTTPException) as exc_info:
        await get_settings(config_type)
    logger.info(f"Raised HTTPException detail: {exc_info.value.detail}")

    assert "Invalid config_type" in exc_info.value.detail, "Expected HTTPException for invalid config_type"

@pytest.mark.asyncio
async def test_update_settings(real_smile):
    """
    Test updating settings by calling update_settings with a valid UpdateSettingsRequest.
    """
    update_request = UpdateSettingsRequest(
        config_type="app_config",
        settings_data={"test_new_setting": "test_value"}
    )
    logger.info(f"[TEST] test_update_settings with update_request={update_request.model_dump_json()}")

    # Typically you'll want to patch writes to disk if you don't want to modify real config files.
    with pytest.raises(HTTPException) as exc_info:
        await update_settings(update_request)

    # Expect a 500 if there's no valid config path or a success if your environment is configured for real updates.
    # Adjust your expectations here based on how your environment is set up.
    logger.info(f"HTTPException detail: {exc_info.value.detail}")

@pytest.mark.asyncio
async def test_update_settings_invalid(real_smile):
    """
    Test that update_settings raises an HTTPException for invalid config_type using the real agent.
    """
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
async def test_get_all_settings(real_smile):
    """
    Test the get_all_settings endpoint to verify it returns app_config and llm_config using the real agent.
    """
    logger.info("[TEST] test_get_all_settings")

    response = await get_all_settings()
    logger.info(f"get_all_settings response: {response}")

    assert response["status"] == "success"
    data = response["data"]
    assert "app_config" in data, "Expected 'app_config' in returned data"
    assert "llm_config" in data, "Expected 'llm_config' in returned data"

@pytest.mark.asyncio
async def test_process_attachments(real_smile):
    """Test that file attachments can be processed correctly"""
    try:
        # Create a test file
        test_content = "This is a test file content"
        test_attachments = [{
            'filename': 'test.txt',
            'content': test_content
        }]
        
        # Get smile instance from the generator
        async for smile_instance in real_smile:
            # Test streaming with attachment
            chunks = []
            async for chunk in smile_instance.stream("Process this file", attachments=test_attachments):
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
            
            # Verify response
            response_text = ''.join(chunks)
            assert len(response_text) > 0, "Response should not be empty"
            assert any(chunk.strip() for chunk in chunks), "At least one chunk should contain non-whitespace content"
            break  # We only need one instance
            
    except Exception as e:
        pytest.fail(f"Failed to process attachments: {str(e)}") 