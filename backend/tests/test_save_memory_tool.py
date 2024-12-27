"""Tests for the SaveMemoryTool."""

import logging
import pytest
from app.tools.memory import SaveMemoryTool
from app.models.memory import CognitiveAspect

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def save_memory_tool():
    """Create a SaveMemoryTool instance for testing."""
    return SaveMemoryTool()

def test_save_memory_basic(save_memory_tool):
    """Test basic memory saving functionality."""
    try:
        result = save_memory_tool._run(
            type="FACTUAL",
            content="This is a test memory created via SaveMemoryTool.",
            importance=0.8,
            confidence=0.9,
            cognitive_aspects=[CognitiveAspect.FACTUAL.value, CognitiveAspect.PROCEDURAL.value],
            structured_data={"test": True},
            context={
                "topics": ["testing", "memory creation"],
                "entities": [],
                "participants": []
            }
        )
        
        # Verify the result
        assert result["status"] == "success"
        assert result["memory_id"] is not None
        assert result["type"] == "FACTUAL"
        assert "test memory" in result["content"]
        
        logger.info(f"Successfully saved memory: {result['memory_id']}")
        
    except Exception as e:
        logger.error(f"Error in test_save_memory_basic: {str(e)}")
        raise

def test_save_memory_with_entities(save_memory_tool):
    """Test saving a memory with entity relationships."""
    try:
        result = save_memory_tool._run(
            type="RELATIONSHIP",
            content="John works at Acme Corp as a software engineer.",
            importance=0.7,
            confidence=0.9,
            cognitive_aspects=[CognitiveAspect.FACTUAL.value, CognitiveAspect.SOCIAL.value],
            structured_data={
                "employment": {
                    "role": "software engineer",
                    "company": "Acme Corp"
                }
            },
            context={
                "topics": ["employment", "professional relationship"],
                "entities": [
                    {
                        "type": "person",
                        "name": "John",
                        "category": "Professional"
                    },
                    {
                        "type": "organization",
                        "name": "Acme Corp",
                        "category": "Employer"
                    }
                ],
                "participants": [
                    {
                        "name": "John",
                        "role": "Employee"
                    }
                ]
            }
        )
        
        # Verify the result
        assert result["status"] == "success"
        assert result["memory_id"] is not None
        assert result["type"] == "RELATIONSHIP"
        assert "John" in result["content"]
        assert "Acme Corp" in result["content"]
        
        logger.info(f"Successfully saved memory with entities: {result['memory_id']}")
        
    except Exception as e:
        logger.error(f"Error in test_save_memory_with_entities: {str(e)}")
        raise

def test_save_invalid_memory(save_memory_tool):
    """Test saving an invalid memory."""
    with pytest.raises(ValueError):
        save_memory_tool._run(
            type="INVALID",  # Invalid type
            content="",  # Empty content
            importance=-1,  # Invalid importance
            confidence=2.0,  # Invalid confidence
            cognitive_aspects=[],  # Empty cognitive aspects
            structured_data={},
            context={}
        )
        
    logger.info("Successfully caught validation error for invalid memory")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 