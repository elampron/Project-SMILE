import logging
import pytest
from datetime import datetime
from pydantic import ValidationError
from app.agents.memory import SmileMemory
from app.models.memory import (
    CognitiveMemory,
    SemanticAttributes,
    TemporalContext,
    ValidationStatus,
    MemoryRelations,
    CognitiveAspect,
    MemorySource
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def memory_agent():
    """Create a memory agent instance for testing."""
    return SmileMemory()

@pytest.fixture
def test_memory():
    """Create a test memory instance."""
    return CognitiveMemory(
        type="FACTUAL",
        content="This is a test memory for unit testing.",
        semantic=SemanticAttributes(
            confidence=0.95,
            importance=0.8,
            cognitive_aspects=[CognitiveAspect.FACTUAL, CognitiveAspect.PROCEDURAL],
            structured_data={"test": True, "purpose": "unit testing"}
        ),
        temporal=TemporalContext(
            timestamp=datetime.utcnow(),
            duration=None,
            recurrence=None
        ),
        validation=ValidationStatus(
            is_valid=True,
            validation_source=MemorySource.DIRECT_OBSERVATION,
            validation_notes="Created during unit testing",
            last_validated=datetime.utcnow()
        ),
        relations=MemoryRelations(
            context={
                "topics": ["testing", "memory", "unit tests"],
                "entities": [],
                "participants": []
            }
        ),
        access_count=0,
        version=1
    )

def test_save_memory(memory_agent, test_memory):
    """Test saving a memory directly."""
    try:
        # Save the memory
        saved_memory = memory_agent.save_memory(test_memory)
        
        # Verify the memory was saved
        assert saved_memory is not None
        assert saved_memory.id is not None
        assert saved_memory.type == "FACTUAL"
        assert saved_memory.content == "This is a test memory for unit testing."
        assert saved_memory.embedding is not None
        
        logger.info(f"Successfully saved and verified memory: {saved_memory.id}")
        
    except Exception as e:
        logger.error(f"Error in test_save_memory: {str(e)}")
        raise

def test_save_memory_with_entities(memory_agent):
    """Test saving a memory with entity relationships."""
    memory_with_entities = CognitiveMemory(
        type="RELATIONSHIP",
        content="John works at Acme Corp as a software engineer.",
        semantic=SemanticAttributes(
            confidence=0.9,
            importance=0.7,
            cognitive_aspects=[CognitiveAspect.FACTUAL, CognitiveAspect.SOCIAL],
            structured_data={
                "employment": {
                    "role": "software engineer",
                    "company": "Acme Corp"
                }
            }
        ),
        temporal=TemporalContext(
            timestamp=datetime.utcnow(),
            duration=None,
            recurrence=None
        ),
        validation=ValidationStatus(
            is_valid=True,
            validation_source=MemorySource.DIRECT_OBSERVATION,
            validation_notes="Created during entity relationship testing",
            last_validated=datetime.utcnow()
        ),
        relations=MemoryRelations(
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
        ),
        access_count=0,
        version=1
    )
    
    try:
        # Save the memory
        saved_memory = memory_agent.save_memory(memory_with_entities)
        
        # Verify the memory was saved
        assert saved_memory is not None
        assert saved_memory.id is not None
        assert saved_memory.type == "RELATIONSHIP"
        assert "John" in saved_memory.content
        assert "Acme Corp" in saved_memory.content
        assert saved_memory.embedding is not None
        
        logger.info(f"Successfully saved and verified memory with entities: {saved_memory.id}")
        
    except Exception as e:
        logger.error(f"Error in test_save_memory_with_entities: {str(e)}")
        raise

def test_save_invalid_memory(memory_agent):
    """Test saving an invalid memory."""
    try:
        invalid_memory = CognitiveMemory(
            type="INVALID",  # Invalid type
            content="",  # Empty content
            semantic=SemanticAttributes(
                confidence=-1,  # Invalid confidence
                importance=2.0,  # Invalid importance
                cognitive_aspects=[],  # Empty cognitive aspects
                structured_data={}
            ),
            temporal=TemporalContext(
                timestamp=None,
                duration=None,
                recurrence=None
            ),
            validation=ValidationStatus(
                is_valid=False,
                validation_source=MemorySource.SYSTEM_ANALYSIS,
                validation_notes="Invalid test memory",
                last_validated=datetime.utcnow()
            ),
            relations=MemoryRelations(
                context={}
            ),
            access_count=0,
            version=1
        )
        pytest.fail("Should have raised ValidationError")
    except ValidationError as ve:
        # Verify that the validation error contains the expected messages
        error_messages = str(ve)
        assert "Content cannot be empty" in error_messages
        logger.info("Successfully caught validation error for empty content")
        
    try:
        # Try with valid content but invalid type
        invalid_memory = CognitiveMemory(
            type="INVALID",  # Invalid type
            content="This is a test memory with an invalid type.",
            semantic=SemanticAttributes(
                confidence=0.5,
                importance=0.5,
                cognitive_aspects=[CognitiveAspect.FACTUAL],
                structured_data={}
            ),
            temporal=TemporalContext(
                timestamp=datetime.utcnow(),
                duration=None,
                recurrence=None
            ),
            validation=ValidationStatus(
                is_valid=False,
                validation_source=MemorySource.SYSTEM_ANALYSIS,
                validation_notes="Invalid test memory",
                last_validated=datetime.utcnow()
            ),
            relations=MemoryRelations(
                context={}
            ),
            access_count=0,
            version=1
        )
        pytest.fail("Should have raised ValidationError")
    except ValidationError as ve:
        # Verify that the validation error contains the expected messages
        error_messages = str(ve)
        assert "Invalid memory type" in error_messages
        logger.info("Successfully caught validation error for invalid type")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 