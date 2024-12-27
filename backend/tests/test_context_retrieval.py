"""Tests for context retrieval functionality."""

import logging
import pytest
from datetime import datetime
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
from app.services.neo4j import driver

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def memory_agent():
    """Create a SmileMemory instance for testing."""
    return SmileMemory()

@pytest.fixture
def test_memories(memory_agent):
    """Create test memories in the database."""
    memories = []
    
    # Create a pinned important preference
    preference = CognitiveMemory(
        type="BEHAVIORAL",
        content="User prefers dark mode for all applications",
        semantic=SemanticAttributes(
            confidence=1.0,
            importance=0.9,
            cognitive_aspects=[CognitiveAspect.BEHAVIORAL, CognitiveAspect.FACTUAL],
            structured_data={"theme": "dark"}
        ),
        temporal=TemporalContext(),
        validation=ValidationStatus(
            is_valid=True,
            validation_source=MemorySource.USER_CONFIRMATION,
            validation_notes="Direct user preference"
        ),
        relations=MemoryRelations(
            context={
                "topics": ["user interface", "preferences"],
                "entities": [],
                "participants": []
            }
        ),
        is_pinned=True,
        access_count=0,
        version=1
    )
    memories.append(preference)
    
    # Create a cognitive memory about a meeting
    meeting = CognitiveMemory(
        type="FACTUAL",
        content="Meeting with John about project timeline on March 14th",
        semantic=SemanticAttributes(
            confidence=0.95,
            importance=0.8,
            cognitive_aspects=[CognitiveAspect.FACTUAL, CognitiveAspect.TEMPORAL],
            structured_data={
                "event_type": "meeting",
                "participant": "John",
                "topic": "project timeline"
            }
        ),
        temporal=TemporalContext(
            timestamp=datetime.now(),
            duration="1 hour"
        ),
        validation=ValidationStatus(
            is_valid=True,
            validation_source=MemorySource.DIRECT_OBSERVATION,
            validation_notes="Created during conversation"
        ),
        relations=MemoryRelations(
            context={
                "topics": ["project", "meeting"],
                "entities": ["John"],
                "participants": ["John"]
            }
        ),
        access_count=0,
        version=1
    )
    memories.append(meeting)
    
    # Save memories to database
    saved_memories = []
    for memory in memories:
        try:
            saved = memory_agent.save_memory(memory)
            saved_memories.append(saved)
            logger.info(f"Saved test memory: {saved.id}")
        except Exception as e:
            logger.error(f"Error saving test memory: {str(e)}")
    
    yield saved_memories
    
    # Cleanup: Delete test memories
    with driver.session() as session:
        for memory in saved_memories:
            try:
                session.run(
                    "MATCH (n:CognitiveMemory) WHERE n.id = $memory_id DETACH DELETE n",
                    memory_id=str(memory.id)  # Convert UUID to string
                )
                logger.info(f"Deleted test memory: {memory.id}")
            except Exception as e:
                logger.error(f"Error deleting test memory: {str(e)}")

def test_get_context_basic(memory_agent, test_memories):
    """Test basic context retrieval functionality."""
    query = "What are the user's interface preferences?"
    context = memory_agent.get_context(query)
    
    # Verify context structure
    assert "PINNED KNOWLEDGE:" in context
    assert "RELEVANT KNOWLEDGE:" in context
    assert "dark mode" in context
    assert "Score:" in context
    
    logger.info("Basic context retrieval test passed")

def test_get_context_with_entities(memory_agent, test_memories):
    """Test context retrieval with entity relationships."""
    query = "Tell me about the meeting with John"
    context = memory_agent.get_context(query)
    
    # Verify entity-related content
    assert "John" in context
    assert "meeting" in context.lower()
    assert "project timeline" in context
    
    logger.info("Entity-based context retrieval test passed")

def test_get_context_ranking(memory_agent, test_memories):
    """Test that context results are properly ranked."""
    query = "What happened in recent meetings?"
    context = memory_agent.get_context(query)
    
    # Split into sections
    sections = context.split("\n\n")
    
    # Verify ranking logic
    for section in sections:
        if "Score:" in section:
            scores = [float(score.split()[-1].strip(")")) 
                     for score in section.split("\n") 
                     if "Score:" in score]
            # Check scores are in descending order
            assert all(scores[i] >= scores[i+1] 
                      for i in range(len(scores)-1))
    
    logger.info("Context ranking test passed")

def test_get_context_error_handling(memory_agent):
    """Test error handling in context retrieval."""
    # Test with empty query
    context = memory_agent.get_context("")
    assert context  # Should return empty or default context
    
    # Test with very long query
    long_query = "test " * 1000
    context = memory_agent.get_context(long_query)
    assert context  # Should handle long queries gracefully
    
    logger.info("Context error handling test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 