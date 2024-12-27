"""Tool for saving memories directly to the knowledge graph."""

import logging
from typing import Optional, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
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

logger = logging.getLogger(__name__)

class SaveMemoryInput(BaseModel):
    """Input schema for SaveMemoryTool."""
    type: str = Field(..., description="Type of memory (e.g., FACTUAL, TEMPORAL, etc.)")
    content: str = Field(..., description="Content of the memory")
    importance: float = Field(default=0.5, description="Importance score (0.0-1.0)")
    confidence: float = Field(default=1.0, description="Confidence score (0.0-1.0)")
    cognitive_aspects: list[str] = Field(default_factory=lambda: ["FACTUAL"], description="List of cognitive aspects")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured representation of the memory")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context information including topics, entities, etc.")

class SaveMemoryTool(BaseTool):
    """Tool for saving memories directly to the knowledge graph."""
    name: ClassVar[str] = "save_memory"
    description: ClassVar[str] = """Save a memory directly to the knowledge graph.
    This tool allows for direct creation of memories without going through the langraph workflow.
    Input should include the memory type, content, and optional metadata."""
    args_schema: ClassVar[type[BaseModel]] = SaveMemoryInput
    memory_agent: SmileMemory = Field(default_factory=SmileMemory)

    def _run(
        self,
        type: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        cognitive_aspects: list[str] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the tool to save a memory.

        Args:
            type (str): Type of memory
            content (str): Content of the memory
            importance (float): Importance score (0.0-1.0)
            confidence (float): Confidence score (0.0-1.0)
            cognitive_aspects (list[str]): List of cognitive aspects
            structured_data (Dict[str, Any]): Structured representation
            context (Dict[str, Any]): Context information

        Returns:
            Dict[str, Any]: Information about the saved memory

        Raises:
            ValueError: If memory validation fails
            Exception: If there's an error saving the memory
        """
        try:
            # Create memory object
            memory = CognitiveMemory(
                type=type,
                content=content,
                semantic=SemanticAttributes(
                    confidence=confidence,
                    importance=importance,
                    cognitive_aspects=[CognitiveAspect(aspect) for aspect in (cognitive_aspects or ["FACTUAL"])],
                    structured_data=structured_data or {}
                ),
                temporal=TemporalContext(),
                validation=ValidationStatus(
                    is_valid=True,
                    validation_source=MemorySource.DIRECT_OBSERVATION,
                    validation_notes="Created via SaveMemoryTool"
                ),
                relations=MemoryRelations(
                    context=context or {
                        "topics": [],
                        "entities": [],
                        "participants": []
                    }
                ),
                access_count=0,
                version=1
            )

            # Save the memory
            saved_memory = self.memory_agent.save_memory(memory)
            
            return {
                "status": "success",
                "memory_id": str(saved_memory.id),
                "type": saved_memory.type,
                "content": saved_memory.content
            }

        except ValueError as ve:
            logger.error(f"Validation error saving memory: {str(ve)}")
            raise

        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            raise

    async def _arun(self, *args, **kwargs):
        """Async implementation of the tool."""
        return self._run(*args, **kwargs) 