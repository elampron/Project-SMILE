from datetime import datetime
from typing import Annotated, Sequence, Optional, Dict, List, Union
from marshmallow import ValidationError
from pydantic import BaseModel, Field, model_validator, EmailStr
from langgraph.graph import add_messages, StateGraph
from uuid import UUID, uuid4
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from enum import Enum
from app.models.memory import ConversationSummary
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AttachmentType(str, Enum):
    """Supported attachment types"""
    TEXT = "text/plain"
    PDF = "application/pdf"
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    # Add more types as needed


class Attachment(BaseModel):
    """
    Model representing a file attachment.
    
    Attributes:
        id: Unique identifier for the attachment
        file_name: Original name of the file
        file_path: Path where the file is stored in the library
        mime_type: MIME type of the file
        content: Extracted text content from the file
        metadata: Additional metadata about the file
        created_at: When the attachment was created
        embedding_id: Optional ID of the embedding in Neo4j
    """
    id: UUID = Field(default_factory=uuid4)
    file_name: str
    file_path: Path
    mime_type: AttachmentType
    content: str
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedding_id: Optional[UUID] = None


def add_summaries(left: List[ConversationSummary], right: List[ConversationSummary]) -> List[ConversationSummary]:
    """Synchronous version of add_summaries for state management."""
    # Ensure inputs are lists
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    
    # Ensure all summaries have IDs
    for summary in left + right:
        if not summary.id:
            summary.id = str(uuid4())

    # Create index of existing summaries by ID
    left_idx_by_id = {str(s.id): i for i, s in enumerate(left)}
    
    # Create new list starting with existing summaries
    merged = left.copy()
    
    # Add or update with new summaries
    for summary in right:
        str_id = str(summary.id)
        if str_id in left_idx_by_id:
            # Replace existing summary
            merged[left_idx_by_id[str_id]] = summary
        else:
            # Add new summary
            merged.append(summary)

    return merged


class ExtractorType(str, Enum):
    ENTITY = "entity_extractor"
    SUMMARY = "conversation_summarizer"
    # Add other extractors as needed


class User(BaseModel):
    """
    Model representing the main user of the system.
    
    Attributes:
        id (UUID): Unique identifier for the user
        name (str): User's full name
        main_email (EmailStr): Primary email address
        person_id (Optional[UUID]): Reference to PersonEntity in Neo4j if exists
        created_at (datetime): Timestamp when user was created
        updated_at (Optional[datetime]): Timestamp of last update
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    main_email: EmailStr
    person_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None 


class AgentState(BaseModel):
    """
    State model for the agent.
    
    Attributes:
        messages: List of conversation messages
        summaries: List of conversation summaries
        attachments: List of file attachments for the current run
        summary: Optional current summary
        user_current_location: Optional user location
        current_mood: Optional user mood
    """
    messages: Annotated[list, add_messages]
    summaries: Annotated[List[ConversationSummary], add_summaries] = []
    attachments: List[Attachment] = []  # Regular list for current run attachments
    summary: Optional[str] = None
    user_current_location: Optional[str] = None
    current_mood: Optional[str] = None

