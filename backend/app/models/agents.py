from datetime import datetime
from typing import Annotated, Sequence, Optional, Dict, List, Union
from marshmallow import ValidationError
from pydantic import BaseModel, Field, model_validator, EmailStr
from langgraph.graph import add_messages, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from enum import Enum
from app.models.memory import ConversationSummary
import logging
from uuid import UUID, uuid4


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    messages: Annotated[list, add_messages]
    summaries: List[ConversationSummary] = []
    summary: Optional[str] = None  # Add this line
    user_current_location: Optional[str] = None
    current_mood: Optional[str] = None

    # @model_validator(mode='before')
    # def convert_messages(cls, values):
    #     messages = values.get('messages', [])
    #     converted_messages = []
    #     for msg in messages:
    #         if isinstance(msg, SmileMessage):
    #             converted_messages.append(msg)
    #         elif isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
    #             # Convert to SmileMessage
    #             converted_msg = convert_messages(msg)
    #             converted_messages.append(converted_msg)
    #         else:
    #             raise ValueError(f"Unsupported message type: {type(msg)}")
    #     values['messages'] = converted_messages
    #     return values

