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

class SmileMessage(BaseMessage):
    """
    Custom message class for Smile.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the message")
    timestamp: datetime = Field(..., description="Timestamp of the message in ISO 8601 format.")
    extractors_processed: Dict[ExtractorType, bool] = Field(
        default_factory=lambda: {extractor: False for extractor in ExtractorType},
        description="Processing status of each extractor for this message."
    )
    content: str
    type: str
    processed: bool = Field(default=False, description="Flag indicating if message has been processed")
    updated_at: Optional[datetime] = None
    summarized: bool = Field(default=False, description="Flag indicating if message has been summarized")

    @classmethod
    def from_base_message(cls, message):
        """
        Class method to create a SmileMessage from a base message.

        Args:
            message: The base message object containing message data.

        Returns:
            SmileMessage: An instance of SmileMessage populated with data from the base message.
        """
        logger.debug(f"Converting base message to SmileMessage: {message}")

        data = {
            'id': uuid4(),
            'content': message.content,
            'type': message.type if hasattr(message, 'type') else message.__class__.__name__,
            'timestamp': datetime.utcnow(),  # Use current time if not available
            'processed': False,
            'summarized': False
        }

        smile_message = cls(**data)

        logger.debug(f"Created SmileMessage: {smile_message}")

        return smile_message

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

