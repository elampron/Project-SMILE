from datetime import datetime
from typing import Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class EventBase(BaseModel):
    """
    Base class for all events in the system.
    Contains common fields that all events must have.
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    priority: int = Field(default=1, ge=1, le=5)  # 1 is highest priority, 5 is lowest
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "email_receive",
                "priority": 1,
                "metadata": {"source": "external_webhook"}
            }
        }

class EventEmailReceive(EventBase):
    """
    Event model for receiving emails.
    Contains email-specific fields in addition to base event fields.
    """
    event_type: Literal["email_receive"] = "email_receive"
    sender_email: str
    recipient_email: str
    subject: str
    body: str
    attachments: Optional[Dict[str, str]] = None  # filename -> content_type
    email_headers: Optional[Dict[str, str]] = None
    spam_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_email": "user@example.com",
                "recipient_email": "service@ourapp.com",
                "subject": "New inquiry",
                "body": "Hello, I have a question...",
                "spam_score": 0.1
            }
        }

