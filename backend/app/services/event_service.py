from datetime import datetime
from typing import Optional, Type, TypeVar, List
from pytz import timezone
from sqlalchemy.orm import Session
from app.models.events import EventBase, EventEmailReceive
from app.models.event_store import EventStore, EventStatus

T = TypeVar('T', bound=EventBase)

class EventService:
    """Service layer for handling event persistence and retrieval"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_event(self, event: EventBase) -> EventStore:
        """
        Create a new event in the database.
        Converts Pydantic model to SQLAlchemy model.
        """
        # Extract common fields
        common_fields = {
            "id": event.id,
            "created_at": event.created_at,
            "event_type": event.event_type,
            "status": event.status,
            "priority": event.priority,
            "retry_count": event.retry_count,
            "max_retries": event.max_retries,
            "last_error": event.last_error,
            "event_metadata": event.metadata,
            "processed_at": event.processed_at
        }
        
        # Store event-specific data in event_data
        event_dict = event.model_dump()
        event_data = {k: v for k, v in event_dict.items() 
                     if k not in common_fields.keys()}
        
        # Create database model
        db_event = EventStore(
            **common_fields,
            event_data=event_data
        )
        
        self.db.add(db_event)
        self.db.commit()
        self.db.refresh(db_event)
        return db_event
    
    def get_event(self, event_id: str, model_class: Type[T]) -> Optional[T]:
        """
        Retrieve an event from the database and convert it to the appropriate Pydantic model.
        """
        db_event = self.db.query(EventStore).filter(EventStore.id == event_id).first()
        if not db_event:
            return None
            
        # Combine common fields with event-specific data
        event_data = {
            "id": db_event.id,
            "created_at": db_event.created_at,
            "event_type": db_event.event_type,
            "status": db_event.status,
            "priority": db_event.priority,
            "retry_count": db_event.retry_count,
            "max_retries": db_event.max_retries,
            "last_error": db_event.last_error,
            "metadata": db_event.event_metadata,
            "processed_at": db_event.processed_at,
            **db_event.event_data
        }
        
        return model_class(**event_data)
    
    def get_events_by_status(self, status: EventStatus, limit: int = 10, offset: int = 0) -> List[EventStore]:
        """Get events by status with pagination"""
        return self.db.query(EventStore)\
            .filter(EventStore.status == status)\
            .order_by(EventStore.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
    
    def get_all_events(self, limit: int = 10, offset: int = 0) -> List[EventStore]:
        """
        Get all events with pagination.
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of events ordered by creation date (newest first)
        """
        return self.db.query(EventStore)\
            .order_by(EventStore.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
    
    def get_event_by_id(self, event_id: str) -> Optional[EventStore]:
        """Get event by ID"""
        return self.db.query(EventStore)\
            .filter(EventStore.id == event_id)\
            .first()
    
    def delete_event(self, event_id: str) -> bool:
        """Delete event by ID"""
        event = self.get_event_by_id(event_id)
        if not event:
            return False
        self.db.delete(event)
        self.db.commit()
        return True
    
    def update_event_status(self, event_id: str, 
                          status: EventStatus, 
                          error: Optional[str] = None) -> bool:
        """Update event status and optionally set error message"""
        event = self.get_event_by_id(event_id)
        if not event:
            return False
            
        event.status = status
        if error:
            event.last_error = error
        if status == EventStatus.FAILED:
            event.retry_count += 1
        elif status == EventStatus.COMPLETED:
            event.processed_at = datetime.utcnow()
            
        self.db.commit()
        return True

# Usage example:
"""
# Create a new email event
email_event = EventEmailReceive(
    sender_email="user@example.com",
    recipient_email="service@ourapp.com",
    subject="Test",
    body="Test message"
)

# Store it in the database
event_service = EventService(db_session)
db_event = event_service.create_event(email_event)

# Retrieve it later
retrieved_event = event_service.get_event(
    db_event.id, 
    EventEmailReceive
)
""" 