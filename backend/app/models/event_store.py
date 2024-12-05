from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, JSON, Enum, Text
from sqlalchemy.dialects.postgresql import UUID
import enum
from app.database import Base  # Assuming you have a database.py with your SQLAlchemy setup

class EventStatus(str, enum.Enum):
    """Enum for event status tracking"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class EventStore(Base):
    """
    SQLAlchemy model for storing all events in a single table.
    Uses a JSON column for event-specific data.
    """
    __tablename__ = "events"

    # Common fields (matching EventBase)
    id = Column(UUID(as_uuid=True), primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    event_type = Column(String(50), nullable=False, index=True)
    status = Column(Enum(EventStatus), nullable=False, default=EventStatus.PENDING, index=True)
    priority = Column(Integer, nullable=False, default=1)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    last_error = Column(Text)
    event_metadata = Column(JSON, nullable=False, default=dict)
    processed_at = Column(DateTime)
    
    # Event-specific data stored as JSON
    event_data = Column(JSON, nullable=False)

    def __repr__(self):
        return f"<Event(id={self.id}, type={self.event_type}, status={self.status})>"

# Example of how to create database migration (you'll need alembic set up):
"""
# In your alembic migration file:

def upgrade():
    op.create_table(
        'events',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('status', sa.Enum('pending', 'processing', 'completed', 'failed', name='eventstatus'), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('last_error', sa.Text()),
        sa.Column('event_metadata', sa.JSON(), nullable=False),
        sa.Column('processed_at', sa.DateTime()),
        sa.Column('event_data', sa.JSON(), nullable=False)
    )
    op.create_index('ix_events_event_type', 'events', ['event_type'])
    op.create_index('ix_events_status', 'events', ['status'])

def downgrade():
    op.drop_index('ix_events_status')
    op.drop_index('ix_events_event_type')
    op.drop_table('events')
""" 