# Event System

A robust event processing system that handles various types of events (email, webhooks, SMS, etc.) with PostgreSQL storage and SQLAlchemy ORM.

## Features

- Event base model with common fields (ID, timestamps, status, priority, etc.)
- Support for different event types (Email, SMS, Webhook, etc.)
- PostgreSQL storage with JSON fields for event-specific data
- Database migrations using Alembic
- Connection pooling and health checks
- Retry mechanism built-in
- Type safety with Pydantic models

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
EVENT_DB_URL=postgresql://postgres:postgres@localhost:5432/events_db
```

4. Create the PostgreSQL database:
```bash
createdb events_db
```

## Database Setup

1. Initialize Alembic (if not already initialized):
```bash
alembic init alembic
```

2. Generate the initial migration:
```bash
alembic revision --autogenerate -m "Initial migration"
```

3. Apply the migration:
```bash
alembic upgrade head
```

## Project Structure

```
.
├── app/
│   ├── models/
│   │   ├── events.py        # Pydantic models for events
│   │   └── event_store.py   # SQLAlchemy models for database
│   ├── services/
│   │   └── event_service.py # Service layer for event handling
│   └── database.py          # Database configuration
├── alembic/
│   ├── versions/           # Migration files
│   └── env.py             # Alembic configuration
├── alembic.ini            # Alembic settings
├── requirements.txt       # Project dependencies
└── .env                  # Environment variables
```

## Usage Examples

### Creating a New Event

```python
from app.models.events import EventEmailReceive
from app.services.event_service import EventService
from app.database import get_db

# Create a new email event
email_event = EventEmailReceive(
    sender_email="user@example.com",
    recipient_email="service@ourapp.com",
    subject="Test Email",
    body="Hello, this is a test email",
    priority=1
)

# Store it in the database
with get_db() as db:
    event_service = EventService(db)
    db_event = event_service.create_event(email_event)
```

### Retrieving Events

```python
# Get a specific event
with get_db() as db:
    event_service = EventService(db)
    event = event_service.get_event(event_id, EventEmailReceive)

# Get pending events
with get_db() as db:
    event_service = EventService(db)
    pending_events = event_service.get_pending_events(limit=10)
```

### Updating Event Status

```python
from app.models.event_store import EventStatus

with get_db() as db:
    event_service = EventService(db)
    event_service.update_event_status(
        event_id="123", 
        status=EventStatus.COMPLETED
    )
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| EVENT_DB_URL | PostgreSQL connection URL | postgresql://postgres:postgres@localhost:5432/events_db |

## Database Schema

The events table uses a single-table inheritance pattern with these columns:

- `id`: UUID primary key
- `created_at`: Timestamp of event creation
- `event_type`: Type of event (email_receive, sms_receive, etc.)
- `status`: Event status (pending, processing, completed, failed)
- `priority`: Integer 1-5 (1 highest)
- `retry_count`: Number of retry attempts
- `max_retries`: Maximum retry attempts allowed
- `last_error`: Last error message if failed
- `metadata`: JSON field for additional metadata
- `processed_at`: Timestamp of processing completion
- `event_data`: JSON field for event-specific data

## Contributing

1. Create a new branch for your feature
2. Write tests for your changes
3. Submit a pull request

## License

[Your License Here]
