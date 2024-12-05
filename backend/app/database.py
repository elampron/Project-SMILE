from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable or use default
EVENT_DB_URL = os.getenv(
    "EVENT_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/events_db"
)

# Create SQLAlchemy engine
engine = create_engine(
    EVENT_DB_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,         # Maximum number of connections to keep in the pool
    max_overflow=10      # Maximum number of connections that can be created beyond pool_size
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Yields:
        Session: Database session
        
    Usage:
        # In FastAPI:
        @app.get("/events/")
        def get_events(db: Session = Depends(get_db)):
            ...
            
        # In scripts:
        with get_db() as db:
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """
    Initialize database by creating all tables.
    Should be called when application starts.
    """
    Base.metadata.create_all(bind=engine)

def dispose_db() -> None:
    """
    Dispose of the database engine.
    Should be called when application shuts down.
    """
    engine.dispose() 