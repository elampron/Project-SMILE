from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app.models.events import EventBase, EventEmailReceive
from app.models.event_store import EventStatus
from app.services.event_service import EventService
from app.agents.event_processor import EventProcessor

router = APIRouter(
    prefix="/events",
    tags=["events"],
    responses={404: {"description": "Not found"}},
)

# Initialize event processor
event_processor = EventProcessor()

@router.post("/", response_model=dict)
async def create_event(
    event: EventEmailReceive,  # For now, we only support email events
    db: Session = Depends(get_db)
):
    """
    Create a new event and trigger its processing.
    """
    try:
        # Store the event
        event_service = EventService(db)
        db_event = event_service.create_event(event)
        
        # Process the event
        process_result = await event_processor.process_event(event)
        
        # Update event status based on processing result
        event_service.update_event_status(
            str(db_event.id),
            process_result.status,
            process_result.error
        )
        
        return {
            "message": "Event created and processed",
            "event_id": str(db_event.id),
            "status": process_result.status,
            "analysis": process_result.analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[dict])
async def list_events(
    status: Optional[EventStatus] = None,
    limit: int = Query(default=10, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List events with optional filtering by status.
    """
    try:
        event_service = EventService(db)
        if status:
            events = event_service.get_events_by_status(status, limit, offset)
        else:
            events = event_service.get_all_events(limit, offset)
        
        return [
            {
                "id": str(event.id),
                "event_type": event.event_type,
                "status": event.status,
                "created_at": event.created_at,
                "processed_at": event.processed_at,
                "event_data": event.event_data
            }
            for event in events
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{event_id}", response_model=dict)
async def get_event(
    event_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific event by ID.
    """
    try:
        event_service = EventService(db)
        event = event_service.get_event_by_id(str(event_id))
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {
            "id": str(event.id),
            "event_type": event.event_type,
            "status": event.status,
            "created_at": event.created_at,
            "processed_at": event.processed_at,
            "event_data": event.event_data,
            "metadata": event.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{event_id}")
async def delete_event(
    event_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a specific event by ID.
    """
    try:
        event_service = EventService(db)
        success = event_service.delete_event(str(event_id))
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {"message": "Event deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{event_id}/reprocess")
async def reprocess_event(
    event_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Reprocess a specific event by ID.
    """
    try:
        event_service = EventService(db)
        event = event_service.get_event_by_id(str(event_id))
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Convert DB event to Pydantic model
        event_model = EventEmailReceive(**event.event_data)
        
        # Reprocess the event
        process_result = await event_processor.process_event(event_model)
        
        # Update event status
        event_service.update_event_status(
            str(event_id),
            process_result.status,
            process_result.error
        )
        
        return {
            "message": "Event reprocessed",
            "status": process_result.status,
            "analysis": process_result.analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
