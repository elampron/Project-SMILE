from typing import Dict, TypeVar, Annotated, Sequence
from langgraph.graph import Graph
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field

from app.models.events import EventBase
from app.models.event_store import EventStatus

class EventProcessorState(BaseModel):
    """State for the event processing workflow"""
    event: EventBase
    analysis: Dict = Field(default_factory=dict)
    error: str | None = None
    status: EventStatus = EventStatus.PENDING

async def event_processor(
    state: EventProcessorState,
    #llm_config: dict,  # We'll add this later
) -> EventProcessorState:
    """
    Main node for processing events.
    Analyzes the event using an LLM and updates the state with the analysis.
    """
    
    # TODO: Add LLM call here
    # For now, just return the state
    return state

async def handle_error(
    state: EventProcessorState,
) -> EventProcessorState:
    """Handle any errors that occur during event processing"""
    if state.error:
        state.status = EventStatus.FAILED
    return state

def create_event_processor_graph() -> Graph:
    """
    Creates the event processor workflow graph.
    
    The graph follows this flow:
    1. event_processor node analyzes the event
    2. If error occurs, handle_error node is called
    3. Returns final state
    """
    workflow = Graph()
    
    # Add the main event processor node
    workflow.add_node("event_processor", event_processor)
    workflow.add_node("handle_error", handle_error)
    
    # Define the edges
    workflow.add_edge("event_processor", "handle_error")
    
    # Set the entry point
    workflow.set_entry_point("event_processor")
    
    # Set the exit point
    workflow.set_finish_point("handle_error")
    
    return workflow

class EventProcessor:
    """
    Main class for processing events using the LangGraph workflow.
    """
    def __init__(self):
        self.graph = create_event_processor_graph()
    
    async def process_event(self, event: EventBase) -> EventProcessorState:
        """
        Process a single event through the workflow.
        
        Args:
            event: The event to process
            
        Returns:
            EventProcessorState: The final state after processing
        """
        # Initialize the state
        initial_state = EventProcessorState(event=event)
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state

# Usage example:
"""
# Initialize the processor
processor = EventProcessor()

# Process an event
event = EventEmailReceive(
    sender_email="user@example.com",
    recipient_email="service@ourapp.com",
    subject="Test",
    body="Hello, this is a test email"
)

# Process the event
result = await processor.process_event(event)

# Check the result
print(f"Status: {result.status}")
print(f"Analysis: {result.analysis}")
if result.error:
    print(f"Error: {result.error}")
"""
