"""
Main agent module for the SMILE agent.

This module contains the main Smile agent class that coordinates all functionality.
"""

import logging
from typing import Optional, List, Dict, Any, AsyncGenerator, Sequence
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase
from app.utils.logger import logger
from app.models.agents import AgentState, User, Attachment, AttachmentType
from app.models.memory import ConversationSummary, SmileDocument
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from app.configs.settings import settings
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from app.tools.custom_tools import (
    save_document,
    search_documents,
    search_entities,
    search_memories
)
from app.tools.public_tools import web_search_tool, file_tools
from app.tools.system.system_tools import execute_python, execute_cmd
from app.utils.llm import llm_factory
from app.services.postgres import PostgresSaver
from app.services.context import ContextManager
from app.services.embeddings import EmbeddingsService
from app.services.neo4j import driver
import time

from .initialization import (
    initialize_main_user,
    initialize_postgres,
    initialize_directories,
    cleanup
)
from .conversation import (
    get_conversation_history as get_history,
    stream_conversation,
    summarize_conversation
)
from .attachments import (
    process_attachments,
    save_document,
    get_document,
    delete_document
)

class Smile:
    """
    Main SMILE agent class that coordinates all functionality.
    
    This class is responsible for:
    - Initializing connections and resources
    - Managing conversations and message history
    - Processing attachments and documents
    - Interacting with the language model
    - Cleaning up resources
    """
    
    def __init__(self):
        """
        Initialize basic attributes.
        """
        self.settings = settings
        self.logger = logger
        self._initialized = False
        self.driver = driver
        
        # Set up PostgreSQL connection string
        self.postgres_url = settings.app_config.get("postgres_config")["conn"]
        
        # Set default thread ID
        self.thread_id = settings.app_config["langchain_config"]["thread_id"]
        
        # Initialize basic attributes
        self.chatbot_agent_llm = llm_factory(self.settings,"chatbot_agent")
        self.chatbot_agent_prompt = PromptTemplate.from_template(self.settings.llm_config["chatbot_agent"]["prompt_template"])
        self.embeddings_client = llm_factory(self.settings,"embeddings")
        self.graph = None
        self.tools = None
        self._checkpointer = None
        self._context_manager = ContextManager(self.driver)
        self._embeddings_service = EmbeddingsService()
        self.main_user = None
        self.attachments_dir = Path("library/attachments")
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize main user
        initialize_main_user(self)
        
    def call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Call the model and handle the response based on the state and config.
        
        Args:
            state (AgentState): Current state containing messages
            config (RunnableConfig): Configuration for the model
        
        Returns:
            AgentState: Updated state with model response
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get formatted messages
                formatted_messages = self.format_messages_for_model(state)
                
                # Get context from last messages
                # Filter out tool messages and get last 5 human/assistant messages
                last_messages = [msg for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))][-5:]
                user_input = "\n".join([msg.content for msg in last_messages if hasattr(msg, 'content')])
                context = self._context_manager.get_formatted_context(user_input)
                self.logger.info(f"Context: {context}")
                
                # Create prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.settings.llm_config.get("chatbot_agent").get("prompt_template")),
                    *formatted_messages
                ])

                # Create chain with tools
                chain = prompt | self.chatbot_agent_llm.bind_tools(self.tools)

                # Prepare prompt values
                prompt_values = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "context": context
                }

                # Get response
                response = chain.invoke(prompt_values)
                
                # Update state with new message
                return {"messages": [response]}
            
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Attempt {retry_count} failed: {str(e)}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise
                
                # Add exponential backoff
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
        
        raise Exception("Failed to get model response after all retries")

    def format_messages_for_model(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Format the state messages for the model.
        
        Args:
            state (AgentState): Current state containing messages
        
        Returns:
            List[Dict[str, Any]]: Formatted messages for the model
        """
        self.logger.debug(f"Formatting state for model: {state}")
        
        try:
            # Get the last max_messages from the state
            max_messages = self.settings.llm_config.get("chatbot_agent").get("max_messages", 10)
            last_messages = state.messages[-max_messages:]
            
            formatted_messages = [
                msg if not isinstance(msg, ToolMessage) else msg.__class__(
                    content=str(msg.content)[:1000],
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    id=msg.id
                )
                for msg in last_messages
            ]

            return formatted_messages
            
        except Exception as e:
            self.logger.error(f"Error in format_messages_for_model: {str(e)}")
            raise

    def should_continue(self, state: AgentState) -> str:
        """
        Determine whether the conversation should continue.
        
        Args:
            state (AgentState): Current state
            
        Returns:
            str: Next node to execute ("tools" or "__end__")
        """
        messages = state.messages
        last_message = messages[-1]
        
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "__end__"
        # Otherwise if there is, we continue
        return "tools"

    async def stream(self, message: str, config: Dict, attachments: List[Attachment] = None) -> AsyncGenerator[str, None]:
        """
        Stream method that handles both message and attachments.
        
        Args:
            message: The user's message
            config: Configuration dictionary
            attachments: Optional list of attachments for this run
            
        Yields:
            str: Response chunks
        """
        try:
            # Initialize state with empty attachments list
            inputs = {
                "messages": [("user", message)],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "attachments": []  # Start with empty list
            }
            
            if not self._initialized or not self._checkpointer:
                await self.initialize()
            if not config:
                config = {"thread_id": "MainThread"}

            # Add attachment content to the message if present
            if attachments:
                # Filter out None values and validate attachments
                valid_attachments = [
                    att for att in attachments 
                    if att is not None and isinstance(att, Attachment)
                ]
                
                if valid_attachments:
                    # Add valid attachments to the state for this run
                    inputs["attachments"] = valid_attachments
                    
                    # Add attachment content as system message for context
                    attachment_context = "\n\nAttached files:\n"
                    for attachment in valid_attachments:
                        attachment_context += (
                            f"\nFile: {attachment.file_name}\n"
                            f"Type: {attachment.mime_type.value}\n"
                            f"Content:\n{attachment.content}\n"
                            f"---\n"
                        )
                    inputs["messages"].append(("system", attachment_context))
                    self.logger.info(f"Added {len(valid_attachments)} valid attachments to context")
                else:
                    self.logger.warning("Received attachments list but no valid attachments found")

            self.logger.info(f"Starting stream with {len(inputs['attachments'])} attachments")
            for msg, metadata in self.graph.stream(
                inputs, 
                stream_mode="messages",
                config=config
            ):
                if isinstance(msg, (AIMessageChunk, AIMessage)):
                    if msg.content:
                        yield msg.content
                        
        except Exception as e:
            self.logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            raise

    async def get_conversation_history(self, thread_id: str, num_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: The ID of the conversation thread
            num_messages: Optional number of most recent messages to return
            
        Returns:
            List[Dict[str, Any]]: List of conversation messages
            
        Raises:
            Exception: If retrieval fails
        """
        return await get_history(self, thread_id, num_messages)
        
    async def initialize(self):
        """
        Initialize the SMILE agent.
        
        This method sets up all necessary connections and resources.
        It should be called after object creation, typically during application startup.
        
        Raises:
            Exception: If initialization fails
        """
        if self._initialized:
            self.logger.info("SMILE agent already initialized")
            return self
            
        try:
            # Initialize checkpointer
            if not self._initialized:
                try:
                    self.logger.info(f"Connecting to PostgreSQL at {self.postgres_url}...")
                    
                    # Create direct connection with optimized settings
                    from psycopg import Connection
                    conn = Connection.connect(
                        self.postgres_url,
                        autocommit=True,
                        prepare_threshold=None,  # Disable prepared statements
                        options="-c synchronous_commit=off"  # Optimize for performance
                    )
                    
                    # Create saver with the connection and use it directly as checkpointer
                    self._checkpointer = PostgresSaver(conn)
                    
                    # Setup tables using the checkpointer
                    await self._checkpointer.setup()
                    self.logger.info("PostgreSQL tables created successfully")
                    self.logger.info("Successfully connected to PostgreSQL")
                except Exception as pg_error:
                    self.logger.error(
                        "Failed to connect to PostgreSQL. Please ensure:\n"
                        "1. PostgreSQL is running\n"
                        "2. The connection string in app_config.yaml is correct\n"
                        "3. If running locally, use 'localhost' instead of 'postgres' as the host\n"
                        f"Error details: {str(pg_error)}"
                    )
                    raise
                
                self._initialized = True
            
            # Initialize graph if needed
            if self.graph is None:
                await self._initialize_graph()
            
            return self
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            if hasattr(self, '_checkpointer') and hasattr(self._checkpointer, 'conn'):
                try:
                    self._checkpointer.conn.close()
                except Exception:
                    pass
            raise

    async def _initialize_graph(self) -> None:
        """Initialize the agent graph with tools and checkpointer."""
        
        # Initialize tools
        tools = [
            # save_document,
            search_documents,
            search_entities,
            search_memories,
            web_search_tool,
            *file_tools,
            execute_python,
            execute_cmd
        ]
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Create tool node
        tool_node = ToolNode(tools)
        
        # Add nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue
        )
        
        workflow.add_edge("tools", "agent")
        
        # Compile workflow
        self.graph = workflow.compile(
            checkpointer=self._checkpointer,
            interrupt_before=None,
            interrupt_after=None,
            debug=False
        )
        
        # Store tools for later use
        self.tools = tools
        
        logger.info("Successfully initialized agent graph")
        
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, '_checkpointer') and self._checkpointer:
            try:
                self._checkpointer = None
                self._initialized = False
                logger.info("Successfully cleaned up Smile agent")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                raise
        
    def __del__(self):
        """Cleanup resources on deletion."""
        cleanup(self)
        
    # Expose key functionality through the main class
    
    def summarize(self, thread_id: str) -> ConversationSummary:
        """Create conversation summary."""
        return summarize_conversation(self, thread_id) 