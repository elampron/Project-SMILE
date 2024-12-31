# app/agents/smile.py
import logging
from typing import Annotated, Sequence, TypedDict, Dict, List, Generator
from app.services.neo4j.users import get_user_by_email
from typing_extensions import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from app.configs.settings import settings, Settings
from app.tools.public_tools import web_search_tool, file_tools 
from langgraph.checkpoint.postgres import PostgresSaver
from app.tools.custom_tools import (
    execute_python,
    execute_cmd,
    save_document,
    SearchDocumentsTool,
    SearchEntitiesTool,
    SearchMemoriesTool
)
from app.utils.llm import llm_factory, prepare_conversation_data
from app.models.agents import AgentState, User, Attachment, AttachmentType
from app.services.neo4j import create_or_update_user, get_or_create_person_entity
from app.services.neo4j import driver
from app.agents.context import ContextManager
from datetime import datetime
import time
from app.services.embeddings import EmbeddingsService
from pathlib import Path
import os
from langchain.tools import StructuredTool
from pydantic import BaseModel
from app.utils.logger import logger



class Smile:
    def __init__(self):
        """
        Initialize Smile agent with main user configuration.
        
        Raises:
            ValueError: If required main user config is missing
        """
        self.settings = settings
        
        # Create a logger
        self.logger = logger
        
        # Initialize basic attributes
        self.chatbot_agent_llm = llm_factory(self.settings,"chatbot_agent")
        self.chatbot_agent_prompt = PromptTemplate.from_template(self.settings.llm_config["chatbot_agent"]["prompt_template"])
        self.embeddings_client = llm_factory(self.settings,"embeddings")
        self.db_path = ".//checkpoints//smile.db"
        self.postgres_url = settings.app_config.get("postgres_config")["conn"]
        self.graph = None
        self.tools = None
        self._checkpointer = None
        self._initialized = False
        self._saver = None
        self.context_manager = ContextManager(driver)
        self.embeddings_service = EmbeddingsService()
        
        # Create attachments directory if it doesn't exist
        self.attachments_dir = Path("library/attachments")
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize main user
        self._initialize_main_user()

    def _initialize_main_user(self):
        """
        Initialize the main user from config and persist to Neo4j.
        Creates or updates both User and PersonEntity records.
        """
        try:
            # Get main user config
            user_config = self.settings.app_config.get("main_user")
            langchain_config = self.settings.app_config.get("langchain_config")
            if not user_config:
                raise ValueError("Main user configuration is missing in app_config.yaml")

            # Create User instance
            self.main_user = User(
                name=user_config["name"],
                main_email=user_config["main_email"]
            )
            self.thread_id = langchain_config["thread_id"]

            # Persist user to Neo4j and get updated user with person_id
            with driver.session() as session:
                # First check if user exists
                existing_user = session.execute_read(
                    get_user_by_email,
                    self.main_user.main_email
                )
                
                if existing_user:
                    self.main_user = existing_user
                    self.logger.info(f"Using existing main user: {self.main_user.name}")
                    return

                # Create/update user record only if it doesn't exist
                # This will also create the corresponding Person node
                self.main_user = session.execute_write(
                    create_or_update_user, 
                    self.main_user
                )
                
                self.logger.info(f"Main user initialized: {self.main_user.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize main user: {str(e)}")
            raise

    def save_document(self, content: str, filename: str) -> Attachment:
        """
        Save a document to the filesystem and create embeddings.
        
        Args:
            content (str): The content of the document
            filename (str): The name of the file
            
        Returns:
            Attachment: The created attachment object
            
        Raises:
            Exception: If there's an error saving the document
        """
        try:
            self.logger.info(f"Saving document: {filename}")
            
            # Determine file path and MIME type
            file_path = self.attachments_dir / filename
            mime_type = AttachmentType.TEXT  # Default to text for now
            
            # Save file to disk
            file_path.write_text(content)
            
            # Create embeddings and save to Neo4j
            embedding_id = self.embeddings_service.create_document_node(
                content=content,
                metadata={
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "user_id": str(self.main_user.id) if hasattr(self, 'main_user') else None
                }
            )
            
            # Create attachment object
            attachment = Attachment(
                file_name=filename,
                file_path=file_path,
                mime_type=mime_type,
                content=content,
                metadata={
                    "user_id": str(self.main_user.id) if hasattr(self, 'main_user') else None
                },
                embedding_id=embedding_id
            )
            
            self.logger.info(f"Successfully saved document: {filename}")
            return attachment
            
        except Exception as e:
            self.logger.error(f"Error saving document {filename}: {str(e)}")
            raise

    def initialize(self):
        """Synchronously initialize the agent graph and tools."""
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
                    self._checkpointer.setup()
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
                self._initialize_graph()
            
            return self
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            if hasattr(self, '_checkpointer') and hasattr(self._checkpointer, 'conn'):
                try:
                    self._checkpointer.conn.close()
                except Exception:
                    pass
            raise
    
    def call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Call the model and handle the response based on the state and config.
        
        Args:
            state (AgentState): Current state containing messages
            config (RunnableConfig): Configuration for the model
        
        Returns:
            AgentState: Updated state with model response
        
        Raises:
            Exception: If all retry attempts fail
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
                context = self.context_manager.get_formatted_context(user_input)
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
    

    def should_continue(self, state: AgentState) -> Literal["tools", "__end__"]:
        messages = state.messages
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "__end__"
        # Otherwise if there is, we continue
        else:
            return "tools"
        
    def cleanup(self):
        """Cleanup resources."""
        if self._saver and self._initialized:
            try:
                self._saver.__exit__(None, None, None)
                self._initialized = False
                self._checkpointer = None
                self._saver = None
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
                raise

    def _initialize_graph(self):
        """Initialize the agent graph with tools and checkpointer."""
        self.logger.info("Initializing agent graph")
        
        # Create a schema for the reload tool
        class ReloadSchema(BaseModel):
            """Schema for the reload tool - no parameters needed"""
            pass
        
        # Create a structured tool for reload
        reload_tool = StructuredTool(
            name="reload",
            description="Reload Smile's configuration and reinitialize the agent",
            func=lambda: self.reload(),
            args_schema=ReloadSchema,
            return_direct=True
        )
        
        self.tools = [
            web_search_tool,
            *file_tools,
            execute_python,
            execute_cmd,
            save_document,  # Save document tool
            SearchDocumentsTool(),  # Document search
            SearchEntitiesTool(),  # Entity search
            SearchMemoriesTool(),  # Memory search
            reload_tool  # Reload tool
        ]
          # Define a new graph
        workflow = StateGraph(AgentState)
        tool_node = ToolNode(self.tools)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
        )

        workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.graph=workflow.compile(
            checkpointer=self._checkpointer,
            interrupt_before=None,
            interrupt_after=None,
            debug=False,
        )


        
        self.logger.info("Agent graph initialized successfully")

    def get_conversation_history(self, num_messages=50, thread_id="MainThread"):
        """
        Retrieve the last `num_messages` messages from the conversation history using LangGraph checkpointing.
        
        Args:
            num_messages (int): The number of messages to retrieve. Defaults to 50.
            thread_id (str): The thread ID for which to retrieve the conversation. Defaults to "MainThread".
        
        Returns:
            List[Dict]: A list of messages with their content, role, and timestamp.
        """
        try:
            if not self._initialized or not self._checkpointer:
                self.initialize()
                
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get all states for this thread
            
            state = self.graph.get_state(config)
            
            
            # Get the latest state's messages
            latest_state = state
            messages = latest_state.values.get('messages', [])
            
            # Prepare the conversation history
            conversation_history = []
            for msg in messages[-num_messages:]:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    message = {
                        "role": "human" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content if hasattr(msg, 'content') else str(msg),
                        "timestamp": getattr(msg, 'timestamp', 
                                          msg.additional_kwargs.get('timestamp', 
                                          datetime.now().isoformat())),
                        "message_id": msg.id if hasattr(msg, 'id') else None
                    }
                    conversation_history.append(message)

            self.logger.info(f"Retrieved {len(conversation_history)} messages from conversation history")
            return conversation_history

        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
            raise

    def stream(self, message: str, config: Dict, attachments: List[Attachment] = None) -> Generator[str, None, None]:
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
                self.initialize()
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

    def format_messages_for_model(self, state: AgentState):
        """
        Format the state for the model and return a properly formatted message.
        
        Args:
            state (AgentState): Current state containing messages and other info
        
        Returns:
            str: Formatted message for the model
        """
        # Add logging
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
            self.logger.error(f"Error in format_for_model: {str(e)}", exc_info=True)
            raise
    
    def save_document(self, content: str, filename: str) -> None:
        """
        Save a document to the vector store and create embeddings.
        
        Args:
            content (str): The content of the document
            filename (str): The name of the file
            
        Raises:
            Exception: If there's an error saving the document
        """
        try:
            self.logger.info(f"Saving document: {filename}")
            
            # Create embeddings and save to Neo4j
            self.embeddings_service.create_document_node(
                content=content,
                metadata={
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "user_id": str(self.main_user.id) if hasattr(self, 'main_user') else None
                }
            )
            
            self.logger.info(f"Successfully saved document: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving document {filename}: {str(e)}")
            raise
    
    def reload(self) -> str:
        """
        Reload Smile's configuration and reinitialize the agent.
        
        This method reloads the configuration files and reinitializes the agent with the new settings.
        Use this after making changes to configuration files to apply them without restarting the server.
        
        Returns:
            str: Status message indicating success or failure
        """
        try:
            self.logger.info("Reloading Smile configuration...")
            
            # Create new settings instance
            new_settings = Settings()
            self.settings = new_settings
            
            # Re-initialize main user
            self._initialize_main_user()
            
            # Re-initialize basic attributes
            self.chatbot_agent_llm = llm_factory(self.settings, "chatbot_agent")
            self.chatbot_agent_prompt = PromptTemplate.from_template(self.settings.llm_config["chatbot_agent"]["prompt_template"])
            self.embeddings_client = llm_factory(self.settings, "embeddings")
            
            # Re-initialize the agent
            self.initialize()
            
            self.logger.info("Smile configuration reloaded successfully")
            return "Configuration reloaded successfully"
            
        except Exception as e:
            error_msg = f"Error reloading configuration: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    

