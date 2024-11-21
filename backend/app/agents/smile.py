from datetime import datetime
import logging
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from app.configs.settings import settings
from app.tools.public_tools import web_search_tool, file_tools
from langgraph.checkpoint.sqlite import SqliteSaver
from app.tools.custom_tools import execute_python, execute_cmd  # Importing both custom tools
from app.utils.llm import llm_factory
from app.models.agents import AgentState, User
from app.services.neo4j import create_or_update_user, get_or_create_person_entity
from app.services.neo4j import driver





class Smile:
    def __init__(self):
        """
        Initialize Smile agent with main user configuration.
        
        Raises:
            ValueError: If required main user config is missing
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.info("Smile class logger initialized")
        
        # Initialize main user
        self._initialize_main_user()
        
        # Rest of existing initialization...
        self.chatbot_agent_llm = llm_factory(self.settings,"chatbot_agent")
        self.embeddings_client = llm_factory(self.settings,"embeddings")
        self.db_path = "..//checkpoints//smile.db"

    def _initialize_main_user(self):
        """
        Initialize the main user from config and persist to Neo4j.
        Creates or updates both User and PersonEntity records.
        """
        try:
            # Get main user config
            user_config = self.settings.app_config.get("main_user")
            if not user_config:
                raise ValueError("Main user configuration is missing in app_config.yaml")

            # Create User instance
            self.main_user = User(
                name=user_config["name"],
                main_email=user_config["main_email"]
            )

            # Persist user to Neo4j and get updated user with person_id
            with driver.session() as session:
                # Create/update user record
                self.main_user = session.execute_write(
                    create_or_update_user, 
                    self.main_user
                )
                
                # Create/update person entity if person details provided
                if person_details := user_config.get("person_details"):
                    person_details["name"] = user_config["name"]  # Ensure name matches
                    self.main_user_person = session.execute_write(
                        get_or_create_person_entity,
                        person_details
                    )
                    
                    # Update user with person_id if needed
                    if not self.main_user.person_id:
                        self.main_user.person_id = self.main_user_person.id
                        self.main_user = session.execute_write(
                            create_or_update_user,
                            self.main_user
                        )

            self.logger.info(f"Main user initialized: {self.main_user.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize main user: {str(e)}")
            raise

    def format_for_model(self, state: AgentState):
        return self.prompt.invoke({"messages": state["messages"][-self.settings.llm_config.get("chatbot_agent").get("max_messages"):],"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    
    
    def stream(self, user_input: str,config = None):
        llm_config = self.settings.llm_config.get("chatbot_agent")

        if not llm_config:
            self.logger.error("Chatbot agent not found")
            return

        if not llm_config.get("prompt_template"):
            self.logger.error("Chatbot agent prompt template not found", extra={"llm_config": llm_config})
            return

        tools = [web_search_tool] + file_tools + [execute_python, execute_cmd]
       

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", llm_config["prompt_template"]),
            ("placeholder", "{messages}"),
            ("system", "Current date and time: {time}"),
        ])

       

        with SqliteSaver.from_conn_string(conn_string=self.db_path) as checkpointer:
            graph = create_react_agent(
                self.chatbot_agent_llm,
                tools,
                state_modifier=self.format_for_model,
                checkpointer=checkpointer 
            )

            
            inputs = {"messages": [("user", user_input)], "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            # Initialize variable to gather AI message content
            ai_message_content = ""

            if not config:
                config = {"thread_id": "MainThread"}

            # Stream messages from the graph
            for msg, metadata in graph.stream(inputs, stream_mode="messages", config=config):
                # Check if the message is an AIMessage or AIMessageChunk
                if isinstance(msg, (AIMessageChunk, AIMessage)):
                    # Yield the content as it comes
                    if msg.content:
                        yield msg.content
                        ai_message_content += msg.content  # Accumulate if needed
                # Optionally handle other message types
                elif isinstance(msg, ToolMessage):
                    # Skip ToolMessages or process them as needed
                    pass
                else:
                    # Handle other message types if necessary
                    pass
    
   
    

