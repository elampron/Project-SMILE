import logging
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, StateGraph
from app.configs.settings import settings
from app.tools.public_tools import web_search_tool, file_tools
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from app.tools.custom_tools import execute_python, execute_cmd  # Importing both custom tools


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Smile:
    def __init__(self):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.info("Smile class logger initialized")
        self.chatbot_agent_llm = self.llm_factory("chatbot_agent")
        self.embeddings_client = self.llm_factory("embeddings")
        self.db_path = "smile.db"
        # Create PostgreSQL connection string from config
        postgres_conn = settings.app_config["postgres_config"]["conn"]
        if not postgres_conn:
            self.logger.error("PostgreSQL connection string not found in config")
            postgres_conn = "postgresql://postgres:postgres@localhost:5432/checkpoints"
        

    def format_for_model(self, state: AgentState):
        return self.prompt.invoke({"messages": state["messages"]})

    def stream(self, user_input: str):
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
        ])

        # Create the checkpoints table before using PostgresSaver
        # create_checkpoints_table(self.settings.app_config["postgres_config"]["conn"])
        

        # with PostgresSaver.from_conn_string(conn_string=self.settings.app_config["postgres_config"]["conn"]) as checkpointer:
        with SqliteSaver.from_conn_string(conn_string=self.db_path) as checkpointer:
            graph = create_react_agent(
                self.chatbot_agent_llm,
                tools,
                state_modifier=self.format_for_model,
                checkpointer=checkpointer
            )

            self.logger.debug("Chatbot agent prompt", extra={"prompt": self.prompt})

            inputs = {"messages": [("user", user_input)]}

            # Initialize variable to gather AI message content
            ai_message_content = ""

            # Stream messages from the graph
            config= {"thread_id": "123"}
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

    def llm_factory(self, llm_name: str):
        llm_config = self.settings.llm_config.get(llm_name)
        if llm_config:
            params = {key: value for key, value in llm_config.get("params", {}).items()}

            provider = llm_config.get("provider")
            if provider == "openai":
                llm = ChatOpenAI(api_key=self.settings.OPENAI_API_KEY, **params)
            elif provider == "anthropic":
                llm = ChatAnthropic(api_key=self.settings.ANTHROPIC_API_KEY, **params)
            else:
                raise ValueError(f"Provider {provider} not supported")

            return llm


# def create_checkpoints_table(conn_string: str):
#     """
#     Creates the checkpoints table if it doesn't exist.
    
#     Args:
#         conn_string (str): PostgreSQL connection string
#     """
#     create_table_sql = """
#     CREATE TABLE IF NOT EXISTS checkpoints (
#         thread_id TEXT,
#         checkpoint_ns TEXT,
#         checkpoint_data BYTEA,
#         PRIMARY KEY (thread_id, checkpoint_ns)
#     );
#     """
    
#     try:
#         with psycopg.connect(conn_string) as conn:
#             with conn.cursor() as cur:
#                 cur.execute(create_table_sql)
#             conn.commit()
#         logging.info("Checkpoints table created successfully")
#     except Exception as e:
#         logging.error(f"Error creating checkpoints table: {str(e)}")
#         raise


    

