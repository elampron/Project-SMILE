import logging
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, StateGraph
from app.configs.settings import settings
from app.tools.public_tools import web_search_tool

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Smile:
    def __init__(self):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.info("Smile class logger initialized")
        self.chatbot_agent_llm = self.llm_factory("chatbot_agent")
        self.embeddings_client = self.llm_factory("embeddings")

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

        tools = [web_search_tool]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", llm_config["prompt_template"]),
            ("placeholder", "{messages}"),
        ])

        graph = create_react_agent(
            self.chatbot_agent_llm,
            tools,
            state_modifier=self.format_for_model
        )

        self.logger.debug("Chatbot agent prompt", extra={"prompt": self.prompt})

        inputs = {"messages": [("user", user_input)]}

        # Initialize variable to gather AI message content
        ai_message_content = ""

        # Stream messages from the graph
        for msg, metadata in graph.stream(inputs, stream_mode="messages"):
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


    

