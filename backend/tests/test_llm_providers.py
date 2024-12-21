import pytest
from typing import List, Dict, Any
import logging
from app.utils.llm import llm_factory
from app.configs.settings import Settings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complex test messages that simulate real conversation
COMPLEX_TEST_MESSAGES = [
    SystemMessage(content="You are a helpful AI assistant that can use tools."),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I help?"),
    ToolMessage(
        content="Tool execution result",
        tool_call_id="test_call_1",
        name="test_tool",
        additional_kwargs={"status": "success"}
    ),
    HumanMessage(content="Can you help me with something?"),
    AIMessage(
        content="Of course!",
        additional_kwargs={
            "tool_calls": [],
            "response_metadata": {
                "finish_reason": "stop",
                "model_name": "test-model"
            }
        }
    )
]

# Sample tool for testing
def calculator(operation: str) -> str:
    """A simple calculator tool for testing purposes."""
    try:
        return str(eval(operation))
    except Exception as e:
        return f"Error: {str(e)}"

# Define test tools
TEST_TOOLS = [
    Tool(
        name="calculator",
        description="Useful for performing mathematical calculations",
        func=calculator,
    )
]

def create_test_prompt():
    """Create a test prompt template similar to SMILE implementation."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that can use tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def test_provider_basic(settings: Settings, provider: str, model: str):
    """Basic test with complex message history."""
    logger.info(f"Testing basic provider: {provider} with model: {model}")
    
    settings.llm_config['test_llm'] = {
        "provider": provider,
        "params": {
            "model": model,
            "temperature": 0.0,
            "streaming": False
        }
    }
    
    try:
        llm = llm_factory(settings, "test_llm")
        logger.info("Successfully created LLM instance")
        
        # Test with simple messages first
        simple_response = llm.invoke([
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hi!")
        ])
        logger.info(f"Simple test response: {simple_response}")
        
        # Test with complex messages
        complex_response = llm.invoke(COMPLEX_TEST_MESSAGES)
        logger.info(f"Complex test response: {complex_response}")
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error in basic test for {provider}: {str(e)}")
        return False, str(e)

def test_provider_with_tools(settings: Settings, provider: str, model: str):
    """Test provider with tool binding and chain invocation."""
    logger.info(f"Testing provider with tools: {provider} with model: {model}")
    
    settings.llm_config['test_llm'] = {
        "provider": provider,
        "params": {
            "model": model,
            "temperature": 0.0,
            "streaming": False
        }
    }
    
    try:
        llm = llm_factory(settings, "test_llm")
        logger.info("Successfully created LLM instance")
        
        # Create prompt
        prompt = create_test_prompt()
        
        # Create chain with tools
        chain = prompt | llm.bind_tools(TEST_TOOLS)
        
        # Test invocation with complex history
        test_input = {
            "input": "Calculate 23 * 45",
            "chat_history": COMPLEX_TEST_MESSAGES,
            "agent_scratchpad": []
        }
        
        response = chain.invoke(test_input)
        logger.info(f"Tool test response: {response}")
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error in tool test for {provider}: {str(e)}")
        return False, str(e)

def run_provider_tests():
    """Run all test scenarios for each provider."""
    settings = Settings()
    
    test_configs = [
        ("openai", "gpt-4"),
        ("xai", "grok-beta"),
    ]
    
    results = {}
    for provider, model in test_configs:
        # Run basic test
        basic_success, basic_error = test_provider_basic(settings, provider, model)
        results[f"{provider}-{model}-basic"] = {
            "success": basic_success,
            "error": basic_error
        }
        
        # Run tool test
        tool_success, tool_error = test_provider_with_tools(settings, provider, model)
        results[f"{provider}-{model}-tools"] = {
            "success": tool_success,
            "error": tool_error
        }
        
    return results

if __name__ == "__main__":
    results = run_provider_tests()
    
    print("\nTest Results:")
    print("-" * 50)
    for config, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"\n{config}:")
        print(f"Status: {status}")
        if not result["success"]:
            print(f"Error: {result['error']}")
    print("-" * 50) 