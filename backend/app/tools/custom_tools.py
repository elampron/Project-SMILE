import subprocess
import logging
from langchain_core.tools import BaseTool, StructuredTool, tool
from typing import Optional
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Define schema for Python execution tool
class PythonExecuteSchema(BaseModel):
    code: str = Field(description="Python code to execute")

# Define schema for Command execution tool
class CommandExecuteSchema(BaseModel):
    command: str = Field(description="Command to execute")

@tool
def execute_python(code: str) -> str:
    """Execute Python code and return the result.
    
    Use this tool when you need to execute Python code. The code will be evaluated
    and the result returned as a string.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Result of code execution or error message
    """
    logger.info(f"Executing Python code: {code}")
    try:
        # Add safety measures and execution logic here
        result = eval(code)  # Be careful with eval - you might want to use a safer execution method
        logger.info(f"Python execution successful: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error executing Python code: {str(e)}")
        return f"Error executing Python code: {str(e)}"

@tool
def execute_cmd(command: str) -> str:
    """Execute a system command and return the output.
    
    Use this tool when you need to run system commands. The command will be executed
    in a shell and the output returned as a string.
    
    Args:
        command (str): Command to execute
        
    Returns:
        str: Command execution output or error message
    """
    logger.info(f"Executing command: {command}")
    try:
        # Add safety measures and execution logic here
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout if result.stdout else result.stderr
        logger.info(f"Command execution successful: {output}")
        return output
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return f"Error executing command: {str(e)}"

