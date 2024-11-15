import subprocess
import logging
from langchain_core.tools import tool
from app.configs.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the output. This tool provides a REPL-like environment for running Python code.

    Args:
        code (str): The Python code to execute. NEVER USE MARKDOWN CODE BLOCKS. Use plain code ONLY.

    Returns:
        str: The output of the executed code or an error message if execution fails.
    
    Raises:
        subprocess.TimeoutExpired: If code execution exceeds timeout limit
        Exception: For any other execution errors
    """
    logger.info(f"Attempting to execute Python code: {code[:100]}...")
    
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info("Python code execution successful")
            return result.stdout
        else:
            logger.error(f"Python code execution failed: {result.stderr}")
            return f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        logger.error("Python code execution timed out")
        return "Error: Code execution timed out"
    except Exception as e:
        logger.error(f"Unexpected error during Python code execution: {str(e)}")
        return f"Error: {str(e)}"

@tool
def execute_cmd(command: str) -> str:
    """
    Execute a CMD command and return the output.

    Args:
        command (str): The CMD command to execute. Use plain text ONLY.

    Returns:
        str: The output of the executed command or an error message if execution fails.
    
    Raises:
        subprocess.SubprocessError: If the command execution fails
        Exception: For any other execution errors
    """
    logger.info(f"Attempting to execute CMD command: {command}")
    
    try:
        result = subprocess.run(
            ["cmd", "/c", command],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("CMD command execution successful")
            return result.stdout
        else:
            error_msg = f"Error executing command '{command}': {result.stderr}"
            logger.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error executing command '{command}': {str(e)}"
        logger.error(error_msg)
        return error_msg

