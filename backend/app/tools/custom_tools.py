import subprocess
from langchain_core.tools import tool
from app.configs.settings import settings

@tool
def execute_python(self, code: str) -> str:
    """
    Execute Python code and return the output. This tool provides a REPL-like environment for running Python code.

    Args:
        code (str): The Python code to execute. NEVER USE MARKDOWN CODE BLOCKS. Use plain code ONLY.

    Returns:
        str: The output of the executed code or an error message if execution fails. Remember to use print statements to display output.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"