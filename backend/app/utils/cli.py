import logging
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from app.agents.smile import Smile
from app.configs.settings import settings
import asyncio

# Load the environment variables from the .env file
load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = settings.app_config["langchain_config"]["endpoint"]
# os.environ["LANGCHAIN_PROJECT"] = settings.app_config["langchain_config"]["project"]

# Logging setup
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLI")

console = Console()

logger.info("Starting CLI")
logger.debug("Settings: %s", settings)
logger.debug("LLM Config: %s", settings.llm_config)

welcome_text = """
# Welcome to Project-SMILE CLI

Smiles is your AI companion, ready to chat, help, and explore ideas with you.

Available commands:
- 1. Chat with Smiles (just type your message)
- 2. Show Menu
- 0. Exit

To use a command, type its number or just start chatting!
"""

# clear the console
console.clear()
# Print welcome text
console.print(Markdown(welcome_text))

smile = Smile()

async def main():
    # Main loop
    while True:
        user_input = Prompt.ask("Enter a command:")
        if user_input.strip() == "0":
            console.print("Goodbye!")
            break
        elif user_input.strip() == "2":
            # Clear the screen before showing menu
            console.clear()
            console.print(Markdown(welcome_text))
            continue
        else:
            # Notify the user that the AI is responding
            console.print("Smiles is typing...\n")

            # Stream the response as it comes
            response_content = ""
            async for chunk in smile.stream(user_input, config={"thread_id": "MainThread"}):
                response_content += chunk
                console.print(chunk, end="")
            console.print("\n")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

