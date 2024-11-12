from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from app.configs.settings import settings




# Web search tool
web_search_tool = TavilySearchResults(name="Web_search_tool", max_results=5,description="Search the web for information")

# File tools
file_tools = [
    CopyFileTool(),
    DeleteFileTool(),
    MoveFileTool(),
    ReadFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
]



    

