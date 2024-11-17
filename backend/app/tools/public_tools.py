import os
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


os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

# Web search tool
web_search_tool = TavilySearchResults(
    name="Web_search_tool", 
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    search_depth="advanced",
    # include_domains = []
    # exclude_domains = []
)

# File tools
file_tools = [
    CopyFileTool(),
    DeleteFileTool(),
    MoveFileTool(),
    ReadFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
]



    

