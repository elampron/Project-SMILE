# ./backend/src/api/main.py

# Import FastAPI class from fastapi module
from time import time
from fastapi import FastAPI

# Import CORSMiddleware to handle Cross-Origin Resource Sharing
from fastapi.middleware.cors import CORSMiddleware

# Import the routers
from app.api.routers import router
from app.api.events_routes import router as events_router

import uvicorn
import logging
import sys
from app.configs.settings import settings
import os
from dotenv import load_dotenv
import asyncio  # Import asyncio to manage event loops

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = settings.app_config["langchain_config"]["tracing_v2"]
os.environ["LANGCHAIN_ENDPOINT"] = settings.app_config["langchain_config"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"] = settings.app_config["langchain_config"]["project"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an instance of the FastAPI class
app = FastAPI(
    title="Project SMILE API",
    description="API for Project SMILE with event system integration",
    version="1.0.0"
)

# Include the routers
app.include_router(router)
app.include_router(events_router, prefix="/api/v1")

# Add middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize necessary components on startup
    """
    logger.info("Initializing application...")
    # Add any initialization code here
    logger.info("Application initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    logger.info("Shutting down application...")
    # Add any cleanup code here
    logger.info("Application shut down successfully")

if __name__ == "__main__":
    logger.info("Starting server on host: 0.0.0.0, port: 8000")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        logger.info("Server running successfully")
    except Exception as e:
        logger.error(f"Server startup failed - Error: {str(e)}", exc_info=True)
        sys.exit(1)
