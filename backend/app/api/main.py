# ./backend/src/api/main.py

# Import FastAPI class from fastapi module
from time import time
from fastapi import FastAPI

# Import CORSMiddleware to handle Cross-Origin Resource Sharing
from fastapi.middleware.cors import CORSMiddleware

# Import the routers
from app.api.routers import router, smile
from app.api.events_routes import router as events_router
from app.services.neo4j.schema import initialize_schema
from app.services.neo4j.driver import close_driver
from app.agents.smile import Smile

import uvicorn
import logging
import sys
from app.configs.settings import settings
import os
from dotenv import load_dotenv
from app.utils.logger import logger

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = settings.app_config["langchain_config"]["tracing_v2"]
os.environ["LANGCHAIN_ENDPOINT"] = settings.app_config["langchain_config"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"] = settings.app_config["langchain_config"]["project"]

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
    Initialize necessary components on startup.
    Order is important:
    1. Initialize Neo4j schema first
    2. Then initialize Smile instance which will handle user creation
    """
    logger.info("Initializing application...")
    
    # Initialize Neo4j schema first
    try:
        initialize_schema()
        logger.info("Neo4j schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j schema: {str(e)}")
        raise
        
    # Initialize Smile instance which will handle user creation
    try:
        global smile
        smile = Smile()
        smile.initialize()
        logger.info("Smile instance initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Smile instance: {str(e)}")
        raise
        
    logger.info("Application initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    logger.info("Shutting down application...")
    
    # Clean up Smile instance
    try:
        if smile:
            smile.cleanup()
            logger.info("Smile instance cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up Smile instance: {str(e)}")
    
    # Close Neo4j driver
    try:
        close_driver()
        logger.info("Neo4j driver closed successfully")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver: {str(e)}")
        
    logger.info("Application shut down successfully")

if __name__ == "__main__":
    logger.info("Starting server on host: 0.0.0.0, port: 8000")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        logger.info("Server running successfully")
    except Exception as e:
        logger.error(f"Server startup failed - Error: {str(e)}", exc_info=True)
        sys.exit(1)
