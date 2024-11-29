import logging
from fastapi import FastAPI
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

@app.get("/status")
async def get_status():
    """
    Forwards the raw metrics from the backend service.
    
    Returns:
        Response: Raw metrics text directly from the backend service
        
    Logs:
        - INFO: When starting to fetch metrics
        - DEBUG: When successfully fetched metrics
        - ERROR: When failing to fetch metrics
    """
    logger.info("Fetching metrics from backend service")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://backend:8002/metrics")
            if response.status_code == 200:
                logger.debug("Successfully fetched metrics from backend")
                return response.text
            else:
                logger.error(f"Failed to fetch metrics. Status code: {response.status_code}")
                return {"error": "Failed to fetch metrics", "status_code": response.status_code}
        except Exception as e:
            logger.error(f"Exception while fetching metrics: {str(e)}")
            return {"error": f"Failed to fetch metrics: {str(e)}"}