import sqlite3
import json
import msgpack
import logging
from app.configs.settings import settings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_checkpoints(db_path):
    """
    Read and deserialize checkpoint data from the SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query the checkpoints table
    cursor.execute("SELECT thread_id, checkpoint_id, checkpoint FROM checkpoints where thread_id = '123'")
    
    # Fetch all rows from the checkpoints table
    rows = cursor.fetchall()
    
    for row in rows:
        thread_id, checkpoint_id, checkpoint_blob = row
        
        try:
            # Deserialize the BLOB data using MessagePack
            checkpoint_data = msgpack.unpackb(checkpoint_blob)
            
            # Convert the deserialized data to a JSON-formatted string for readability
            checkpoint_json = json.dumps(checkpoint_data, indent=2, default=str)
            
            # Log the checkpoint data
            logger.debug(f"Thread ID: {thread_id}")
            logger.debug(f"Checkpoint ID: {checkpoint_id}")
            logger.debug(f"Checkpoint Data: {checkpoint_json}")
            
            # # Print the checkpoint data
            # print(f"Thread ID: {thread_id}")
            # print(f"Checkpoint ID: {checkpoint_id}")
            # print(f"Checkpoint Data: {checkpoint_json}")
        
        except Exception as e:
            # Log any errors that occur during deserialization
            logger.error(f"Error deserializing checkpoint data: {e}")
            continue  # Skip to the next row if deserialization fails
    
    # Close the database connection
    conn.close()


if __name__ == "__main__":
    # Read and print the checkpoints from the database
    read_checkpoints(settings.app_config["checkpoint_path"])