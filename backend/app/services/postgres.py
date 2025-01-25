"""
PostgreSQL service module for managing database connections and operations.
"""

from typing import Any, Dict, Optional, Tuple
import logging
from psycopg import Connection
from app.utils.logger import logger
from dataclasses import dataclass
from langgraph.checkpoint.postgres import PostgresSaver as LangGraphPostgresSaver
from langchain_core.runnables import RunnableConfig

@dataclass
class CheckpointTuple:
    checkpoint: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]

class PostgresSaver(LangGraphPostgresSaver):
    """
    Class for managing PostgreSQL connections and operations.
    Inherits from langgraph.checkpoint.postgres.PostgresSaver to maintain compatibility.
    """
    
    def __init__(self, conn: Connection):
        """
        Initialize PostgresSaver with a connection.
        
        Args:
            conn: PostgreSQL connection object
        """
        super().__init__(conn)
        self.logger = logger
        
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Asynchronously get a checkpoint tuple.
        
        Args:
            config: Configuration for the checkpoint
            
        Returns:
            Optional[CheckpointTuple]: The checkpoint tuple if found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None
                
            state = self.load(thread_id)
            if not state:
                return None
                
            return CheckpointTuple(
                checkpoint=state,
                config=config,
                metadata={"thread_id": thread_id}
            )
        except Exception as e:
            self.logger.error(f"Error getting checkpoint tuple: {str(e)}")
            return None
            
    async def asave_tuple(self, checkpoint_tuple: CheckpointTuple) -> None:
        """
        Asynchronously save a checkpoint tuple.
        
        Args:
            checkpoint_tuple: The tuple to save
        """
        try:
            self.save(checkpoint_tuple.checkpoint)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint tuple: {str(e)}")
            raise
            
    async def setup(self):
        """
        Set up necessary tables in the database.
        
        Raises:
            Exception: If table creation fails
        """
        try:
            with self.conn.cursor() as cur:
                # Create tables if they don't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attachments (
                        id SERIAL PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS summaries (
                        id SERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
            self.conn.commit()
            self.logger.info("Successfully created PostgreSQL tables")
            
        except Exception as e:
            self.logger.error(f"Error setting up PostgreSQL tables: {str(e)}")
            raise
            
    def save(self, state: Dict[str, Any]):
        """
        Save state to the database.
        
        Args:
            state: State dictionary to save
            
        Raises:
            Exception: If save operation fails
        """
        try:
            with self.conn.cursor() as cur:
                # Save messages
                if "messages" in state:
                    for msg in state["messages"]:
                        cur.execute("""
                            INSERT INTO conversations (thread_id, message_id, role, content, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            state.get("thread_id"),
                            msg.get("id"),
                            msg.get("role"),
                            msg.get("content"),
                            msg.get("metadata", {})
                        ))
                        
                # Save attachments
                if "attachments" in state:
                    for att in state["attachments"]:
                        cur.execute("""
                            INSERT INTO attachments (message_id, file_name, content, mime_type, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            att.get("message_id"),
                            att.get("file_name"),
                            att.get("content"),
                            att.get("mime_type"),
                            att.get("metadata", {})
                        ))
                        
            self.conn.commit()
            self.logger.info("Successfully saved state to PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"Error saving state to PostgreSQL: {str(e)}")
            raise
            
    def load(self, thread_id: str) -> Dict[str, Any]:
        """
        Load state from the database.
        
        Args:
            thread_id: Thread ID to load state for
            
        Returns:
            Dict[str, Any]: Loaded state
            
        Raises:
            Exception: If load operation fails
        """
        try:
            state = {"thread_id": thread_id, "messages": [], "attachments": []}
            
            with self.conn.cursor() as cur:
                # Load messages
                cur.execute("""
                    SELECT message_id, role, content, metadata
                    FROM conversations
                    WHERE thread_id = %s
                    ORDER BY timestamp ASC
                """, (thread_id,))
                
                for msg_id, role, content, metadata in cur.fetchall():
                    state["messages"].append({
                        "id": msg_id,
                        "role": role,
                        "content": content,
                        "metadata": metadata
                    })
                    
                # Load attachments
                cur.execute("""
                    SELECT message_id, file_name, content, mime_type, metadata
                    FROM attachments
                    WHERE message_id IN (
                        SELECT message_id FROM conversations WHERE thread_id = %s
                    )
                    ORDER BY timestamp ASC
                """, (thread_id,))
                
                for msg_id, file_name, content, mime_type, metadata in cur.fetchall():
                    state["attachments"].append({
                        "message_id": msg_id,
                        "file_name": file_name,
                        "content": content,
                        "mime_type": mime_type,
                        "metadata": metadata
                    })
                    
            self.logger.info(f"Successfully loaded state for thread {thread_id}")
            return state
            
        except Exception as e:
            self.logger.error(f"Error loading state from PostgreSQL: {str(e)}")
            raise 