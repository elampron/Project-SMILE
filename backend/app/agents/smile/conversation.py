"""
Conversation management module for the SMILE agent.

This module contains functions for managing conversations and message history.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from app.utils.logger import logger
from app.models.memory import ConversationSummary
from .attachments import process_attachments

async def get_conversation_history(agent: Any, thread_id: str, num_messages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history from the agent's graph state.
    
    Args:
        agent: The Smile agent instance
        thread_id: The ID of the conversation thread
        num_messages: Optional number of most recent messages to return
        
    Returns:
        List[Dict[str, Any]]: List of conversation messages
        
    Raises:
        Exception: If retrieval fails
    """
    try:
        if not agent._initialized:
            await agent.initialize()
                
        config = {"configurable": {"thread_id": thread_id}}
            
        # Get state from the graph
        state = agent.graph.get_state(config)
            
        # Get the latest state's messages
        messages = state.values.get('messages', [])
            
        # Prepare the conversation history
        conversation_history = []
        for msg in messages[-num_messages:] if num_messages else messages:
            if hasattr(msg, 'content'):
                message = {
                    "role": "human" if msg.type == "human" else "assistant",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat()),
                    "message_id": msg.id if hasattr(msg, 'id') else None
                }
                conversation_history.append(message)

        logger.info(f"Retrieved {len(conversation_history)} messages from conversation history")
        return conversation_history

    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise

async def stream_conversation(agent: Any, message: str, attachments: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
    """
    Stream conversation using the agent's graph.
    
    Args:
        agent: The Smile agent instance
        message: The input message
        attachments: Optional list of attachment dictionaries
        
    Yields:
        str: Message chunks
        
    Raises:
        Exception: If streaming fails
    """
    try:
        # Initialize messages list
        messages = [{"role": "user", "content": message}]
            
        # Add attachment content if present
        if attachments:
            processed_attachments = await process_attachments(agent, attachments)
            if processed_attachments:
                attachment_context = "\n\nAttached files:\n"
                for attachment in processed_attachments:
                    attachment_context += (
                        f"\nFile: {attachment.file_name}\n"
                        f"Type: {attachment.mime_type.value}\n"
                        f"Content:\n{attachment.content}\n"
                        f"---\n"
                    )
                messages.append({"role": "system", "content": attachment_context})
                logger.info(f"Added {len(processed_attachments)} attachments to context")

        logger.info(f"Starting stream with {len(attachments or [])} attachments")
        
        # Get the response from the model
        async for chunk in agent.chatbot_agent_llm.astream(messages):
            yield chunk.content
            
    except Exception as e:
        logger.error(f"Error streaming conversation: {str(e)}", exc_info=True)
        raise

def summarize_conversation(agent: Any, thread_id: str) -> Dict[str, Any]:
    """
    Create a summary of the conversation.
    
    Args:
        agent: The Smile agent instance
        thread_id: The conversation thread ID
        
    Returns:
        ConversationSummary: Summary of the conversation
        
    Raises:
        Exception: If summarization fails
    """
    try:
        # Get conversation history
        history = get_conversation_history(agent, thread_id)
        if not history:
            return ConversationSummary(
                thread_id=thread_id,
                summary="No conversation history found.",
                last_updated=datetime.utcnow()
            )
            
        # Extract messages for summarization
        messages = [msg['content'] for msg in history if 'content' in msg]
        
        # Create summary prompt
        summary_prompt = (
            "Please provide a brief summary of the following conversation:\n\n" +
            "\n".join(messages)
        )
        
        # Get summary from model
        summary = agent.chatbot_agent_llm([{
            'role': 'user',
            'content': summary_prompt
        }])
        
        # Create and return summary object
        return ConversationSummary(
            thread_id=thread_id,
            summary=summary.content if hasattr(summary, 'content') else str(summary),
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {str(e)}")
        raise 