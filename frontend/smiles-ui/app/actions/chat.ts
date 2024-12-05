'use server'

import { revalidatePath } from 'next/cache'
import { ChatMessage } from '../components/ChatInterface'

// Get API URL from environment variable
const SMILES_API_URL = process.env.SMILES_API_URL || 'http://backend:8000'

interface ChatResponse {
  success: boolean
  stream?: ReadableStream
  error?: string
}

/**
 * Fetches chat messages from the backend
 */
export async function getMessages(threadId: string, numMessages: number): Promise<ChatMessage[]> {
  try {
    const response = await fetch(
      `${SMILES_API_URL}/history?thread_id=${threadId}&num_messages=${numMessages}`,
      { 
        cache: 'no-store',
        headers: {
          'Content-Type': 'application/json',
        }
      }
    )
    
    if (!response.ok) {
      throw new Error(`Failed to fetch messages: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    
    if (data.status !== 'success' || !Array.isArray(data.data)) {
      throw new Error(data.message || 'Invalid response format')
    }

    return data.data.map((msg: any) => ({
      content: msg.content || '',
      role: msg.role === 'human' ? 'user' : 'assistant',
      timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
      thread_id: threadId,
      files: msg.files || []
    }))
  } catch (error) {
    console.error('Error fetching messages:', error)
    throw new Error('Failed to fetch messages. Please try again later.')
  }
}

/**
 * Sends a message to the chat backend and handles streaming response
 * @param message The message text to send
 * @param threadId The thread ID for the conversation
 * @param files Optional array of files to upload
 */
export async function sendMessage(
  message: string, 
  threadId: string,
  files?: File[]
): Promise<ChatResponse> {
  if (!message?.trim() && (!files || files.length === 0)) {
    return { success: false, error: 'Message and files cannot both be empty' }
  }

  try {
    // Use JSON endpoint if no files, form-data if files exist
    if (!files || files.length === 0) {
      const response = await fetch(`${SMILES_API_URL}/chat/json`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message.trim(),
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        return { 
          success: false, 
          error: `Request failed with status: ${response.status}. ${errorText}` 
        };
      }

      if (!response.body) {
        return { 
          success: false, 
          error: 'No response from server' 
        };
      }

      return { 
        success: true, 
        stream: response.body 
      };
    }

    // Handle file uploads with form-data
    const formData = new FormData();
    formData.append('message', message.trim());
    formData.append('thread_id', threadId);
    
    files.forEach(file => {
      formData.append('files', file, file.name);
    });

    console.log('Sending form data:', {
      message: message.trim(),
      threadId,
      files: files.map(f => f.name)
    });

    const response = await fetch(`${SMILES_API_URL}/chat`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error response:', errorText);
      return { 
        success: false, 
        error: `Request failed with status: ${response.status}. ${errorText}` 
      };
    }

    if (!response.body) {
      return { 
        success: false, 
        error: 'No response from server' 
      };
    }

    return { 
      success: true, 
      stream: response.body 
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Send message error:', error);
    return { 
      success: false, 
      error: errorMessage 
    };
  }
} 