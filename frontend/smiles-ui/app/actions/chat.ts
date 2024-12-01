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
      thread_id: threadId
    }))
  } catch (error) {
    console.error('Error fetching messages:', error)
    throw new Error('Failed to fetch messages. Please try again later.')
  }
}

/**
 * Sends a message to the chat backend and handles streaming response
 */
export async function sendMessage(message: string, threadId: string): Promise<ChatResponse> {
  if (!message?.trim()) {
    return { success: false, error: 'Message cannot be empty' }
  }

  try {
    const response = await fetch(`${SMILES_API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message.trim(),
        thread_id: threadId
      }),
    })

    if (!response.ok) {
      return { 
        success: false, 
        error: `Request failed with status: ${response.status}` 
      }
    }

    if (!response.body) {
      return { 
        success: false, 
        error: 'No response from server' 
      }
    }

    // Create a new stream from the response body
    const stream = response.body

    revalidatePath('/')
    return { 
      success: true, 
      stream 
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return { 
      success: false, 
      error: errorMessage 
    }
  }
} 