import { Suspense } from 'react'
import ChatInput from './ChatInput'
import { MessageList } from './MessageList'
import { getMessages } from '../actions/chat'

// Types
export interface ChatMessage {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  thread_id?: string
}

// Server Component
export default async function ChatInterface() {
  // Fetch initial messages on the server
  const messages = await getMessages("Testing-02", 50)

  return (
    <div className="flex-grow flex flex-col h-full">
      <div className="flex-grow border border-green-500 m-4 p-4 relative overflow-hidden">
        <Suspense fallback={<div>Loading messages...</div>}>
          <MessageList initialMessages={messages} />
        </Suspense>
        <ChatInput threadId="Testing-02" />
      </div>
    </div>
  )
} 