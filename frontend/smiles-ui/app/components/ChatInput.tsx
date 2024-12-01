'use client'

import React, { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { sendMessage } from '../actions/chat'
import { useRouter } from 'next/navigation'

interface ChatInputProps {
  threadId: string;
}

export default function ChatInput({ threadId }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const router = useRouter()

  // Handle message submission
  const handleSubmit = useCallback(async () => {
    if (!message.trim()) return
    
    const currentMessage = message
    setMessage('')
    if (textareaRef.current) {
      textareaRef.current.value = ''
    }

    // Dispatch user message event
    window.dispatchEvent(new CustomEvent('newMessage', {
      detail: {
        content: currentMessage,
        role: 'user',
        timestamp: new Date(),
        thread_id: threadId
      }
    }))

    // Set AI thinking state
    window.dispatchEvent(new CustomEvent('aiThinking', { detail: true }))

    const response = await sendMessage(currentMessage, threadId)
    if (!response.success) {
      console.error('Failed to send message:', response.error)
      window.dispatchEvent(new CustomEvent('aiThinking', { detail: false }))
      return
    }

    // Handle streaming response
    if (response.stream) {
      const reader = response.stream.getReader()
      const decoder = new TextDecoder()
      
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          const chunk = decoder.decode(value)
          window.dispatchEvent(new CustomEvent('aiStream', { detail: chunk }))
        }
      } catch (error) {
        console.error('Error reading stream:', error)
      } finally {
        window.dispatchEvent(new CustomEvent('aiThinking', { detail: false }))
        router.refresh()
      }
    }
  }, [message, threadId, router])

  // Handle textarea input
  const handleInput = useCallback((e: React.FormEvent<HTMLTextAreaElement>) => {
    const value = e.currentTarget.value
    setMessage(value)
  }, [])

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm border-t border-green-900/50 p-4 pb-6">
      <div className="max-w-4xl mx-auto relative">
        <div className="flex flex-col gap-2 mb-2">
          <div className="flex items-end gap-2">
            <Textarea
              ref={textareaRef}
              placeholder="Type a message..."
              className="min-h-[60px] flex-1 bg-black/50 border-green-900/50 text-green-500 placeholder:text-green-800/50 focus:border-green-500/70 focus:ring-1 focus:ring-green-500/70 resize-none shadow-lg"
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              value={message}
            />
            <Button 
              onClick={handleSubmit}
              className="bg-black/90 border-green-900/50 hover:bg-green-900/30 hover:border-green-500/70 text-green-500 shadow-lg"
            >
              Send
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
} 