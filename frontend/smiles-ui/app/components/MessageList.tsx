'use client'

import { useEffect, useRef, useState } from 'react'
import { ChatMessage } from './ChatInterface'

interface MessageListProps {
  initialMessages: ChatMessage[]
}

export function MessageList({ initialMessages }: MessageListProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [isAiResponding, setIsAiResponding] = useState(false)
  const [currentAiMessage, setCurrentAiMessage] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAiMessage])

  // Listen for custom events
  useEffect(() => {
    const handleNewMessage = (e: CustomEvent<ChatMessage>) => {
      setMessages(prev => [...prev, e.detail])
    }

    const handleAiThinking = (e: CustomEvent<boolean>) => {
      setIsAiResponding(e.detail)
      if (e.detail) {
        setCurrentAiMessage('')
      }
    }

    const handleAiStream = (e: CustomEvent<string>) => {
      setCurrentAiMessage(prev => prev + e.detail)
    }

    window.addEventListener('newMessage' as any, handleNewMessage as any)
    window.addEventListener('aiThinking' as any, handleAiThinking as any)
    window.addEventListener('aiStream' as any, handleAiStream as any)

    return () => {
      window.removeEventListener('newMessage' as any, handleNewMessage as any)
      window.removeEventListener('aiThinking' as any, handleAiThinking as any)
      window.removeEventListener('aiStream' as any, handleAiStream as any)
    }
  }, [])

  if (!messages.length && !isAiResponding) {
    return (
      <div className="absolute top-4 left-4 right-4 bottom-20 flex items-center justify-center">
        <div className="text-green-500 opacity-50">
          No messages yet. Start a conversation!
        </div>
      </div>
    )
  }

  return (
    <div className="absolute top-4 left-4 right-4 bottom-20 overflow-y-auto">
      <div className="flex flex-col">
        {messages.map((message, index) => (
          <div
            key={`${message.timestamp.getTime()}-${index}`}
            className={`mb-4 p-2 rounded ${
              message.role === 'user'
                ? 'bg-green-900/20 ml-auto max-w-[80%]'
                : 'bg-green-900/10 mr-auto max-w-[80%]'
            }`}
          >
            <div className="text-sm opacity-50 mb-1">
              {message.role === 'user' ? 'You' : 'S.M.I.L.E.'}
              <span className="ml-2 text-xs">
                {new Date(message.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="whitespace-pre-wrap">{message.content}</div>
          </div>
        ))}
        
        {/* AI Thinking/Streaming Message */}
        {(isAiResponding || currentAiMessage) && (
          <div className="mb-4 p-2 rounded bg-green-900/10 mr-auto max-w-[80%]">
            <div className="text-sm opacity-50 mb-1">
              S.M.I.L.E.
              <span className="ml-2 text-xs">
                {new Date().toLocaleTimeString()}
              </span>
            </div>
            <div className="whitespace-pre-wrap">
              {currentAiMessage || (
                <span className="flex items-center">
                  <span className="animate-pulse">Thinking</span>
                  <span className="animate-[bounce_1s_infinite] ml-1">.</span>
                  <span className="animate-[bounce_1s_infinite] animation-delay-200 ml-1">.</span>
                  <span className="animate-[bounce_1s_infinite] animation-delay-400 ml-1">.</span>
                </span>
              )}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  )
} 