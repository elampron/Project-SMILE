'use client'

import React, { useState, useRef, useCallback } from 'react'
import { Button } from './ui/button'
import { Textarea } from './ui/textarea'
import { sendMessage } from '../actions/chat'
import { useRouter } from 'next/navigation'
import { Paperclip, X } from 'lucide-react'

interface ChatInputProps {
  threadId: string;
}

export default function ChatInput({ threadId }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  // Handle file selection
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => [...prev, ...newFiles])
    }
  }, [])

  // Remove a file from the selection
  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }, [])

  // Handle message submission
  const handleSubmit = useCallback(async () => {
    if (!message.trim() && files.length === 0) return
    
    const currentMessage = message
    const currentFiles = files
    setMessage('')
    setFiles([])
    if (textareaRef.current) {
      textareaRef.current.value = ''
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }

    // Dispatch user message event
    window.dispatchEvent(new CustomEvent('newMessage', {
      detail: {
        content: currentMessage,
        role: 'user',
        timestamp: new Date(),
        thread_id: threadId,
        files: currentFiles.map(f => f.name)
      }
    }))

    // Set AI thinking state
    window.dispatchEvent(new CustomEvent('aiThinking', { detail: true }))

    const response = await sendMessage(currentMessage, threadId, currentFiles)
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
        // Add a small delay before refreshing to ensure database is updated
        setTimeout(() => {
          router.refresh()
        }, 1000)
      }
    }
  }, [message, files, threadId, router])

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
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {files.map((file, index) => (
              <div 
                key={index}
                className="flex items-center gap-2 bg-green-900/20 text-green-500 px-2 py-1 rounded-md text-sm"
              >
                <span className="truncate max-w-[200px]">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-green-500 hover:text-green-400"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="flex flex-col gap-2">
          <div className="flex items-end gap-2">
            <Textarea
              ref={textareaRef}
              placeholder="Type a message..."
              className="min-h-[60px] flex-1 bg-black/50 border-green-900/50 text-green-500 placeholder:text-green-800/50 focus:border-green-500/70 focus:ring-1 focus:ring-green-500/70 resize-none shadow-lg"
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              value={message}
            />
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              className="hidden"
              multiple
            />
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              className="bg-black/90 border-green-900/50 hover:bg-green-900/30 hover:border-green-500/70 text-green-500 shadow-lg"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
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