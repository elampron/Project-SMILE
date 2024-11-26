'use client'

import { Bell, Mic } from 'lucide-react'
import { useState, useEffect, useRef } from 'react'

// Define interface for chat messages
interface ChatMessage {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  thread_id?: string
}

export default function ChatInterface() {
  const [isRippling, setIsRippling] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const synth = useRef<SpeechSynthesis | null>(null)
  const [threadId, setThreadId] = useState<string>("MainThread")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    synth.current = window.speechSynthesis
  }, [])

  // Fetch conversation history on page load
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`http://elmonster.local:8000/history?thread_id=${threadId}&num_messages=50`)
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)

        const data = await response.json()
        if (data.status === 'success') {
          // Map the data to ChatMessage format
          const historyMessages: ChatMessage[] = data.data.map((msg: any) => ({
            content: msg.content,
            role: msg.role === 'human' ? 'user' : 'assistant',
            timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
            thread_id: threadId
          }))
          setMessages(historyMessages)
        } else {
          console.error('Unexpected response format:', data)
        }
      } catch (error) {
        console.error('Error fetching conversation history:', error)
      }
    }

    fetchHistory()
  }, [threadId])

  // Function to handle sending messages
  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    console.log('Attempting to send message:', inputMessage, 'Thread:', threadId)

    const userMessage: ChatMessage = {
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
      thread_id: threadId
    }
    setMessages(prev => [...prev, userMessage])

    try {
      const response = await fetch('http://elmonster.local:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: inputMessage,
          thread_id: threadId 
        }),
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)

      // Handle streaming response
      const reader = response.body?.getReader()
      if (!reader) throw new Error('Response body is null')

      let assistantResponse = ''
      
      // Create placeholder message for streaming
      const assistantMessage: ChatMessage = {
        content: '',
        role: 'assistant',
        timestamp: new Date(),
        thread_id: threadId
      }
      setMessages(prev => [...prev, assistantMessage])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Decode and accumulate the chunks
        const chunk = new TextDecoder().decode(value)
        assistantResponse += chunk

        // Update the last message with accumulated response
        setMessages(prev => prev.map((msg, index) => 
          index === prev.length - 1 
            ? { ...msg, content: assistantResponse }
            : msg
        ))
      }

      console.log('Streaming completed successfully')
    } catch (error) {
      console.error('Error sending message:', error)
    }

    setInputMessage('')
  }

  const speak = (text: string) => {
    if (synth.current) {
      setIsRippling(true)
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.onend = () => setIsRippling(false)
      synth.current.speak(utterance)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="flex flex-col h-screen bg-black text-green-500 font-mono">
      {/* Header */}
      <header className="flex justify-between items-center p-4 border-b border-green-500">
        <h1 className="text-3xl tracking-wider">S.M.I.L.E.</h1>
        <Bell className="w-6 h-6" />
      </header>

      {/* Navigation */}
      <nav className="flex justify-center space-x-4 p-2 border-b border-green-500">
        {['Chat', 'Code', 'Tasks', 'Admin'].map((tab) => (
          <button
            key={tab}
            className={`px-4 py-2 border border-green-500 ${
              tab === 'Chat' ? 'bg-green-500 text-black' : ''
            }`}
          >
            {tab}
          </button>
        ))}
      </nav>
      <main className="flex-grow flex overflow-hidden">
        {/* Chat container - add overflow-hidden to contain scrolling */}
        <div className="flex-grow border border-green-500 m-4 p-4 relative overflow-hidden">
          {/* Messages area - explicit height calculation */}
          <div className="absolute top-4 left-4 right-4 bottom-20 overflow-y-auto">
            <div className="flex flex-col">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-4 p-2 rounded ${
                    message.role === 'user' 
                      ? 'bg-green-900/20 ml-auto max-w-[80%]' 
                      : 'bg-green-900/10 mr-auto max-w-[80%]'
                  }`}
                >
                  <div className="text-sm opacity-50 mb-1">
                    {message.role === 'user' ? 'You' : 'S.M.I.L.E.'}
                  </div>
                  <div>{message.content}</div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input area - position absolute from bottom */}
          <div className="absolute bottom-4 left-4 right-4 flex items-center">
            <input
              type="text"
              className="flex-grow bg-gray-800 text-green-500 p-2 outline-none"
              placeholder="Type your message..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button 
              className="ml-2 p-2 border border-green-500"
              onClick={() => speak("Hello, I am SMILE. How can I assist you today?")}
            >
              <Mic className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* HAL icon section - fixed width */}
       
      </main>
    </div>
  )
}