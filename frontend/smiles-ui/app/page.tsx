'use client'

import { Bell, Mic } from 'lucide-react'
import { useState, useEffect, useRef } from 'react'

export default function ChatInterface() {
  const [isRippling, setIsRippling] = useState(false)
  const synth = useRef<SpeechSynthesis | null>(null)

  useEffect(() => {
    synth.current = window.speechSynthesis
  }, [])

  const speak = (text: string) => {
    if (synth.current) {
      setIsRippling(true)
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.onend = () => setIsRippling(false)
      synth.current.speak(utterance)
    }
  }

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
      <main className="flex-grow flex">
        <div className="flex-grow border border-green-500 m-4 p-4 relative">
          {/* Chat area content */}
          <div className="absolute bottom-4 left-4 right-4 flex items-center">
            <input
              type="text"
              className="flex-grow bg-gray-800 text-green-500 p-2 outline-none"
              placeholder="Type your message..."
            />
            <button 
              className="ml-2 p-2 border border-green-500"
              onClick={() => speak("Hello, I am SMILE. How can I assist you today?")}
            >
              <Mic className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* HAL-like icon with ripple effect */}
        <div className="flex items-center justify-center p-4">
          <div className={`w-32 h-32 rounded-full bg-gradient-to-br from-red-700 to-red-500 flex items-center justify-center relative overflow-hidden ${isRippling ? 'animate-pulse' : ''}`}>
            <div className="w-24 h-24 rounded-full bg-red-600 flex items-center justify-center">
              <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
            </div>
            {isRippling && (
              <>
                <div className="absolute inset-0 bg-red-400 opacity-50 animate-ripple"></div>
                <div className="absolute inset-0 bg-red-400 opacity-50 animate-ripple animation-delay-500"></div>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}