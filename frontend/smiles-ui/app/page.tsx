'use client'

import { Bell, Mic, ChevronDown, ChevronRight, Save } from 'lucide-react'
import { useState, useEffect, useRef } from 'react'

// Define interfaces
interface ChatMessage {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  thread_id?: string
}

interface Settings {
  app_config: any;
  llm_config: any;
}

// AdminPanel Component
function AdminPanel() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [expandedSections, setExpandedSections] = useState<{ [key: string]: boolean }>({});
  const [editMode, setEditMode] = useState<{ path: string; value: string } | null>(null);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await fetch('http://backend:8002/settings');
      const data = await response.json();
      if (data.status === 'success') {
        setSettings(data.data);
      }
    } catch (error) {
      console.error('Error fetching settings:', error);
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleEdit = async (path: string, newValue: string) => {
    try {
      const pathParts = path.split('.');
      // Remove 'root' from the path if it exists
      if (pathParts[0] === 'root') {
        pathParts.shift();
      }
      
      // Determine which config type we're updating
      const configType = pathParts[0];
      
      // Get the current settings for this config type
      const currentSettings = settings?.[configType as keyof Settings];
      if (!currentSettings) {
        throw new Error(`Invalid config type: ${configType}`);
      }

      // Create a deep copy of the current settings
      const updateValue = JSON.parse(JSON.stringify(currentSettings));
      
      // Navigate through the path to set the new value
      let current = updateValue;
      for (let i = 1; i < pathParts.length - 1; i++) {
        if (!(pathParts[i] in current)) {
          current[pathParts[i]] = {};
        }
        current = current[pathParts[i]];
      }
      
      // Set the new value
      const lastKey = pathParts[pathParts.length - 1];
      
      // Try to parse the value if it's a boolean or number
      let parsedValue: string | boolean | number = newValue;
      if (newValue.toLowerCase() === 'true' || newValue.toLowerCase() === 'false') {
        parsedValue = newValue.toLowerCase() === 'true';
      } else if (!isNaN(Number(newValue)) && newValue.trim() !== '') {
        parsedValue = Number(newValue);
      }
      
      current[lastKey] = parsedValue;

      console.log('Updating settings:', {
        config_type: configType,
        settings_data: updateValue
      });

      const response = await fetch('http://backend:8002/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config_type: configType,
          settings_data: updateValue
        }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success') {
          fetchSettings(); // Refresh settings
          setEditMode(null);
        } else {
          throw new Error(result.message || 'Failed to update settings');
        }
      } else {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error updating setting:', error);
      // You might want to show this error to the user
      alert(`Failed to update setting: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const renderValue = (value: any, path: string, indent: number = 0) => {
    if (value === null) return <span className="text-gray-500">null</span>;
    
    if (typeof value === 'object') {
      return (
        <div style={{ marginLeft: `${indent}px` }}>
          {Object.entries(value).map(([key, val]) => (
            <div key={key} className="my-1">
              <div 
                className="flex items-center cursor-pointer hover:bg-green-900/20"
                onClick={() => toggleSection(`${path}.${key}`)}
              >
                {typeof val === 'object' && val !== null ? (
                  expandedSections[`${path}.${key}`] ? <ChevronDown size={16} /> : <ChevronRight size={16} />
                ) : null}
                <span className="font-semibold">{key}:</span>
              </div>
              {expandedSections[`${path}.${key}`] && (
                renderValue(val, `${path}.${key}`, indent + 20)
              )}
            </div>
          ))}
        </div>
      );
    }

    return (
      <div className="flex items-center gap-2">
        {editMode?.path === path ? (
          <>
            <input
              type="text"
              value={editMode.value}
              onChange={(e) => setEditMode({ ...editMode, value: e.target.value })}
              className="bg-gray-800 text-green-500 px-2 py-1 rounded"
            />
            <button
              onClick={() => handleEdit(path, editMode.value)}
              className="p-1 hover:bg-green-900/20 rounded"
            >
              <Save size={16} />
            </button>
          </>
        ) : (
          <span
            className="cursor-pointer hover:underline"
            onClick={() => setEditMode({ path, value: String(value) })}
          >
            {String(value)}
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="p-4 h-full overflow-y-auto">
      <div className="border border-green-500 p-4 rounded">
        <h2 className="text-xl mb-4">Settings</h2>
        {settings ? (
          renderValue(settings, 'root')
        ) : (
          <div>Loading settings...</div>
        )}
      </div>
    </div>
  );
}

export default function ChatInterface() {
  const [activeTab, setActiveTab] = useState('Chat');
  const [isRippling, setIsRippling] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const synth = useRef<SpeechSynthesis | null>(null)
  const [threadId, setThreadId] = useState<string>("Testing-01")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    synth.current = window.speechSynthesis
  }, [])

  // Fetch conversation history on page load
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`http://backend:8002/history?thread_id=Testing-01&num_messages=50`)
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)

        const data = await response.json()
        if (data.status === 'success') {
          // Map the data to ChatMessage format with correct role mapping
          const historyMessages: ChatMessage[] = data.data.map((msg: any) => ({
            content: msg.content,
            // Map 'human' to 'user' and 'assistant' to 'assistant'
            role: msg.role === 'human' ? 'user' : 'assistant',
            timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
            thread_id: threadId
          }))
          console.log('Loaded history messages:', historyMessages); // Debug log
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
    if (!inputMessage.trim()) {
      console.log('Empty message, returning early');
      return;
    }

    console.log('1. Starting sendMessage function');
    console.log('Input message:', inputMessage);
    console.log('Thread ID:', threadId);

    const userMessage: ChatMessage = {
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
      thread_id: threadId
    }
    
    // Store message before clearing input
    const messageToSend = inputMessage;
    
    console.log('2. Adding user message to UI');
    setMessages(prev => [...prev, userMessage])
    
    console.log('3. Clearing input');
    setInputMessage('')

    try {
      console.log('4. Attempting API call to:', 'http://backend:8002/chat');
      const response = await fetch('http://backend:8002/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: messageToSend,
          thread_id: threadId 
        }),
      })

      console.log('5. API response received:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}, statusText: ${response.statusText}`);
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      if (!reader) throw new Error('Response body is null')

      // Create placeholder message for streaming
      const assistantMessage: ChatMessage = {
        content: '',
        role: 'assistant',
        timestamp: new Date(),
        thread_id: threadId
      }
      setMessages(prev => [...prev, assistantMessage])

      let streamedContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Decode and accumulate the chunks
        const chunk = new TextDecoder().decode(value)
        console.log('Received chunk:', chunk)
        
        // Accumulate the content instead of replacing it
        streamedContent += chunk
        
        // Update the message with accumulated content
        setMessages(prev => prev.map((msg, index) => 
          index === prev.length - 1 
            ? { ...msg, content: streamedContent }
            : msg
        ))
      }

      console.log('Streaming completed successfully')
    } catch (error) {
      console.error('Error sending message:', error)
    }
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

  const renderContent = () => {
    switch (activeTab) {
      case 'Chat':
        return (
          <div className="flex-grow border border-green-500 m-4 p-4 relative overflow-hidden">
            {/* Messages area */}
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

            {/* Input area */}
            <div className="absolute bottom-4 left-4 right-4 flex items-center">
              <textarea
                className="flex-grow bg-gray-800 text-green-500 p-2 outline-none resize-none"
                placeholder="Type your message... (Shift+Enter for new line)"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                rows={1}
                style={{ minHeight: '40px', maxHeight: '120px' }}
              />
              <button 
                className="ml-2 p-2 border border-green-500"
                onClick={() => speak("Hello, I am SMILE. How can I assist you today?")}
              >
                <Mic className="w-6 h-6" />
              </button>
            </div>
          </div>
        );
      case 'Admin':
        return <AdminPanel />;
      default:
        return <div className="p-4">Coming soon...</div>;
    }
  };

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
              tab === activeTab ? 'bg-green-500 text-black' : ''
            }`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>
      <main className="flex-grow flex overflow-hidden">
        {renderContent()}
      </main>
    </div>
  );
}