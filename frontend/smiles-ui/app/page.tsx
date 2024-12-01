import { Suspense } from 'react'
import ChatInterface from './components/ChatInterface'

// Server Component
export default async function Home() {
  return (
    <div className="flex flex-col h-screen bg-black text-green-500 font-mono">
      <header className="flex justify-center items-center p-4 border-b border-green-500">
        <h1 className="text-3xl tracking-wider font-bold">S.M.I.L.E.</h1>
      </header>
      
      <Suspense fallback={
        <div className="flex items-center justify-center h-full">
          <div className="text-green-500">Loading chat interface...</div>
        </div>
      }>
        <ChatInterface />
      </Suspense>
    </div>
  )
}