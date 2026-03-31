import { Outlet } from 'react-router-dom'
import { useEffect } from 'react'
import { useWebSocket } from '../controllers/connection'
import Header from './Header'
import ToastContainer from './Toast'

export default function Layout() {
  const { connect, disconnect } = useWebSocket()

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return (
    <div className="flex flex-col h-screen bg-bg text-tx">
      <Header />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
      <ToastContainer />
    </div>
  )
}
