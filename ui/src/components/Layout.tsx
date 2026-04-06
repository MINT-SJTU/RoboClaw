import { Outlet, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'
import Header from './Header'
import ToastContainer from './Toast'

export default function Layout() {
  const { connect, disconnect } = useWebSocket()
  const { fetchHardwareStatus } = useDashboard()
  const navigate = useNavigate()

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  // Auto-redirect to setup page when no hardware is configured
  useEffect(() => {
    fetchHardwareStatus().then(() => {
      const hs = useDashboard.getState().hardwareStatus
      if (hs && hs.arms.length === 0 && hs.cameras.length === 0) {
        navigate('/setup')
      }
    })
  }, [])

  return (
    <div className="flex flex-col h-screen bg-bg text-tx font-base">
      <Header />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
      <ToastContainer />
    </div>
  )
}
