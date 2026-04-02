import { Outlet } from 'react-router-dom'
import { useEffect } from 'react'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'
import { useSetup } from '../controllers/setup'
import Header from './Header'
import ToastContainer from './Toast'
import SetupWizardModal from '../views/SetupWizardModal'

export default function Layout() {
  const { connect, disconnect } = useWebSocket()
  const { fetchHardwareStatus } = useDashboard()

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  // Auto-open setup wizard when no hardware is configured
  useEffect(() => {
    fetchHardwareStatus().then(() => {
      const hs = useDashboard.getState().hardwareStatus
      if (hs && hs.arms.length === 0 && hs.cameras.length === 0) {
        useSetup.getState().setOpen(true)
      }
    })
  }, [])

  return (
    <div className="flex flex-col h-screen bg-bg text-tx">
      <Header />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
      <ToastContainer />
      <SetupWizardModal />
    </div>
  )
}
