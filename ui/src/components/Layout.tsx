import { Outlet, Link, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'

export default function Layout() {
  const location = useLocation()
  const { connect, disconnect, connected, sessionId } = useWebSocket()
  const { networkInfo, fetchNetworkInfo } = useDashboard()

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  useEffect(() => {
    fetchNetworkInfo()
  }, [fetchNetworkInfo])

  const navItems = [
    { path: '/dashboard', label: '采集' },
    { path: '/chat', label: '对话' },
    { path: '/settings', label: '设置' },
  ]

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4">
          <h1 className="text-2xl font-bold">RoboClaw</h1>
          <div className="mt-2 text-sm">
            <span className={`inline-block w-2 h-2 rounded-full mr-2 ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            {connected ? '已连接' : '未连接'}
          </div>
          {sessionId && (
            <div className="mt-2 text-xs text-gray-400 break-all">
              会话: {sessionId}
            </div>
          )}
        </div>

        <nav className="mt-8 flex-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`block px-4 py-3 hover:bg-gray-700 transition-colors ${
                location.pathname === item.path ? 'bg-gray-700 border-l-4 border-blue-500' : ''
              }`}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* LAN info */}
        {networkInfo && (
          <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
            {networkInfo.lan_ip}:{networkInfo.port}
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
