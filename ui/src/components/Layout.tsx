import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'
import ChatPanel from './ChatPanel'
import Header from './Header'
import ToastContainer from './Toast'

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

export default function Layout() {
  const location = useLocation()
  const navigate = useNavigate()
  const { connect, disconnect, connected, messages } = useWebSocket()
  const { fetchHardwareStatus } = useDashboard()
  const { t } = useI18n()
  const [chatOpen, setChatOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  // Auto-redirect to setup page when no hardware is configured
  useEffect(() => {
    fetchHardwareStatus().then(() => {
      const hs = useDashboard.getState().hardwareStatus
      const shouldGuardControlRoute =
        location.pathname === '/'
        || location.pathname === '/control'
        || location.pathname === '/dashboard'
      if (shouldGuardControlRoute && hs && hs.arms.length === 0 && hs.cameras.length === 0) {
        navigate('/settings')
      }
    })
  }, [fetchHardwareStatus, location.pathname, navigate])

  const navItems = [
    { path: '/control', label: t('controlCenter') },
    { path: '/data', label: t('dataCenter') },
    { path: '/explorer', label: t('datasetExplorer') },
    { path: '/quality', label: t('qualityWorkbench') },
    { path: '/text-alignment', label: t('textAlignment') },
    { path: '/settings', label: t('settings') },
    { path: '/logs', label: t('logs') },
  ]

  return (
    <div className="app-shell">
      <aside className={cn('app-sidebar', sidebarCollapsed && 'app-sidebar--collapsed')}>
        <div className="app-sidebar__header">
          <button
            type="button"
            onClick={() => setSidebarCollapsed((value) => !value)}
            className="app-sidebar__toggle"
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? '>' : '<'}
          </button>
        </div>

        <nav className="app-sidebar__nav">
          {navItems.map((item) => {
            const active =
              location.pathname === item.path
              || location.pathname.startsWith(`${item.path}/`)
            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn('app-sidebar__link', active && 'app-sidebar__link--active')}
                title={sidebarCollapsed ? item.label : undefined}
              >
                <span className="app-sidebar__link-indicator" />
                {!sidebarCollapsed && <span>{item.label}</span>}
              </Link>
            )
          })}
        </nav>
      </aside>

      <div className="app-shell__main">
        <Header />
        <main className="app-shell__content">
          <Outlet />
        </main>

        <div className="chat-widget">
          {chatOpen && (
            <div className="chat-widget__panel">
              <ChatPanel variant="widget" onClose={() => setChatOpen(false)} />
            </div>
          )}

          <button
            type="button"
            onClick={() => setChatOpen((value) => !value)}
            className={cn('chat-widget__trigger', chatOpen && 'chat-widget__trigger--open')}
            aria-expanded={chatOpen}
            aria-label={chatOpen ? 'Close chat' : 'Open chat'}
          >
            <span className={cn('chat-widget__dot', connected && 'chat-widget__dot--live')} />
            <span className="chat-widget__label">AI</span>
            {!chatOpen && messages.length > 0 && (
              <span className="chat-widget__count">{Math.min(messages.length, 99)}</span>
            )}
          </button>
        </div>

        <ToastContainer />
      </div>
    </div>
  )
}
