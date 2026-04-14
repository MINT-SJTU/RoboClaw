import { Link, Outlet, useLocation } from 'react-router-dom'
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

const NAV_ICONS: Record<string, JSX.Element> = {
  '/control': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  ),
  '/data': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
    </svg>
  ),
  '/explorer': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="M21 21l-4.35-4.35" />
    </svg>
  ),
  '/quality': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 11l3 3L22 4" />
      <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
    </svg>
  ),
  '/text-alignment': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 6h16" />
      <path d="M4 12h10" />
      <path d="M4 18h14" />
    </svg>
  ),
  '/settings': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
    </svg>
  ),
  '/logs': (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  ),
}

export default function Layout() {
  const location = useLocation()
  const { connect, disconnect, connected, messages } = useWebSocket()
  const { fetchHardwareStatus } = useDashboard()
  const { t } = useI18n()
  const [chatOpen, setChatOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  useEffect(() => {
    fetchHardwareStatus()
  }, [fetchHardwareStatus, location.pathname])

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
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={{ transition: 'transform 0.2s ease', transform: sidebarCollapsed ? 'rotate(180deg)' : 'none' }}
            >
              <polyline points="11 17 6 12 11 7" />
              <polyline points="18 17 13 12 18 7" />
            </svg>
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
                <span className="app-sidebar__link-icon">
                  {NAV_ICONS[item.path]}
                </span>
                {!sidebarCollapsed && <span className="app-sidebar__link-label">{item.label}</span>}
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
