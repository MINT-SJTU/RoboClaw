import { useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'

export default function Header() {
  const location = useLocation()
  const { connected } = useWebSocket()
  const { networkInfo, fetchNetworkInfo } = useDashboard()
  const { t, locale, setLocale } = useI18n()

  useEffect(() => {
    fetchNetworkInfo()
  }, [fetchNetworkInfo])

  const navItems = [
    { path: '/dashboard', label: t('dataCollection') },
    { path: '/chat', label: t('chat') },
    { path: '/settings', label: t('settings') },
  ]

  return (
    <header className="flex items-center gap-3 px-4 py-2 bg-white border-b border-bd/40 shadow-sm flex-wrap">
      <h1 className="text-base font-bold tracking-tight text-ac whitespace-nowrap">RoboClaw</h1>

      <span
        className={`flex items-center gap-1.5 text-2xs font-medium ${
          connected ? 'text-gn' : 'text-rd'
        }`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-gn' : 'bg-rd'}`} />
        {connected ? t('connected') : t('disconnected')}
      </span>

      <nav className="flex items-center gap-1 ml-4">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`px-3 py-1 text-sm transition-colors ${
              location.pathname === item.path
                ? 'text-ac font-medium border-b-2 border-ac'
                : 'text-tx2 hover:text-tx'
            }`}
          >
            {item.label}
          </Link>
        ))}
        <Link
          to="/setup"
          className={`px-3 py-1 text-sm transition-colors ${
            location.pathname === '/setup'
              ? 'text-ac font-medium border-b-2 border-ac'
              : 'text-tx2 hover:text-tx'
          }`}
        >
          {t('setup')}
        </Link>
      </nav>

      <div className="flex-1" />

      {networkInfo && (
        <span className="text-2xs text-tx3 font-mono whitespace-nowrap mr-2">
          {networkInfo.lan_ip}:{networkInfo.port}
        </span>
      )}

      <button
        onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
        className="text-tx3 hover:text-tx2 text-xs transition-colors"
      >
        {locale === 'zh' ? 'EN' : '中文'}
      </button>
    </header>
  )
}
