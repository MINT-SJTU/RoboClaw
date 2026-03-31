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
    <header className="flex items-center gap-3 px-4 py-2 bg-sf border-b border-bd flex-wrap">
      <h1 className="text-lg font-semibold text-ac whitespace-nowrap">RoboClaw</h1>

      <span
        className={`inline-block px-2 py-0.5 rounded-sm text-2xs font-semibold tracking-wide ${
          connected ? 'bg-gn/15 text-gn' : 'bg-rd/15 text-rd'
        }`}
      >
        {connected ? t('connected') : t('disconnected')}
      </span>

      <nav className="flex items-center gap-1 ml-4">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              location.pathname === item.path
                ? 'bg-ac/10 text-ac font-medium'
                : 'text-tx2 hover:text-tx hover:bg-sf'
            }`}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="flex-1" />

      {networkInfo && (
        <span className="text-2xs text-tx2 whitespace-nowrap mr-2">
          {networkInfo.lan_ip}:{networkInfo.port}
        </span>
      )}

      <button
        onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
        className="px-2.5 py-1 border border-bd rounded text-sm text-tx2 hover:text-tx hover:bg-sf transition-colors"
      >
        {locale === 'zh' ? 'EN' : '中文'}
      </button>
    </header>
  )
}
