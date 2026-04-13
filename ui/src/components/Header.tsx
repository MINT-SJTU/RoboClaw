import { useEffect, useMemo } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useWebSocket } from '../controllers/connection'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'
import { StatusPill } from './ux'

export default function Header() {
  const location = useLocation()
  const { connected } = useWebSocket()
  const { networkInfo, fetchNetworkInfo } = useDashboard()
  const { t, locale, setLocale } = useI18n()

  useEffect(() => {
    fetchNetworkInfo()
  }, [fetchNetworkInfo])

  const pageTitle = useMemo(() => {
    if (location.pathname.startsWith('/control')) return t('controlCenter')
    if (location.pathname.startsWith('/data')) return t('dataCenter')
    if (location.pathname.startsWith('/explorer')) return t('datasetExplorer')
    if (location.pathname.startsWith('/quality')) return t('qualityWorkbench')
    if (location.pathname.startsWith('/text-alignment')) return t('textAlignment')
    if (location.pathname.startsWith('/workflow')) return t('workflow')
    if (location.pathname.startsWith('/logs')) return t('logs')
    if (location.pathname.startsWith('/settings')) return t('settings')
    if (location.pathname.startsWith('/chat')) return t('assistantChat')
    return 'RoboClaw'
  }, [location.pathname, t])

  return (
    <header className="app-topbar">
      <div className="app-topbar__title">
        <div className="space-y-2">
          <Link to="/control" className="display-title text-[1.95rem] text-tx">
            RoboClaw
          </Link>
          <div className="eyebrow">{pageTitle}</div>
        </div>
      </div>

      <div className="app-topbar__actions">
        {networkInfo && (
          <div className="rounded-full bg-white/70 px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-tx2">
            {networkInfo.lan_ip}:{networkInfo.port}
          </div>
        )}

        <StatusPill active={connected}>
          {connected ? t('connected') : t('disconnected')}
        </StatusPill>

        <button
          onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
          className="app-topbar__locale"
        >
          {locale === 'zh' ? 'EN' : '中文'}
        </button>
      </div>
    </header>
  )
}
