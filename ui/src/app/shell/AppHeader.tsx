import { useEffect, useMemo } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useChatSocket } from '@/domains/chat/store/useChatSocket'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'
import { StatusPill } from '@/shared/ui'

export default function AppHeader() {
  const location = useLocation()
  const { connected } = useChatSocket()
  const networkInfo = useHardwareStore((state) => state.networkInfo)
  const fetchNetworkInfo = useHardwareStore((state) => state.fetchNetworkInfo)
  const { t, locale, setLocale } = useI18n()

  useEffect(() => {
    fetchNetworkInfo()
  }, [fetchNetworkInfo])

  const pageTitle = useMemo(() => {
    if (location.pathname.startsWith('/control')) return t('controlCenter')
    if (location.pathname.startsWith('/datasets/explorer')) return t('datasetExplorer')
    if (location.pathname.startsWith('/datasets')) return t('datasetsNav')
    if (location.pathname.startsWith('/curation/data-overview')) return t('dataOverview')
    if (location.pathname.startsWith('/curation/text-alignment')) return t('textAlignment')
    if (location.pathname.startsWith('/curation/quality')) return t('qualityWorkbench')
    if (location.pathname.startsWith('/curation')) return t('pipelineNav')
    if (location.pathname.startsWith('/logs')) return t('logs')
    if (location.pathname.startsWith('/settings/hardware')) return t('settingsHardware')
    if (location.pathname.startsWith('/settings/provider')) return t('settingsProvider')
    if (location.pathname.startsWith('/settings/hub')) return t('hfConfig')
    if (location.pathname.startsWith('/settings')) return t('settings')
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
