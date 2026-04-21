import { useEffect, useState } from 'react'
import SettingsPageFrame from '@/domains/settings/components/SettingsPageFrame'
import { useI18n } from '@/i18n'
import {
  classifyHfEndpoint,
  fetchHfConfig,
  saveHfConfig,
  type HfEndpointMode,
} from '@/domains/hub/api/hubConfigApi'

function modeToLabel(mode: HfEndpointMode, labels: {
  default: string
  mirror: string
  custom: string
}) {
  if (mode === 'default') return labels.default
  if (mode === 'mirror') return labels.mirror
  return labels.custom
}

export default function HubSettingsPage() {
  const { t } = useI18n()
  const [hfEndpoint, setHfEndpoint] = useState('')
  const [hfEndpointMode, setHfEndpointMode] = useState<HfEndpointMode>('default')
  const [hfToken, setHfToken] = useState('')
  const [hfMaskedToken, setHfMaskedToken] = useState('')
  const [hfProxy, setHfProxy] = useState('')
  const [hfSaving, setHfSaving] = useState(false)
  const [hfNotice, setHfNotice] = useState('')
  const [hfError, setHfError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function loadConfig() {
      try {
        const data = await fetchHfConfig()
        if (cancelled) return
        const endpoint = data.endpoint || ''
        setHfEndpoint(endpoint)
        setHfEndpointMode(classifyHfEndpoint(endpoint))
        setHfMaskedToken(data.masked_token || '')
        setHfProxy(data.proxy || '')
      } catch (error) {
        if (!cancelled) {
          setHfError(error instanceof Error ? error.message : String(error))
        }
      }
    }

    loadConfig()
    return () => { cancelled = true }
  }, [])

  function handleEndpointMode(mode: HfEndpointMode) {
    setHfEndpointMode(mode)
    if (mode === 'default') setHfEndpoint('')
    if (mode === 'mirror') setHfEndpoint('https://hf-mirror.com')
  }

  async function handleSave() {
    setHfSaving(true)
    setHfNotice('')
    setHfError('')

    try {
      const data = await saveHfConfig({
        endpoint: hfEndpoint,
        token: hfToken,
        proxy: hfProxy,
      })
      setHfMaskedToken(data.masked_token || '')
      setHfToken('')
      setHfNotice(t('hfSaved'))
      setTimeout(() => setHfNotice(''), 3000)
    } catch (error) {
      setHfError(error instanceof Error ? error.message : t('saveSettings'))
    } finally {
      setHfSaving(false)
    }
  }

  const summaryItems = [
    {
      label: t('hfEndpoint'),
      value: modeToLabel(hfEndpointMode, {
        default: t('hfDefault'),
        mirror: t('hfMirror'),
        custom: t('hfCustomEndpoint'),
      }),
    },
    {
      label: t('hfToken'),
      value: hfMaskedToken ? t('saved') : t('settingsNotConfigured'),
    },
    {
      label: t('hfProxy'),
      value: hfProxy || t('settingsNotConfigured'),
    },
  ]

  return (
    <SettingsPageFrame
      title={t('hfConfig')}
      description={t('settingsHubDesc')}
    >
      <div className="grid gap-6 xl:grid-cols-[minmax(280px,0.75fr)_minmax(0,1.25fr)]">
        <section className="rounded-2xl border border-bd/30 bg-white p-5 shadow-card">
          <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{t('settingsHubSummary')}</h3>
          <div className="mt-5 grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
            {summaryItems.map((item) => (
              <div key={item.label} className="min-w-0 rounded-xl border border-bd/30 bg-sf px-4 py-3">
                <div className="text-2xs uppercase tracking-[0.16em] text-tx3">{item.label}</div>
                <div className="mt-2 break-words text-sm font-semibold text-tx">{item.value}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="min-w-0 space-y-3">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{t('hfEndpoint')}</h3>
                <p className="mt-2 text-sm text-tx3">{t('settingsHubEndpointDesc')}</p>
              </div>
              {(['default', 'mirror', 'custom'] as HfEndpointMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => handleEndpointMode(mode)}
                  className={`w-full rounded-xl border px-3 py-3 text-left text-sm transition-all ${
                    hfEndpointMode === mode
                      ? 'border-ac bg-ac/10 font-semibold text-ac'
                      : 'border-bd/30 bg-white text-tx2 hover:border-ac/30'
                  }`}
                >
                  {modeToLabel(mode, {
                    default: t('hfDefault'),
                    mirror: t('hfMirror'),
                    custom: t('hfCustomEndpoint'),
                  })}
                </button>
              ))}
              {hfEndpointMode === 'custom' && (
                <input
                  value={hfEndpoint}
                  onChange={(e) => setHfEndpoint(e.target.value)}
                  placeholder="https://..."
                  className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac"
                />
              )}
            </div>

            <div className="min-w-0 space-y-3">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{t('hfToken')}</h3>
                <p className="mt-2 text-sm text-tx3">{t('settingsHubTokenDesc')}</p>
              </div>
              {hfMaskedToken && (
                <div className="rounded-xl border border-bd/30 bg-white px-3 py-2.5 font-mono text-xs text-tx2">
                  {hfMaskedToken}
                </div>
              )}
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder={t('hfTokenPlaceholder')}
                className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac"
              />
            </div>

            <div className="min-w-0 space-y-3">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{t('hfProxy')}</h3>
                <p className="mt-2 text-sm text-tx3">{t('settingsHubProxyDesc')}</p>
              </div>
              <input
                value={hfProxy}
                onChange={(e) => setHfProxy(e.target.value)}
                placeholder={t('hfProxyPlaceholder')}
                className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac"
              />
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => { void handleSave() }}
              disabled={hfSaving}
              className="rounded-full bg-gn px-5 py-2.5 text-sm font-semibold text-white shadow-glow-gn transition-all hover:bg-gn/90 disabled:opacity-40"
            >
              {hfSaving ? t('saving') : t('saveSettings')}
            </button>
            {hfNotice && <span className="text-sm text-gn">{hfNotice}</span>}
            {hfError && <span className="text-sm text-rd">{hfError}</span>}
          </div>
        </section>
      </div>
    </SettingsPageFrame>
  )
}
