import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchProviderStatus, saveProviderConfig, type ProviderOption } from '../controllers/provider'
import { useSetup } from '../controllers/setup'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'
import DeviceList from '../components/setup/DeviceList'
import DiscoveryWizard from '../components/setup/DiscoveryWizard'
import PermissionPanel from '../components/setup/PermissionPanel'
import { TemperatureHeatMap } from '../components/TemperatureHeatMap'
import { CalibrationPanel } from '../components/CalibrationPanel'
import { api, postJson } from '../controllers/api'

// Providers that make sense to show in the UI selector
const UI_PROVIDERS = [
  'anthropic', 'openai', 'deepseek', 'dashscope', 'gemini',
  'zhipu', 'moonshot', 'minimax',
  'openrouter', 'aihubmix', 'siliconflow', 'volcengine',
  'ollama', 'vllm',
  'custom',
]

function providerCategory(p: ProviderOption): 'standard' | 'gateway' | 'local' | 'custom' {
  if (p.name === 'custom') return 'custom'
  if (p.local) return 'local'
  if (p.name === 'openrouter' || p.name === 'aihubmix' || p.name === 'siliconflow' || p.name === 'volcengine') return 'gateway'
  return 'standard'
}

function needsApiKey(p: ProviderOption): boolean {
  const cat = providerCategory(p)
  return cat === 'standard' || cat === 'gateway' || cat === 'custom'
}

function needsBaseUrl(p: ProviderOption): boolean {
  const cat = providerCategory(p)
  return cat === 'gateway' || cat === 'local' || cat === 'custom'
}

export default function SettingsView() {
  const navigate = useNavigate()
  const { t } = useI18n()

  const { wizardActive, startWizard, loadDevices, loadCatalog, checkPermissions } = useSetup()
  const { fetchHardwareStatus } = useDashboard()

  const sessionState = useDashboard((s) => s.session.state)
  const sessionCalArm = useDashboard((s) => s.session.calibration_arm)
  const [calibratingArm, setCalibratingArm] = useState<string | null>(null)
  const [providerLoading, setProviderLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  // Restore calibration panel after page refresh
  useEffect(() => {
    if (sessionState === 'calibrating' && sessionCalArm && !calibratingArm) {
      setCalibratingArm(sessionCalArm)
    }
  }, [sessionState, sessionCalArm])

  // Provider state
  const [providers, setProviders] = useState<ProviderOption[]>([])
  const [activeProvider, setActiveProvider] = useState<string | null>(null)
  const [activeModel, setActiveModel] = useState('')
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [apiBase, setApiBase] = useState('')

  // HF config state
  const [hfEndpoint, setHfEndpoint] = useState('')
  const [hfEndpointMode, setHfEndpointMode] = useState<'default' | 'mirror' | 'custom'>('default')
  const [hfToken, setHfToken] = useState('')
  const [hfMaskedToken, setHfMaskedToken] = useState('')
  const [hfProxy, setHfProxy] = useState('')
  const [hfSaving, setHfSaving] = useState(false)
  const [hfNotice, setHfNotice] = useState('')
  const [hfError, setHfError] = useState('')

  useEffect(() => {
    loadDevices()
    loadCatalog()
    fetchHardwareStatus()
    checkPermissions()

    const hwInterval = setInterval(() => {
      if (document.visibilityState === 'visible') fetchHardwareStatus()
    }, 5000)

    let cancelled = false
    async function loadProvider() {
      try {
        const payload = await fetchProviderStatus()
        if (cancelled) return
        const uiProviders = payload.providers.filter(p => UI_PROVIDERS.includes(p.name))
        setProviders(uiProviders)
        setActiveProvider(payload.active_provider)
        setActiveModel(payload.default_model)

        const initial = payload.active_provider && uiProviders.some(p => p.name === payload.active_provider)
          ? payload.active_provider
          : 'custom'
        setSelectedProvider(initial)
        const sel = uiProviders.find(p => p.name === initial)
        if (sel) setApiBase(sel.api_base || '')
      } catch (loadError) {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : 'Failed to load settings.')
      } finally {
        if (!cancelled) setProviderLoading(false)
      }
    }
    loadProvider()

    async function loadHfConfig() {
      try {
        const data = await api('/api/system/hf-config')
        if (cancelled) return
        const ep = data.endpoint || ''
        setHfEndpoint(ep)
        setHfEndpointMode(
          ep === '' ? 'default' :
          ep === 'https://hf-mirror.com' ? 'mirror' : 'custom'
        )
        setHfMaskedToken(data.masked_token || '')
        setHfProxy(data.proxy || '')
      } catch (e) { console.warn('Failed to load HF config', e) }
    }
    loadHfConfig()

    return () => {
      cancelled = true
      clearInterval(hwInterval)
    }
  }, [])

  function handleSelectProvider(name: string) {
    setSelectedProvider(name)
    setError('')
    setNotice('')
    setApiKey('')
    const p = providers.find(pr => pr.name === name)
    setApiBase(p?.api_base || '')
  }

  async function handleSave(event: React.FormEvent) {
    event.preventDefault()
    setSaving(true)
    setError('')
    setNotice('')

    try {
      const payload = await saveProviderConfig({ provider: selectedProvider || 'custom', api_key: apiKey, api_base: apiBase })
      const uiProviders = payload.providers.filter(p => UI_PROVIDERS.includes(p.name))
      setProviders(uiProviders)
      setActiveProvider(payload.active_provider)
      setActiveModel(payload.default_model)
      setNotice(t('saveSuccess'))
      setApiKey('')
      window.setTimeout(() => navigate('/chat'), 600)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Failed to save settings.')
    } finally {
      setSaving(false)
    }
  }

  function handleHfEndpointMode(mode: 'default' | 'mirror' | 'custom') {
    setHfEndpointMode(mode)
    if (mode === 'default') setHfEndpoint('')
    else if (mode === 'mirror') setHfEndpoint('https://hf-mirror.com')
  }

  async function handleHfSave() {
    setHfSaving(true)
    setHfNotice('')
    setHfError('')
    try {
      const data = await postJson('/api/system/hf-config', {
        endpoint: hfEndpoint,
        token: hfToken,
        proxy: hfProxy,
      })
      setHfMaskedToken(data.masked_token || '')
      setHfToken('')
      setHfNotice(t('hfSaved'))
      setTimeout(() => setHfNotice(''), 3000)
    } catch (e) {
      setHfError(e instanceof Error ? e.message : t('saveSettings'))
    } finally {
      setHfSaving(false)
    }
  }

  const selected = providers.find(p => p.name === selectedProvider) || null

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('settingsTitle')}</h2>
        <p className="mt-1 text-sm text-tx3">{t('settingsDesc')}</p>
      </div>

      <div className="flex-1 p-6 grid grid-cols-2 gap-6 items-start max-[900px]:grid-cols-1">
        {/* Permissions card */}
        <PermissionPanel onFixed={() => checkPermissions()} />

        {/* Hardware section */}
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('settingsHardware')}</h3>
            {!wizardActive && (
              <button
                onClick={startWizard}
                className="px-4 py-2 bg-ac text-white rounded-lg text-sm font-semibold transition-all hover:bg-ac2 active:scale-[0.97] shadow-glow-ac"
              >
                {t('addDevice')}
              </button>
            )}
          </div>

          <DeviceList onCalibrate={async (alias) => {
            setCalibratingArm(alias)
            await postJson('/api/calibration/start', { arm_alias: alias })
          }} />
          {wizardActive && <div className="mt-4"><DiscoveryWizard /></div>}

          {calibratingArm && (
            <div className="mt-4">
              <CalibrationPanel
                armAlias={calibratingArm}
                onClose={() => { setCalibratingArm(null); fetchHardwareStatus() }}
              />
            </div>
          )}

          <div className="mt-4">
            <TemperatureHeatMap />
          </div>
        </section>

        {/* Provider section */}
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-yl">
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('settingsProvider')}</h3>

          {activeProvider && (
            <div className="rounded-lg bg-bg border border-bd/50 p-3 mb-4 flex items-center gap-3 text-sm">
              <span className="text-tx3">{t('currentProvider')}:</span>
              <span className="font-semibold text-tx">{providers.find(p => p.name === activeProvider)?.label || activeProvider}</span>
              {activeModel && (
                <>
                  <span className="text-tx3">|</span>
                  <span className="font-mono text-xs text-tx2">{activeModel}</span>
                </>
              )}
              <span className="w-2 h-2 rounded-full bg-gn" />
            </div>
          )}

          {providerLoading && <p className="text-tx3 text-sm">{t('loading')}</p>}
          {!providerLoading && (
            <>
              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2 mb-5">
                {providers.map(p => {
                  const isSelected = p.name === selectedProvider
                  const isActive = p.name === activeProvider
                  return (
                    <button
                      key={p.name}
                      onClick={() => handleSelectProvider(p.name)}
                      className={`relative flex flex-col items-center gap-1 px-3 py-2.5 rounded-lg border text-sm transition-all
                        ${isSelected
                          ? 'border-ac bg-ac/10 text-ac font-semibold shadow-glow-ac'
                          : 'border-bd/40 bg-bg text-tx2 hover:border-bd hover:bg-sf2'
                        }`}
                    >
                      <span className="truncate max-w-full">{p.label}</span>
                      <span className="flex items-center gap-1">
                        {p.configured && <span className="w-1.5 h-1.5 rounded-full bg-gn" />}
                        {isActive && <span className="text-2xs text-gn font-medium">{t('inUse')}</span>}
                      </span>
                    </button>
                  )
                })}
              </div>

              {selected && (
                <form onSubmit={handleSave} className="space-y-4 border-t border-bd/30 pt-4">
                  {error && (
                    <div className="rounded-lg border border-rd/30 border-l-4 border-l-rd bg-rd/5 p-3 text-sm text-rd">
                      {error}
                    </div>
                  )}
                  {notice && (
                    <div className="rounded-lg border border-gn/30 border-l-4 border-l-gn bg-gn/5 p-3 text-sm text-gn">
                      {notice}
                    </div>
                  )}

                  <div className="text-xs text-tx3 mb-2">
                    {t('configuring')}: <span className="font-semibold text-tx">{selected.label}</span>
                    {selected.configured && selected.masked_api_key && (
                      <span className="ml-2 font-mono text-tx2">({selected.masked_api_key})</span>
                    )}
                  </div>

                  {needsBaseUrl(selected) && (
                    <label className="flex flex-col gap-1 text-xs text-tx2 font-medium">
                      Base URL
                      <input
                        value={apiBase}
                        onChange={(e) => setApiBase(e.target.value)}
                        className="bg-bg border border-bd text-tx px-3 py-2.5 rounded-lg text-sm
                          focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3"
                        placeholder={t('baseUrlPlaceholder')}
                      />
                    </label>
                  )}

                  {needsApiKey(selected) && (
                    <label className="flex flex-col gap-1 text-xs text-tx2 font-medium">
                      API Key
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        className="bg-bg border border-bd text-tx px-3 py-2.5 rounded-lg text-sm
                          focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3"
                        placeholder={t('apiKeyPlaceholder')}
                      />
                    </label>
                  )}

                  <div className="flex items-center gap-3">
                    <button
                      type="submit"
                      disabled={saving}
                      className="bg-gn text-white px-5 py-2.5 rounded-lg text-sm font-semibold transition-all
                        hover:bg-gn/90 active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed shadow-glow-gn"
                    >
                      {saving ? t('saving') : t('saveSettings')}
                    </button>
                    <span className="text-xs text-tx3">{t('saveRedirectHint')}</span>
                  </div>
                </form>
              )}
            </>
          )}
        </section>

        {/* HuggingFace Hub config */}
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-gn col-span-2 max-[900px]:col-span-1">
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('hfConfig')}</h3>

          <div className="grid grid-cols-3 gap-4 max-[900px]:grid-cols-1">
            {/* Endpoint */}
            <div className="space-y-2">
              <label className="text-xs text-tx2 font-medium">{t('hfEndpoint')}</label>
              <div className="flex flex-col gap-1.5">
                {(['default', 'mirror', 'custom'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => handleHfEndpointMode(mode)}
                    className={`px-3 py-2 rounded-lg text-sm text-left transition-all border
                      ${hfEndpointMode === mode
                        ? 'border-ac bg-ac/10 text-ac font-semibold'
                        : 'border-bd/40 bg-bg text-tx2 hover:border-bd'}`}
                  >
                    {mode === 'default' ? t('hfDefault') : mode === 'mirror' ? t('hfMirror') : t('hfCustomEndpoint')}
                  </button>
                ))}
              </div>
              {hfEndpointMode === 'custom' && (
                <input
                  value={hfEndpoint}
                  onChange={(e) => setHfEndpoint(e.target.value)}
                  placeholder="https://..."
                  className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm
                    focus:outline-none focus:border-ac"
                />
              )}
            </div>

            {/* Token */}
            <div className="space-y-2">
              <label className="text-xs text-tx2 font-medium">{t('hfToken')}</label>
              {hfMaskedToken && (
                <div className="text-xs text-tx3 font-mono bg-bg rounded px-2 py-1 border border-bd/30">
                  {hfMaskedToken}
                </div>
              )}
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder={t('hfTokenPlaceholder')}
                className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
            </div>

            {/* Proxy */}
            <div className="space-y-2">
              <label className="text-xs text-tx2 font-medium">{t('hfProxy')}</label>
              <input
                value={hfProxy}
                onChange={(e) => setHfProxy(e.target.value)}
                placeholder={t('hfProxyPlaceholder')}
                className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={handleHfSave}
              disabled={hfSaving}
              className="bg-gn text-white px-5 py-2 rounded-lg text-sm font-semibold transition-all
                hover:bg-gn/90 active:scale-[0.97] disabled:opacity-30 shadow-glow-gn"
            >
              {hfSaving ? t('saving') : t('saveSettings')}
            </button>
            {hfNotice && <span className="text-xs text-gn">{hfNotice}</span>}
            {hfError && <span className="text-xs text-rd">{hfError}</span>}
          </div>
        </section>
      </div>
    </div>
  )
}
