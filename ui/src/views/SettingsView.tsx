import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchProviderStatus, saveProviderConfig } from '../controllers/provider'
import { useI18n } from '../controllers/i18n'

export default function SettingsView() {
  const navigate = useNavigate()
  const { t } = useI18n()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [savedKeyMask, setSavedKeyMask] = useState('')
  const [hasSavedKey, setHasSavedKey] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const payload = await fetchProviderStatus()
        if (cancelled) return
        setApiBase(payload.custom_provider.api_base || '')
        setSavedKeyMask(payload.custom_provider.masked_api_key || '')
        setHasSavedKey(Boolean(payload.custom_provider.has_api_key))
      } catch (loadError) {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : 'Failed to load settings.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [])

  async function handleSave(event: React.FormEvent) {
    event.preventDefault()
    setSaving(true)
    setError('')
    setNotice('')

    try {
      const payload = await saveProviderConfig({ api_key: apiKey, api_base: apiBase })
      setApiBase(payload.custom_provider.api_base || '')
      setSavedKeyMask(payload.custom_provider.masked_api_key || '')
      setHasSavedKey(Boolean(payload.custom_provider.has_api_key))
      setNotice(t('saveSuccess'))
      setApiKey('')
      window.setTimeout(() => navigate('/chat'), 600)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Failed to save settings.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/40 p-4">
        <h2 className="text-xl font-bold tracking-tight">{t('settingsTitle')}</h2>
        <p className="mt-1.5 text-sm text-tx2">{t('settingsDesc')}</p>
      </div>

      <div className="flex-1 p-5 max-w-3xl">
        {loading && <p className="text-tx2">{t('loading')}</p>}
        {!loading && (
          <form onSubmit={handleSave} className="space-y-5">
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

            <section className="rounded-lg border border-bd/30 bg-white p-6 shadow-card space-y-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium">{t('globalProvider')}</h3>

              <label className="flex flex-col gap-1 text-xs text-tx2">
                {t('baseUrl')}
                <input
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  className="bg-sf2 border border-bd text-tx px-3 py-2 rounded text-sm focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3"
                  placeholder="https://your-openai-compatible-endpoint/v1"
                />
              </label>

              <label className="flex flex-col gap-1 text-xs text-tx2">
                {t('apiKey')}
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="bg-sf2 border border-bd text-tx px-3 py-2 rounded text-sm focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3"
                  placeholder={t('apiKeyPlaceholder')}
                />
              </label>

              <div className="rounded-lg border border-bd/30 bg-sf p-3 text-sm text-tx2">
                <div>{t('savedStatus')}: {hasSavedKey ? t('saved') : t('notSaved')}</div>
                {savedKeyMask && <div>{t('savedKey')}: {savedKeyMask}</div>}
              </div>
            </section>

            <div className="flex items-center gap-4">
              <button
                type="submit"
                disabled={saving}
                className="bg-gn text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors hover:bg-gn/90 active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {saving ? t('saving') : t('saveSettings')}
              </button>
              <span className="text-sm text-tx2">{t('saveRedirectHint')}</span>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
