import { useEffect, useMemo, useState } from 'react'
import SettingsPageFrame from '@/domains/settings/components/SettingsPageFrame'
import { useI18n } from '@/i18n'
import {
  fetchProviderStatus,
  saveProviderConfig,
  testProviderConfig,
  type ProviderDiagnosticResponse,
  type ProviderOption,
} from '@/domains/provider/api/providerApi'

const UI_PROVIDERS = [
  'openai_codex',
  'anthropic', 'openai', 'deepseek', 'dashscope', 'gemini',
  'zhipu', 'moonshot', 'minimax',
  'openrouter', 'aihubmix', 'siliconflow', 'volcengine',
  'ollama', 'vllm',
  'custom',
]

const DEFAULT_PROVIDER_MODELS: Record<string, string> = {
  custom: 'claude-sonnet-4-5-20250929',
  anthropic: 'claude-sonnet-4-5',
  openai: 'gpt-4.1',
  openrouter: 'anthropic/claude-sonnet-4-5',
  aihubmix: 'claude-sonnet-4-5-20250929',
  siliconflow: 'Qwen/Qwen3-Coder-480B-A35B-Instruct',
  volcengine: 'doubao-seed-1-6',
  deepseek: 'deepseek-chat',
  dashscope: 'qwen-plus',
  gemini: 'gemini-2.5-pro',
  zhipu: 'glm-4.5',
  moonshot: 'kimi-k2-0711-preview',
  minimax: 'abab6.5s-chat',
  openai_codex: 'openai-codex/gpt-5.5',
}

function suggestedModelForProvider(name: string, activeProvider: string | null, activeModel: string): string {
  if (name === activeProvider && activeModel) return activeModel
  return ''
}

function providerCategory(p: ProviderOption): 'standard' | 'gateway' | 'local' | 'custom' {
  if (p.name === 'custom') return 'custom'
  if (p.local) return 'local'
  if (p.name === 'openrouter' || p.name === 'aihubmix' || p.name === 'siliconflow' || p.name === 'volcengine') return 'gateway'
  return 'standard'
}

function needsApiKey(p: ProviderOption): boolean {
  if (p.oauth) return false
  const category = providerCategory(p)
  return category === 'standard' || category === 'gateway' || category === 'custom'
}

function needsBaseUrl(p: ProviderOption): boolean {
  if (p.oauth) return false
  const category = providerCategory(p)
  return category === 'gateway' || category === 'local' || category === 'custom'
}

export default function ProviderSettingsPage() {
  const { t } = useI18n()
  const [providers, setProviders] = useState<ProviderOption[]>([])
  const [activeProvider, setActiveProvider] = useState<string | null>(null)
  const [activeModel, setActiveModel] = useState('')
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const [modelName, setModelName] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [diagnostic, setDiagnostic] = useState<ProviderDiagnosticResponse | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    let cancelled = false

    async function loadProvider() {
      try {
        const payload = await fetchProviderStatus()
        if (cancelled) return
        const uiProviders = payload.providers.filter((provider) => UI_PROVIDERS.includes(provider.name))
        setProviders(uiProviders)
        setActiveProvider(payload.active_provider)
        setActiveModel(payload.default_model)
        setModelName(payload.default_model || '')

        const initial = payload.active_provider && uiProviders.some((provider) => provider.name === payload.active_provider)
          ? payload.active_provider
          : 'custom'
        setSelectedProvider(initial)
        setApiBase(uiProviders.find((provider) => provider.name === initial)?.api_base || '')
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load settings.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadProvider()
    return () => { cancelled = true }
  }, [])

  function handleSelectProvider(name: string) {
    setSelectedProvider(name)
    setError('')
    setNotice('')
    setDiagnostic(null)
    setApiKey('')
    setApiBase(providers.find((provider) => provider.name === name)?.api_base || '')
    setModelName(suggestedModelForProvider(name, activeProvider, activeModel))
  }

  async function handleSave(event: React.FormEvent) {
    event.preventDefault()
    setSaving(true)
    setError('')
    setNotice('')

    try {
      const payload = await saveProviderConfig({
        provider: selectedProvider || 'custom',
        model: modelName,
        api_key: apiKey,
        api_base: apiBase,
      })
      const uiProviders = payload.providers.filter((provider) => UI_PROVIDERS.includes(provider.name))
      setProviders(uiProviders)
      setActiveProvider(payload.active_provider)
      setActiveModel(payload.default_model)
      setModelName(payload.default_model || modelName)
      setNotice(t('saveSuccess'))
      setApiKey('')
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Failed to save settings.')
    } finally {
      setSaving(false)
    }
  }

  async function handleTestProvider() {
    setTesting(true)
    setError('')
    setNotice('')
    setDiagnostic(null)

    try {
      const payload = await testProviderConfig({
        provider: selectedProvider || 'custom',
        model: modelName,
        api_key: apiKey,
        api_base: apiBase,
      })
      setDiagnostic(payload)
    } catch (testError) {
      setError(testError instanceof Error ? testError.message : 'Provider connection test failed.')
    } finally {
      setTesting(false)
    }
  }

  const selected = providers.find((provider) => provider.name === selectedProvider) || null

  const groups = useMemo(() => ([
    { key: 'standard', title: t('providerGroupStandard') },
    { key: 'gateway', title: t('providerGroupGateway') },
    { key: 'local', title: t('providerGroupLocal') },
    { key: 'custom', title: t('providerGroupCustom') },
  ]).map((group) => ({
    ...group,
    items: providers.filter((provider) => providerCategory(provider) === group.key),
  })).filter((group) => group.items.length > 0), [providers, t])

  return (
    <SettingsPageFrame
      title={t('settingsProvider')}
      description={t('settingsProviderDesc')}
    >
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(340px,0.8fr)]">
        <div className="space-y-6">
          <section className="rounded-2xl border border-bd/30 bg-white p-5 shadow-card">
            <div className="flex flex-wrap items-center gap-3">
              <div className="text-sm font-semibold text-tx">{t('currentProvider')}</div>
              <span className="rounded-full bg-ac/10 px-3 py-1 text-sm font-semibold text-ac">
                {providers.find((provider) => provider.name === activeProvider)?.label || t('providerNotConfigured')}
              </span>
              <span className="text-sm text-tx3">
                {activeModel || t('settingsNoModel')}
              </span>
            </div>
          </section>

          <section className="rounded-2xl border border-ac/20 bg-ac/5 p-4 text-sm leading-6 text-tx2 shadow-card">
            <div>国内用户推荐：DeepSeek / 智谱 / 月之暗面</div>
            <div>海外用户推荐：OpenAI / Anthropic</div>
            <div>本地部署：Ollama（无需 API Key）</div>
          </section>

          {loading && (
            <section className="rounded-2xl border border-bd/30 bg-white p-5 text-sm text-tx3 shadow-card">
              {t('loading')}
            </section>
          )}

          {!loading && groups.map((group) => (
            <section key={group.key} className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
              <div className="mb-4">
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">{group.title}</h3>
              </div>

              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {group.items.map((provider) => {
                  const isSelected = provider.name === selectedProvider
                  const isActive = provider.name === activeProvider
                  return (
                    <button
                      key={provider.name}
                      type="button"
                      onClick={() => handleSelectProvider(provider.name)}
                      className={`rounded-2xl border px-4 py-4 text-left transition-all ${
                        isSelected
                          ? 'border-ac bg-ac/10 text-ac shadow-glow-ac'
                          : 'border-bd/30 bg-white text-tx hover:border-ac/30'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="break-words text-sm font-semibold">{provider.label}</div>
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-2xs">
                            {provider.configured && (
                              <span className="rounded-full bg-gn/10 px-2 py-0.5 font-medium text-gn">
                                {t('saved')}
                              </span>
                            )}
                            {isActive && (
                              <span className="rounded-full bg-ac/10 px-2 py-0.5 font-medium text-ac">
                                {t('inUse')}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </section>
          ))}
        </div>

        <div>
          {selected && (
            <form onSubmit={handleSave} className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card xl:sticky xl:top-6">
              <div className="border-b border-bd/30 pb-4">
                <div className="text-2xs font-semibold uppercase tracking-[0.18em] text-tx3">
                  {t('configuring')}
                </div>
                <h3 className="mt-2 text-xl font-semibold text-tx">{selected.label}</h3>
                {selected.configured && selected.masked_api_key && (
                  <div className="mt-2 font-mono text-xs text-tx2">{selected.masked_api_key}</div>
                )}
                {selected.api_key_warning && (
                  <div className="mt-3 rounded-xl border border-rd/30 bg-rd/5 p-3 text-xs text-rd">
                    {selected.api_key_warning}
                  </div>
                )}
              </div>

              <div className="mt-5 space-y-4">
                {error && (
                  <div className="rounded-xl border border-rd/30 border-l-4 border-l-rd bg-rd/5 p-3 text-sm text-rd">
                    {error}
                  </div>
                )}
                {notice && (
                  <div className="rounded-xl border border-gn/30 border-l-4 border-l-gn bg-gn/5 p-3 text-sm text-gn">
                    {notice}
                  </div>
                )}
                {diagnostic && (
                  <div className={`rounded-xl border p-3 text-sm leading-relaxed ${
                    diagnostic.capability === 'agent'
                      ? 'border-gn/30 bg-gn/5 text-gn'
                      : diagnostic.capability === 'planner'
                        ? 'border-yl/40 bg-yl/10 text-tx'
                        : 'border-rd/30 bg-rd/5 text-rd'
                  }`}>
                    <div className="font-semibold">
                      {diagnostic.capability === 'agent'
                        ? 'Agent 能力可用'
                        : diagnostic.capability === 'planner'
                          ? '只能作为规划模型使用'
                          : 'Provider 不可用'}
                    </div>
                    <div className="mt-1 text-xs text-tx2">{diagnostic.recommendation}</div>
                    <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
                      <div className="rounded-lg bg-white/70 p-2">
                        <div className="font-semibold text-tx">普通聊天</div>
                        <div className={diagnostic.text.ok ? 'text-gn' : 'text-rd'}>
                          {diagnostic.text.ok ? '通过' : `失败：${diagnostic.text.errorCode || diagnostic.text.message}`}
                        </div>
                      </div>
                      <div className="rounded-lg bg-white/70 p-2">
                        <div className="font-semibold text-tx">Agent 工具调用</div>
                        <div className={diagnostic.tools.ok ? 'text-gn' : 'text-rd'}>
                          {diagnostic.tools.ok ? '通过' : `未通过：${diagnostic.tools.errorCode || diagnostic.tools.message}`}
                        </div>
                      </div>
                    </div>
                    {diagnostic.skillMatrix?.length > 0 && (
                      <div className="mt-3 space-y-2">
                        <div className="text-xs font-semibold text-tx">Skill / 工具链兼容性</div>
                        <div className="grid gap-2">
                          {diagnostic.skillMatrix.map((item) => (
                            <div key={item.id} className="rounded-lg bg-white/70 p-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="font-semibold text-tx">{item.label}</div>
                                <span className={`rounded-full px-2 py-0.5 text-2xs font-semibold ${
                                  item.ok ? 'bg-gn/10 text-gn' : 'bg-rd/10 text-rd'
                                }`}>
                                  {item.ok ? '可执行' : '不可执行'}
                                </span>
                              </div>
                              <div className="mt-1 text-xs text-tx3">{item.description}</div>
                              <div className="mt-1 text-xs text-tx2">
                                工具：{item.expectedTool} · {item.message}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {selected.oauth && (
                  <div className="rounded-xl border border-ac/25 bg-ac/5 p-3 text-sm leading-relaxed text-tx2">
                    <div className="font-semibold text-tx">OAuth provider，不需要在网页里填写 API key。</div>
                    <div className="mt-1">
                      保存后，后端会使用 OpenAI Codex 作为平台 AI。若服务器还没有登录，请在运行后端的机器上执行：
                    </div>
                    <code className="mt-2 block rounded-lg bg-bg px-3 py-2 font-mono text-xs text-tx">
                      roboclaw provider login openai-codex
                    </code>
                    <div className="mt-1 text-xs text-tx3">
                      如果这台机器已经有 Codex CLI 登录态，RoboClaw 会优先复用本机 OAuth token，不会把 token 写进开源代码。
                    </div>
                  </div>
                )}

                {needsBaseUrl(selected) && (
                  <label className="block">
                    <div className="mb-1.5 text-xs font-medium text-tx2">Base URL</div>
                    <input
                      value={apiBase}
                      onChange={(e) => setApiBase(e.target.value)}
                      className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac focus:shadow-glow-ac"
                      placeholder={t('baseUrlPlaceholder')}
                    />
                  </label>
                )}

                {needsApiKey(selected) && (
                  <label className="block">
                    <div className="mb-1.5 text-xs font-medium text-tx2">API Key</div>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac focus:shadow-glow-ac"
                      placeholder={t('apiKeyPlaceholder')}
                    />
                  </label>
                )}

                {!selected.oauth && (
                  <label className="block">
                    <div className="mb-1.5 text-xs font-medium text-tx2">模型覆盖（可选）</div>
                    <input
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="w-full rounded-xl border border-bd bg-white px-3 py-2.5 text-sm text-tx outline-none transition-all focus:border-ac focus:shadow-glow-ac"
                      placeholder={`留空自动发现，例如 ${DEFAULT_PROVIDER_MODELS[selected.name] || 'claude-sonnet-4-5'}`}
                    />
                    <div className="mt-1 text-xs text-tx3">
                      通常只填 Base URL 和 API Key 就够了；系统会先查询 /models 自动选择。只有中转站不开放模型列表，或你想固定某个模型时才填这里。
                    </div>
                  </label>
                )}

                <div className="flex items-center gap-3">
                  <button
                    type="submit"
                    disabled={saving}
                    className="rounded-full bg-gn px-5 py-2.5 text-sm font-semibold text-white shadow-glow-gn transition-all hover:bg-gn/90 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {saving ? t('saving') : t('saveSettings')}
                  </button>
                  <button
                    type="button"
                    disabled={testing}
                    onClick={handleTestProvider}
                    className="rounded-full border border-ac/30 bg-white px-5 py-2.5 text-sm font-semibold text-ac transition-all hover:border-ac hover:bg-ac/5 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {testing ? '测试中...' : '测试连接'}
                  </button>
                  <span className="text-xs text-tx3">{t('saveRedirectHint')}</span>
                </div>
              </div>
            </form>
          )}
        </div>
      </div>
    </SettingsPageFrame>
  )
}
