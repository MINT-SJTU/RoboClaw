import { useEffect, useMemo, useState } from 'react'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'
import type { TrainingCapabilities, TrainingPresetCapability, TrainingStatusData } from '@/domains/training/store/useTrainingStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { LossCurvePanel } from '@/domains/training/components/LossCurvePanel'
import { useI18n } from '@/i18n'

const TRAINING_LOCATIONS = ['current_machine', 'remote_backend'] as const
const REMOTE_PROVIDER_ORDER = ['aliyun', 'autodl'] as const
const ALIYUN_IMAGE_STORAGE_KEY = 'roboclaw.aliyunTrainingImage'
const POLICY_TYPES = [
  'act',
  'diffusion',
  'groot',
  'multi_task_dit',
  'pi0',
  'pi0_fast',
  'pi05',
  'reward_classifier',
  'sac',
  'sarm',
  'smolvla',
  'tdmpc',
  'vqbet',
  'wall_x',
  'xvla',
]

function loadAliyunImage() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(ALIYUN_IMAGE_STORAGE_KEY) || ''
}

function persistAliyunImage(value: string) {
  if (typeof window === 'undefined') return
  if (value.trim()) {
    window.localStorage.setItem(ALIYUN_IMAGE_STORAGE_KEY, value.trim())
  } else {
    window.localStorage.removeItem(ALIYUN_IMAGE_STORAGE_KEY)
  }
}

function parseStatusMessage(message: string) {
  const entries = message
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map((line) => {
      const idx = line.indexOf(':')
      if (idx < 0) return null
      return [line.slice(0, idx).trim(), line.slice(idx + 1).trim()] as const
    })
    .filter((entry): entry is readonly [string, string] => Boolean(entry))
  return Object.fromEntries(entries)
}

function statusTone(status: string) {
  const normalized = status.toLowerCase()
  if (normalized === 'running') return 'bg-gn/10 text-gn'
  if (normalized === 'queued' || normalized === 'pending') return 'bg-yl/10 text-yl'
  if (normalized === 'succeeded' || normalized === 'finished') return 'bg-ac/10 text-ac'
  if (normalized === 'failed' || normalized === 'missing') return 'bg-rd/10 text-rd'
  return 'bg-bd/20 text-tx2'
}

const FALLBACK_TRAINING_CAPABILITIES: TrainingCapabilities = {
  locations: {
    current_machine: { configured: true },
    remote_backend: {
      configured: false,
      mode: 'unavailable',
      notice: '',
    },
  },
  providers: {
    local: {
      id: 'local',
      display_name: 'Current machine',
      kind: 'current_machine' as const,
      configured: true,
      presets: [],
      supports_image_override: false,
      supports_resource_overrides: false,
    },
  },
}

function remoteModeCopy(
  mode: TrainingCapabilities['locations']['remote_backend']['mode'],
  t: (key: any) => string,
) {
  if (mode === 'managed') return t('managedRemoteTrainingMode')
  if (mode === 'self_hosted') return t('selfHostedRemoteTrainingMode')
  return t('remoteTrainingUnavailableMode')
}

function remoteModeHint(
  mode: TrainingCapabilities['locations']['remote_backend']['mode'],
  t: (key: any) => string,
) {
  if (mode === 'managed') return t('managedRemoteTrainingHint')
  if (mode === 'self_hosted') return t('selfHostedRemoteTrainingHint')
  return t('remoteTrainingBackendUnavailableHint')
}

function providerDetailSummary(providerData: TrainingStatusData['provider_data']) {
  if (!providerData || typeof providerData !== 'object') return ''
  for (const key of ['failure_reason', 'reason', 'error', 'detail']) {
    const value = providerData[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  const text = JSON.stringify(providerData)
  return text === '{}' ? '' : text
}

function failureReason(
  status: TrainingStatusData | null,
  parsed: Record<string, string>,
  rawMessage: string,
  t: (key: any) => string,
) {
  const structured = providerDetailSummary(status?.provider_data)
  if (structured) return structured

  const parsedProviderData = parsed.provider_data?.trim()
  if (parsedProviderData && parsedProviderData !== '{}') return parsedProviderData

  const message = (status?.message || parsed.message || '').trim()
  const genericMessages = new Set(['failed', 'pending', 'queued', 'running', 'finished', 'succeeded', 'stop_requested'])
  if (message && !genericMessages.has(message.toLowerCase())) return message

  if (rawMessage.trim()) return t('noFailureDetails')
  return ''
}

export default function TrainingCenterPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const session = useSessionStore((state) => state.session)
  const policies = useTrainingStore((state) => state.policies)
  const loadPolicies = useTrainingStore((state) => state.loadPolicies)
  const trainingCapabilities = useTrainingStore((state) => state.trainingCapabilities)
  const loadTrainingCapabilities = useTrainingStore((state) => state.loadTrainingCapabilities)
  const restoreCurrentTrainJob = useTrainingStore((state) => state.restoreCurrentTrainJob)
  const doTrainStart = useTrainingStore((state) => state.doTrainStart)
  const doTrainStop = useTrainingStore((state) => state.doTrainStop)
  const fetchTrainStatus = useTrainingStore((state) => state.fetchTrainStatus)
  const currentTrainJobId = useTrainingStore((state) => state.currentTrainJobId)
  const trainJobMessage = useTrainingStore((state) => state.trainJobMessage)
  const trainJobStatus = useTrainingStore((state) => state.trainJobStatus)
  const trainingLoading = useTrainingStore((state) => state.trainingLoading)
  const trainingStopLoading = useTrainingStore((state) => state.trainingStopLoading)
  const hubLoading = useHubTransferStore((state) => state.hubLoading)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushPolicy = useHubTransferStore((state) => state.pushPolicy)
  const pullPolicy = useHubTransferStore((state) => state.pullPolicy)
  const { t } = useI18n()
  const runtimeDatasets = datasets.filter((dataset) => dataset.capabilities.can_train && dataset.runtime)

  const [trainDataset, setTrainDataset] = useState('')
  const [trainingLocation, setTrainingLocation] = useState<(typeof TRAINING_LOCATIONS)[number]>('current_machine')
  const [remoteProvider, setRemoteProvider] = useState<'aliyun' | 'autodl'>('aliyun')
  const [policyType, setPolicyType] = useState('act')
  const [trainSteps, setTrainSteps] = useState(100000)
  const [trainStepsInput, setTrainStepsInput] = useState('100000')
  const [trainDevice, setTrainDevice] = useState('cuda')
  const [trainJobName, setTrainJobName] = useState('')
  const [aliyunPreset, setAliyunPreset] = useState('')
  const [aliyunImage, setAliyunImage] = useState(loadAliyunImage)
  const [aliyunGpuType, setAliyunGpuType] = useState('A10')
  const [aliyunGpuCount, setAliyunGpuCount] = useState(1)
  const [aliyunCpuCores, setAliyunCpuCores] = useState(16)
  const [aliyunMemoryGb, setAliyunMemoryGb] = useState(128)
  const [aliyunNodeCount, setAliyunNodeCount] = useState(1)
  const [showAliyunOverrides, setShowAliyunOverrides] = useState(false)
  const [pullPolicyRepo, setPullPolicyRepo] = useState('')
  const parsedTrainStatus = useMemo(() => parseStatusMessage(trainJobMessage), [trainJobMessage])
  const trainStatus = trainJobStatus
  const capabilities = trainingCapabilities ?? FALLBACK_TRAINING_CAPABILITIES
  const configuredRemoteProviders = useMemo(
    () =>
      REMOTE_PROVIDER_ORDER.filter((provider) => capabilities.providers[provider]?.configured),
    [capabilities],
  )
  const effectiveProvider = trainingLocation === 'current_machine' ? 'local' : remoteProvider
  const aliyunPresets = capabilities.providers.aliyun?.presets ?? []
  const selectedAliyunPreset =
    aliyunPresets.find((preset: TrainingPresetCapability) => preset.id === aliyunPreset) ?? aliyunPresets[0] ?? null
  const remoteBackendMode = capabilities.locations.remote_backend.mode
  const remoteBackendNotice = capabilities.locations.remote_backend.notice
  const trainJobId = String(trainStatus?.job_id || parsedTrainStatus.job_id || '').trim()
  const remoteJobId = String(trainStatus?.remote_job_id || parsedTrainStatus.remote_job_id || '').trim()
  const trainProvider = String(trainStatus?.provider || parsedTrainStatus.provider || '').trim()
  const trainStatusValue = (trainStatus?.status || parsedTrainStatus.status || '').trim()
  const trainStatusMessage = (trainStatus?.message || parsedTrainStatus.message || '').trim()
  const activeTrainJobId = String(currentTrainJobId || trainJobId).trim()
  const failureSummary = trainStatusValue.toLowerCase() === 'failed'
    ? failureReason(trainStatus, parsedTrainStatus, trainJobMessage, t)
    : ''
  const logTail = typeof trainStatus?.log_tail === 'string' ? trainStatus.log_tail.trim() : ''
  const showRemoteJobId = Boolean(remoteJobId && trainProvider !== 'local' && remoteJobId !== trainJobId)
  const showMessage = Boolean(
    trainStatusMessage
      && trainStatusMessage.toLowerCase() !== trainStatusValue.toLowerCase(),
  )
  const showDebugDetails = Boolean(
    trainJobMessage.trim()
      && (
        trainJobMessage.includes('\n')
        || Boolean(failureSummary)
        || Boolean(logTail)
      ),
  )
  const hasActiveTraining = Boolean(
    activeTrainJobId
    || trainStatus?.running
    || parsedTrainStatus.running === 'True',
  )

  useEffect(() => {
    void loadDatasets()
    void loadPolicies()
    void loadTrainingCapabilities()
    void restoreCurrentTrainJob()
  }, [loadDatasets, loadPolicies, loadTrainingCapabilities, restoreCurrentTrainJob])

  useEffect(() => {
    if (!activeTrainJobId) return
    void fetchTrainStatus(activeTrainJobId)
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') void fetchTrainStatus(activeTrainJobId)
    }, 15_000)
    return () => clearInterval(timer)
  }, [activeTrainJobId, fetchTrainStatus])

  useEffect(() => {
    persistAliyunImage(aliyunImage)
  }, [aliyunImage])

  useEffect(() => {
    if (!configuredRemoteProviders.length) {
      setTrainingLocation('current_machine')
      return
    }
    setTrainingLocation((current) => (current === 'current_machine' ? 'remote_backend' : current))
    if (!configuredRemoteProviders.includes(remoteProvider)) {
      setRemoteProvider(configuredRemoteProviders[0])
    }
  }, [configuredRemoteProviders, remoteProvider])

  useEffect(() => {
    if (trainingLocation === 'remote_backend') {
      setTrainDevice('cuda')
    }
  }, [trainingLocation, effectiveProvider])

  useEffect(() => {
    if (!aliyunPresets.length) {
      setAliyunPreset('')
      return
    }
    if (!aliyunPresets.some((preset: TrainingPresetCapability) => preset.id === aliyunPreset)) {
      setAliyunPreset(aliyunPresets[0].id)
    }
  }, [aliyunPreset, aliyunPresets])

  useEffect(() => {
    if (!selectedAliyunPreset) return
    setAliyunGpuType(selectedAliyunPreset.gpu_type)
    setAliyunGpuCount(selectedAliyunPreset.gpu_count)
    setAliyunCpuCores(selectedAliyunPreset.cpu_cores)
    setAliyunMemoryGb(selectedAliyunPreset.memory_gb)
    setAliyunNodeCount(selectedAliyunPreset.node_count)
  }, [selectedAliyunPreset])

  const resetAliyunOverrides = () => {
    if (!selectedAliyunPreset) return
    setAliyunImage('')
    setAliyunGpuType(selectedAliyunPreset.gpu_type)
    setAliyunGpuCount(selectedAliyunPreset.gpu_count)
    setAliyunCpuCores(selectedAliyunPreset.cpu_cores)
    setAliyunMemoryGb(selectedAliyunPreset.memory_gb)
    setAliyunNodeCount(selectedAliyunPreset.node_count)
  }

  const promptPushPolicy = (value: string) => {
    const repoId = prompt(t('enterRepoId'))
    if (!repoId) return
    void pushPolicy(value, repoId)
  }

  const handleStartTraining = () => {
    if (hasActiveTraining) return
    const confirmed = window.confirm(
      trainingLocation === 'current_machine'
        ? t('confirmCurrentMachineTraining')
        : t('confirmRemoteTraining'),
    )
    if (!confirmed) return
    void doTrainStart({
      dataset_name: trainDataset,
      policy_type: policyType,
      steps: trainSteps,
      device: trainDevice,
      provider: effectiveProvider,
      preset: effectiveProvider === 'aliyun' ? (selectedAliyunPreset?.backend_preset || '') : '',
      job_name: trainJobName.trim(),
      image: effectiveProvider === 'aliyun' ? aliyunImage.trim() : '',
      gpu_type: effectiveProvider === 'aliyun' ? aliyunGpuType.trim() : '',
      gpu_count: effectiveProvider === 'aliyun' ? aliyunGpuCount : undefined,
      cpu_cores: effectiveProvider === 'aliyun' ? aliyunCpuCores : undefined,
      memory_gb: effectiveProvider === 'aliyun' ? aliyunMemoryGb : undefined,
      node_count: effectiveProvider === 'aliyun' ? aliyunNodeCount : undefined,
    })
  }

  const handleStopTraining = () => {
    if (!activeTrainJobId || trainingStopLoading) return
    if (!window.confirm(t('confirmStopTraining'))) return
    void doTrainStop(activeTrainJobId)
  }

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('trainingCenter')}</h2>
      </div>

      <div className="flex-1 p-6 grid grid-cols-2 gap-6 items-start max-[1100px]:grid-cols-1">
        <div className="space-y-6">
          <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-yl">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('training')}</h3>
            <select
              value={trainDataset}
              onChange={(e) => setTrainDataset(e.target.value)}
              className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm mb-3
                focus:outline-none focus:border-ac"
            >
              <option value="">{t('selectDataset')}</option>
              {runtimeDatasets.map(d => (
                <option key={d.id} value={d.runtime!.name}>{d.label}</option>
              ))}
            </select>
            <div className={`grid gap-3 mb-2 ${trainingLocation === 'current_machine' ? 'grid-cols-2 max-[700px]:grid-cols-1' : 'grid-cols-2 max-[700px]:grid-cols-1'}`}>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                {t('trainingLocation')}
                <select
                  value={trainingLocation}
                  onChange={(e) => setTrainingLocation(e.target.value as (typeof TRAINING_LOCATIONS)[number])}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                >
                  <option value="current_machine">{t('currentMachineTraining')}</option>
                  <option value="remote_backend" disabled={!configuredRemoteProviders.length}>
                    {t('remoteTrainingBackend')}
                  </option>
                </select>
              </label>
              {trainingLocation === 'remote_backend' ? (
                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                  {t('remoteBackend')}
                  <select
                    value={remoteProvider}
                    onChange={(e) => setRemoteProvider(e.target.value as 'aliyun' | 'autodl')}
                    disabled={!configuredRemoteProviders.length}
                    className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac disabled:opacity-50"
                  >
                    {configuredRemoteProviders.length > 0 ? (
                      configuredRemoteProviders.map((provider) => (
                        <option key={provider} value={provider}>
                          {capabilities.providers[provider]?.display_name || provider}
                        </option>
                      ))
                    ) : (
                      <option value="">{t('remoteBackendUnavailable')}</option>
                    )}
                  </select>
                </label>
              ) : (
                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                  {t('device')}
                  <select
                    value={trainDevice}
                    onChange={(e) => setTrainDevice(e.target.value)}
                    className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                  >
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </select>
                </label>
              )}
            </div>
            <div className="mb-3 text-[11px] text-tx3">
              {trainingLocation === 'current_machine' ? (
                t('currentMachineTrainingHint')
              ) : (
                <div className="space-y-1">
                  <div className="text-tx2">
                    {t('deploymentMode')}: {remoteModeCopy(remoteBackendMode, t)}
                  </div>
                  <div>{remoteModeHint(remoteBackendMode, t)}</div>
                  {remoteBackendNotice && <div>{remoteBackendNotice}</div>}
                </div>
              )}
            </div>
            {hasActiveTraining && (
              <div className="mb-3 rounded-lg border border-yl/30 bg-yl/8 px-3 py-2 text-[11px] text-tx2">
                <div className="font-medium text-yl mb-1">{t('activeTrainingLabel')}</div>
                <div>{t('activeTrainingNotice')}</div>
              </div>
            )}
            <div className={`grid gap-3 mb-3 ${trainingLocation === 'remote_backend' && effectiveProvider === 'aliyun' ? 'grid-cols-3 max-[700px]:grid-cols-1' : 'grid-cols-2 max-[700px]:grid-cols-1'}`}>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('policyType')}
                <select
                  value={policyType}
                  onChange={(e) => setPolicyType(e.target.value)}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                >
                  {POLICY_TYPES.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('steps')}
                <input
                  type="number"
                  min={1}
                  value={trainStepsInput}
                  onChange={(e) => {
                    const next = e.target.value
                    setTrainStepsInput(next)
                    if (next === '') return
                    const parsed = Number(next)
                    if (Number.isFinite(parsed) && parsed >= 1) {
                      setTrainSteps(Math.floor(parsed))
                    }
                  }}
                  onBlur={() => {
                    const parsed = Number(trainStepsInput)
                    if (!Number.isFinite(parsed) || parsed < 1) {
                      setTrainSteps(100000)
                      setTrainStepsInput('100000')
                      return
                    }
                    const normalized = String(Math.floor(parsed))
                    setTrainSteps(Math.floor(parsed))
                    setTrainStepsInput(normalized)
                  }}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                />
              </label>
              {trainingLocation === 'remote_backend' && effectiveProvider === 'aliyun' && selectedAliyunPreset && (
                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                  {t('trainingPreset')}
                  <select
                    value={aliyunPreset}
                    onChange={(e) => setAliyunPreset(e.target.value)}
                    className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                  >
                    {aliyunPresets.map((preset: TrainingPresetCapability) => (
                      <option key={preset.id} value={preset.id}>{preset.label}</option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono mb-3">
              {t('jobName')}
              <input
                value={trainJobName}
                onChange={(e) => setTrainJobName(e.target.value)}
                placeholder={t('jobNamePlaceholder')}
                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
              />
            </label>
            {effectiveProvider === 'aliyun' && selectedAliyunPreset && (
              <div className="mb-3 rounded-lg border border-bd/70 bg-bg/50 p-3">
                <div className="bg-white/70 border border-bd/60 text-tx px-3 py-2 rounded-lg text-sm mb-2">
                  <div className="text-2xs text-tx3 font-mono mb-1">{t('presetSummary')}</div>
                  {selectedAliyunPreset.summary}
                </div>
                <div className="text-[11px] text-tx3">
                  {t('aliyunTrainingHint')}
                </div>
                <div className="mt-3 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setShowAliyunOverrides(value => !value)}
                    className="text-xs font-medium text-ac hover:text-ac2"
                  >
                    {showAliyunOverrides ? t('hideTrainingOverrides') : t('showTrainingOverrides')}
                  </button>
                  {showAliyunOverrides && (
                    <button
                      type="button"
                      onClick={resetAliyunOverrides}
                      className="text-xs font-medium text-tx3 hover:text-tx2"
                    >
                      {t('resetTrainingOverrides')}
                    </button>
                  )}
                </div>
                {showAliyunOverrides && (
                  <div className="grid grid-cols-2 gap-3 mt-3 max-[700px]:grid-cols-1">
                    <div className="col-span-2 rounded-md border border-bd/50 bg-white/60 px-3 py-2 text-[11px] text-tx3">
                      {t('trainingOverridesHint')}
                    </div>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono col-span-2">
                      {t('trainingImageOverride')}
                      <input
                        value={aliyunImage}
                        onChange={(e) => setAliyunImage(e.target.value)}
                        placeholder={t('trainingImagePlaceholder')}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {t('gpuType')}
                      <input
                        value={aliyunGpuType}
                        onChange={(e) => setAliyunGpuType(e.target.value)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {t('gpuCount')}
                      <input
                        type="number"
                        min={1}
                        value={aliyunGpuCount}
                        onChange={(e) => setAliyunGpuCount(Number(e.target.value) || 1)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {t('cpuCores')}
                      <input
                        type="number"
                        min={1}
                        value={aliyunCpuCores}
                        onChange={(e) => setAliyunCpuCores(Number(e.target.value) || 1)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {t('memoryGb')}
                      <input
                        type="number"
                        min={1}
                        value={aliyunMemoryGb}
                        onChange={(e) => setAliyunMemoryGb(Number(e.target.value) || 1)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono">
                      {t('nodeCount')}
                      <input
                        type="number"
                        min={1}
                        value={aliyunNodeCount}
                        onChange={(e) => setAliyunNodeCount(Number(e.target.value) || 1)}
                        className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac"
                      />
                    </label>
                  </div>
                )}
              </div>
            )}
            {effectiveProvider === 'autodl' && (
              <div className="mb-3 rounded-lg border border-bd/70 bg-bg/50 p-3 text-[11px] text-tx3">
                {t('autodlTrainingHint')}
              </div>
            )}
            <div className="flex gap-3 max-[520px]:flex-col">
              <button
                disabled={
                  hasActiveTraining
                  || (session.state !== 'idle' && session.state !== 'error')
                  || !trainDataset
                  || (trainingLocation === 'remote_backend' && !configuredRemoteProviders.length)
                  || !!trainingLoading
                }
                onClick={handleStartTraining}
                className="flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-ac hover:bg-ac2 shadow-glow-ac
                  transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed disabled:shadow-none"
              >
                {trainingLoading ? t('startingTraining') : t('startTraining')}
              </button>
              <button
                disabled={!activeTrainJobId || !!trainingStopLoading}
                onClick={handleStopTraining}
                className="px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-rd hover:bg-rd/90
                  transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed"
              >
                {trainingStopLoading ? t('stoppingTraining') : t('stopTraining')}
              </button>
            </div>
            {trainJobMessage && (
              <div className="mt-3 rounded-lg border border-bd/60 bg-bg p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs font-semibold text-tx">{t('trainJobStatus')}</div>
                  {trainStatusValue && (
                    <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${statusTone(trainStatusValue)}`}>
                      {trainStatusValue}
                    </span>
                  )}
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs max-[700px]:grid-cols-1">
                  {trainJobId && <StatusRow label="Job ID" value={trainJobId} />}
                  {showRemoteJobId && <StatusRow label="Remote Job ID" value={remoteJobId} />}
                  {trainProvider && <StatusRow label={t('trainingProvider')} value={trainProvider} />}
                  {showMessage && <StatusRow label="Message" value={trainStatusMessage} />}
                  {(trainStatus?.output_dir || parsedTrainStatus.output_dir) && <StatusRow label="Output" value={String(trainStatus?.output_dir || parsedTrainStatus.output_dir)} />}
                </div>
                {(failureSummary || logTail) && (
                  <div className="mt-3 space-y-2">
                    {failureSummary && (
                      <div className="rounded-lg border border-rd/20 bg-rd/5 px-3 py-2">
                        <div className="text-[11px] font-semibold text-rd mb-1">{t('failureReason')}</div>
                        <div className="text-xs text-tx2 whitespace-pre-wrap break-all">{failureSummary}</div>
                      </div>
                    )}
                    {logTail && (
                      <div className="rounded-lg border border-bd/60 bg-white/60 px-3 py-2">
                        <div className="text-[11px] font-semibold text-tx2 mb-1">{t('recentLogs')}</div>
                        <div className="text-xs text-tx2 font-mono whitespace-pre-wrap break-all">{logTail}</div>
                      </div>
                    )}
                  </div>
                )}
                {showDebugDetails && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs text-tx3 hover:text-tx2">{t('rawTrainingStatus')}</summary>
                    <div className="mt-2 text-xs text-tx2 font-mono break-all whitespace-pre-wrap">
                      {trainJobMessage}
                    </div>
                  </details>
                )}
              </div>
            )}
          </section>

          <LossCurvePanel />
        </div>

        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-gn">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('policies') || 'Policies'}</h3>
            <button
              onClick={() => { void loadPolicies() }}
              className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors"
            >
              {t('refresh')}
            </button>
          </div>

          {policies.length === 0 && (
            <div className="text-tx3 text-center py-4 text-sm">{t('noPolicies')}</div>
          )}
          <div className="space-y-1.5">
            {policies.map((p: any, i: number) => (
              <div key={i} className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-sm flex items-center gap-2">
                <span className="flex-1 font-mono text-tx2 truncate">
                  {typeof p === 'string' ? p : p.name || JSON.stringify(p)}
                </span>
                <button
                  disabled={!!hubLoading}
                  onClick={() => promptPushPolicy(typeof p === 'string' ? p : p.name)}
                  className="px-2 py-0.5 text-ac/60 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                >
                  {t('pushToHub')}
                </button>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-bd/40">
            <h4 className="text-xs font-bold text-tx3 uppercase mb-2">{t('downloadPolicy')}</h4>
            <div className="flex gap-2">
              <input
                placeholder={t('repoIdPlaceholder')}
                value={pullPolicyRepo}
                onChange={(e) => setPullPolicyRepo(e.target.value)}
                className="flex-1 bg-bg border border-bd text-tx px-3 py-1.5 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
              <button
                disabled={!pullPolicyRepo || !!hubLoading}
                onClick={() => {
                  void pullPolicy(pullPolicyRepo)
                  setPullPolicyRepo('')
                }}
                className="px-3 py-1.5 bg-ac/10 text-ac rounded-lg text-sm font-medium
                  hover:bg-ac/20 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                {hubLoading === 'pullPolicy' ? t('downloading') : t('download')}
              </button>
            </div>
          </div>

          {hubProgress && !hubProgress.done && hubLoading === 'pullPolicy' && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-2xs text-tx3 mb-1">
                <span>{hubProgress.operation}</span>
                <span>{hubProgress.progress_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-bd/30 rounded-full h-1.5">
                <div
                  className="bg-gn h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(hubProgress.progress_percent, 100)}%` }}
                />
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-bd/40 bg-white/60 px-2.5 py-2">
      <div className="text-[11px] uppercase tracking-wide text-tx3">{label}</div>
      <div className="mt-1 font-mono text-tx break-all">{value}</div>
    </div>
  )
}
