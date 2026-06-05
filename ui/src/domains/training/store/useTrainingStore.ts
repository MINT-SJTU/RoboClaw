import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'

const TRAIN = '/api/train'
const CLOUD_TRAIN = '/api/train/cloud'
const EVO_STUDIO_AGENT = '/api/evo-studio/agent-consult'
const POLICIES = '/api/policies'
const CURRENT_TRAIN_JOB_KEY = 'roboclaw.currentTrainJobId'
const CURRENT_TRAIN_MODE_KEY = 'roboclaw.currentTrainMode'
const CURRENT_TRAIN_USERNAME_KEY = 'roboclaw.currentTrainUsername'
const CURRENT_CLOUD_START_PARAMS_KEY = 'roboclaw.currentCloudStartParams'
const SUPPRESSED_CLOUD_JOB_KEY = 'roboclaw.suppressedCloudJobId'
const CLOUD_AUTOMATION_MODE_KEY = 'roboclaw.cloudAutomationMode.v2'
const TRAIN_JOB_HISTORY_KEY = 'roboclaw.trainJobHistory'

const supervisorWatchJobKeys = new Set<string>()
const runtimeRebindRestartJobKeys = new Set<string>()

function loadStoredTrainJobId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(CURRENT_TRAIN_JOB_KEY) || ''
}

function storeTrainJobId(jobId: string) {
  if (typeof window === 'undefined') return
  if (jobId) {
    window.localStorage.setItem(CURRENT_TRAIN_JOB_KEY, jobId)
  } else {
    window.localStorage.removeItem(CURRENT_TRAIN_JOB_KEY)
  }
}

function loadStoredTrainMode(): 'local' | 'cloud' {
  if (typeof window === 'undefined') return 'local'
  return window.localStorage.getItem(CURRENT_TRAIN_MODE_KEY) === 'cloud' ? 'cloud' : 'local'
}

function storeTrainMode(mode: 'local' | 'cloud') {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(CURRENT_TRAIN_MODE_KEY, mode)
}

function loadStoredTrainUsername() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(CURRENT_TRAIN_USERNAME_KEY) || ''
}

function storeTrainUsername(username: string) {
  if (typeof window === 'undefined') return
  if (username) {
    window.localStorage.setItem(CURRENT_TRAIN_USERNAME_KEY, username)
  } else {
    window.localStorage.removeItem(CURRENT_TRAIN_USERNAME_KEY)
  }
}

function loadStoredCloudStartParams(): Record<string, any> {
  if (typeof window === 'undefined') return {}
  const raw = window.localStorage.getItem(CURRENT_CLOUD_START_PARAMS_KEY)
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw)
    return isRecord(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function storeCloudStartParams(params: Record<string, any>) {
  if (typeof window === 'undefined') return
  if (Object.keys(params).length === 0) {
    window.localStorage.removeItem(CURRENT_CLOUD_START_PARAMS_KEY)
    return
  }
  window.localStorage.setItem(CURRENT_CLOUD_START_PARAMS_KEY, JSON.stringify(params))
}

function loadSuppressedCloudJobId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(SUPPRESSED_CLOUD_JOB_KEY) || ''
}

function storeSuppressedCloudJobId(jobId: string) {
  if (typeof window === 'undefined') return
  if (jobId) {
    window.localStorage.setItem(SUPPRESSED_CLOUD_JOB_KEY, jobId)
  } else {
    window.localStorage.removeItem(SUPPRESSED_CLOUD_JOB_KEY)
  }
}

export interface TrainJobHistoryItem {
  jobId: string
  mode: 'local' | 'cloud'
  username: string
  status: string
  message: string
  updatedAt: string
}

function normalizeTrainStatus(data: any): string {
  const rawStatus = String(data?.status || '').trim()
  const running = data?.running === true || String(data?.running || '').toLowerCase() === 'true'
  const message = dataMessage(data).toLowerCase()
  const errorText = `${data?.error || ''}\n${message}`.toLowerCase()
  const remediation = firstRecord(data?.failureRemediation, data?.failure_remediation)
  const hasFailureSignal = Boolean(data?.error || remediation.code)
  if (
    hasFailureSignal ||
    errorText.includes('ssh connection failed') ||
    errorText.includes('error reading ssh protocol banner') ||
    errorText.includes('configured ssh instance is not reachable') ||
    errorText.includes('unable to reach evo_train')
  ) {
    return 'failed'
  }
  if (running) return 'running'
  if (rawStatus) return rawStatus
  if (message.includes('running: true')) return 'running'
  if (message.includes('submitting') || message.includes('submitted')) return 'submitting'
  if (message.includes('failed') || message.includes('error:')) return 'failed'
  if (message.includes('stopped')) return 'stopped'
  if (message.includes('completed') || message.includes('success')) return 'completed'
  return 'unknown'
}

function isActiveTrainStatus(status: string): boolean {
  const text = status.toLowerCase()
  return ['running', 'submitting', 'submitted', 'pending', 'queued', 'starting', 'repairing'].some(item => text.includes(item))
}

function isCompletedTrainStatus(status: string, message = ''): boolean {
  const text = `${status}\n${message}`.toLowerCase()
  return (
    text.includes('succeeded') ||
    text.includes('success') ||
    text.includes('completed') ||
    text.includes('complete') ||
    text.includes('任务已成功完成') ||
    text.includes('__evo_stage_done__=collect_artifacts') ||
    text.includes('__evo_rlinf_metrics_captured__')
  )
}

function shouldKeepCloudFailure(status: string, message: string): boolean {
  const text = `${status}\n${message}`.toLowerCase()
  return (
    text.includes('missing') ||
    text.includes('failed') ||
    text.includes('create task failed') ||
    text.includes('error:') ||
    text.includes('ssh connection failed') ||
    text.includes('error reading ssh protocol banner') ||
    text.includes('configured ssh instance is not reachable') ||
    text.includes('unable to reach evo_train') ||
    text.includes('__evo_stage_failed__') ||
    text.includes('stage_failed') ||
    text.includes('terminated') ||
    text.includes('killed') ||
    text.includes('traceback') ||
    text.includes('syntaxerror') ||
    text.includes('no matching distribution') ||
    text.includes('requires a different python')
  )
}

function cloudFailureCode(data: any): string {
  const remediation = firstRecord(data?.failureRemediation, data?.failure_remediation)
  const supervisor = firstRecord(data?.supervisor)
  return String(remediation.code || supervisor.failureCode || '').trim().toUpperCase()
}

function isRuntimeBindingFailure(data: any): boolean {
  const code = cloudFailureCode(data)
  const text = dataMessage(data).toLowerCase()
  return (
    code === 'CLOUD_GPU_UNAVAILABLE' ||
    code === 'CLOUD_INSTANCE_UNREACHABLE' ||
    text.includes('__evo_gpu_unavailable__') ||
    text.includes('ssh connection failed') ||
    text.includes('error reading ssh protocol banner') ||
    text.includes('configured ssh instance is not reachable')
  )
}

async function bridgeReadyAfterRuntimeRebind(): Promise<boolean> {
  const existing = useTrainingStore.getState().cloudBridgeStatus
  if (existing?.enabled && existing.configurationReady === true) return true
  try {
    const bridge = await api(`${CLOUD_TRAIN}/bridge`) as CloudBridgeStatus
    useTrainingStore.setState({ cloudBridgeStatus: bridge })
    return Boolean(bridge.enabled && bridge.configurationReady === true)
  } catch {
    return false
  }
}

function supervisorRuntime(data: any): Record<string, any> {
  return firstRecord(firstRecord(data?.supervisor).runtime)
}

function payloadLooksLikeFailedCurrentJob(data: any, runtimeCurrentJobId: string): boolean {
  const payloadJobId = String(data?.job_id || data?.jobId || '').trim()
  if (!payloadJobId || !runtimeCurrentJobId || payloadJobId !== runtimeCurrentJobId) return false
  const rawStatus = String(data?.status || '').toLowerCase()
  const errorText = String(data?.error || data?.message || '').toLowerCase()
  return Boolean(
    cloudFailureCode(data) ||
    ['failed', 'missing'].some(item => rawStatus.includes(item)) ||
    errorText.includes('ssh connection failed') ||
    errorText.includes('error reading ssh protocol banner') ||
    errorText.includes('configured ssh instance is not reachable') ||
    errorText.includes('__evo_stage_failed__') ||
    errorText.includes('create task failed'),
  )
}

function supervisorCurrentJobId(data: any): string {
  const runtime = supervisorRuntime(data)
  const state = String(runtime.state || '').toLowerCase()
  const currentJobId = String(runtime.currentJobId || '').trim()
  if (!currentJobId) return ''
  if (payloadLooksLikeFailedCurrentJob(data, currentJobId)) return ''
  if (['watching', 'repairing', 'repair_submitted'].includes(state)) return currentJobId
  return ''
}

function supervisorRuntimeActive(data: any): boolean {
  const runtime = supervisorRuntime(data)
  const state = String(runtime.state || '').toLowerCase()
  const currentJobId = String(runtime.currentJobId || '').trim()
  if (payloadLooksLikeFailedCurrentJob(data, currentJobId)) return false
  const status = normalizeTrainStatus({
    status: runtime.status,
    running: runtime.running,
    message: runtime.message,
  })
  return ['watching', 'repairing', 'repair_submitted'].includes(state) || isActiveTrainStatus(status)
}

function cloudJobChainRoot(jobId: string): string {
  return String(jobId || '')
    .replace(/-intervention-.+$/i, '')
    .replace(/-repair-.+$/i, '')
    .replace(/-restart-.+$/i, '')
}

function sameCloudJobChain(a: string, b: string): boolean {
  const rootA = cloudJobChainRoot(a)
  const rootB = cloudJobChainRoot(b)
  return Boolean(rootA && rootB && rootA === rootB)
}

function existingCloudJobStillLooksActive(jobId: string, message: string): boolean {
  if (!jobId.trim()) return false
  return isActiveTrainStatus(normalizeTrainStatus({ message }))
}

function shouldIgnoreTransientCloudCurrent(
  payload: Record<string, any>,
  existingJobId: string,
  existingMessage: string,
): boolean {
  if (!existingCloudJobStillLooksActive(existingJobId, existingMessage)) return false
  const payloadJobId = String(payload.job_id || payload.jobId || '').trim()
  if (payloadJobId) return false
  const message = dataMessage(payload)
  const status = normalizeTrainStatus(payload)
  if (shouldKeepCloudFailure(status, message)) return false
  const lowered = message.toLowerCase()
  return (
    !message ||
    lowered.includes('任务启动后') ||
    lowered.includes('等待') ||
    lowered.includes('idle') ||
    lowered.includes('no current') ||
    lowered.includes('not started')
  )
}

async function resolveActiveCloudJob(username: string, candidateJobId: string): Promise<Record<string, any> | null> {
  const jobId = candidateJobId.trim()
  if (!jobId) return null
  try {
    const status = await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`)
    const statusRecord = firstRecord(status)
    const supervisorJobId = supervisorCurrentJobId(statusRecord)
    const statusText = normalizeTrainStatus(statusRecord)
    if (supervisorJobId && supervisorJobId !== jobId) {
      const supervisorStatus = await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(supervisorJobId)}?username=${encodeURIComponent(username)}`)
      const supervisorRecord = firstRecord(supervisorStatus)
      if (isActiveTrainStatus(normalizeTrainStatus(supervisorRecord)) || supervisorRuntimeActive(supervisorRecord)) {
        return { ...supervisorRecord, job_id: supervisorJobId }
      }
    }
    if (isActiveTrainStatus(statusText) || supervisorRuntimeActive(statusRecord)) {
      return { ...statusRecord, job_id: jobId }
    }
  } catch {
    return null
  }
  return null
}

function maybeStartBackendSupervisorWatch(username: string, jobId: string, status: string, statusData?: any): boolean {
  const automationPolicy = cloudAutomationPolicy(useTrainingStore.getState().cloudAutomationMode)
  const watchable = isActiveTrainStatus(status) || shouldKeepCloudFailure(status, dataMessage(statusData))
  if (!automationPolicy.autoRetrySameRuntime || !jobId || !watchable) return false
  const runtime = supervisorRuntime(statusData)
  const rootJobId = String(runtime.rootJobId || '').trim()
  if (rootJobId && rootJobId !== jobId) return true
  const key = `${username}:${jobId}:${automationPolicy.mode}`
  if (supervisorWatchJobKeys.has(key)) return true
  supervisorWatchJobKeys.add(key)
  void postJson(`${CLOUD_TRAIN}/supervisor/watch`, {
    username,
    jobId,
    automationPolicy,
  }).catch(() => {
    supervisorWatchJobKeys.delete(key)
  })
  return true
}

function loadStoredTrainJobHistory(): TrainJobHistoryItem[] {
  if (typeof window === 'undefined') return []
  const raw = window.localStorage.getItem(TRAIN_JOB_HISTORY_KEY)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter(item => isRecord(item) && typeof item.jobId === 'string')
      .slice(0, 20) as TrainJobHistoryItem[]
  } catch {
    return []
  }
}

function storeTrainJobHistory(history: TrainJobHistoryItem[]) {
  if (typeof window === 'undefined') return
  if (history.length === 0) {
    window.localStorage.removeItem(TRAIN_JOB_HISTORY_KEY)
    return
  }
  window.localStorage.setItem(TRAIN_JOB_HISTORY_KEY, JSON.stringify(history.slice(0, 20)))
}

function upsertTrainJobHistory(history: TrainJobHistoryItem[], item: TrainJobHistoryItem): TrainJobHistoryItem[] {
  const next = [
    item,
    ...history.filter(existing => existing.jobId !== item.jobId),
  ].slice(0, 20)
  storeTrainJobHistory(next)
  return next
}

function makeTrainJobHistoryItem(
  jobId: string,
  mode: 'local' | 'cloud',
  username: string,
  data: any,
): TrainJobHistoryItem {
  return {
    jobId,
    mode,
    username,
    status: normalizeTrainStatus(data),
    message: dataMessage(data),
    updatedAt: new Date().toISOString(),
  }
}

export type CloudAutomationMode = 'ask' | 'safe_auto' | 'full_auto'

function normalizeCloudAutomationMode(value: unknown): CloudAutomationMode {
  const mode = String(value || '').trim()
  return mode === 'safe_auto' || mode === 'full_auto' ? mode : 'ask'
}

function loadStoredCloudAutomationMode(): CloudAutomationMode {
  if (typeof window === 'undefined') return 'full_auto'
  return normalizeCloudAutomationMode(window.localStorage.getItem(CLOUD_AUTOMATION_MODE_KEY) || 'full_auto')
}

function storeCloudAutomationMode(mode: CloudAutomationMode) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(CLOUD_AUTOMATION_MODE_KEY, mode)
}

export function cloudAutomationPolicy(mode: CloudAutomationMode) {
  const autoRepair = mode === 'safe_auto' || mode === 'full_auto'
  return {
    mode,
    autoInspectLogs: autoRepair,
    autoRepairPlan: autoRepair,
    autoRetrySameRuntime: autoRepair,
    allowAgentRepairSameRuntime: mode === 'full_auto',
    paidStartRequiresConfirmation: mode === 'ask',
    allowRuntimeChangeWithoutConfirmation: false,
    allowSecretEditingInChat: false,
  }
}

function activeUsername(username?: string): string {
  if (typeof window === 'undefined') return username || ''
  return (username || window.localStorage.getItem('roboclaw.dataset.username') || 'pearl').trim()
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function firstRecord(...values: unknown[]): Record<string, any> {
  for (const value of values) {
    if (isRecord(value)) return value
  }
  return {}
}

function usesExistingRuntime(...values: unknown[]): boolean {
  for (const value of values) {
    const item = firstRecord(value)
    const runtime = firstRecord(item.runtime, item.trainingContract?.runtime)
    const mode = String(
      item.runtimeMode
        || item.mode
        || item.provider
        || runtime.runtimeMode
        || runtime.mode
        || '',
    ).toLowerCase()
    if (
      item.useExistingInstance === true
      || item.existingInstance === true
      || item.configurationReady === true && String(item.deploymentMode || item.mode || '').toLowerCase() === 'ssh'
      || String(item.deploymentMode || '').toLowerCase() === 'ssh'
      || runtime.existingInstance === true
      || mode === 'ssh'
      || mode === 'ssh_existing_instance'
      || mode === 'existing_ssh'
    ) {
      return true
    }
  }
  return false
}

function normalizeAgentConsultPlan(data: Record<string, any>): Record<string, unknown> {
  const vlaPlan = firstRecord(data.vlaPlan, data.plan)
  const runtimeMatch = firstRecord(data.runtimeMatch)
  const configuration = firstRecord(data.configuration)
  const provider = String(data.provider || firstRecord(data.bridge).provider || '').trim()
  const normalizedPlan = {
    ...vlaPlan,
    workflow: String(vlaPlan.workflow || data.workflow || ''),
    params: firstRecord(vlaPlan.params, data.params),
    readyToStart: Boolean(vlaPlan.readyToStart || data.readyForConfirmation),
    runtimeMode: String(data.runtimeMode || runtimeMatch.runtimeMode || configuration.mode || configuration.deploymentMode || ''),
  }
  return {
    ...data,
    provider,
    vlaPlan: normalizedPlan,
    plan: normalizedPlan,
    runtimeMatch,
    runtimeMode: String(data.runtimeMode || runtimeMatch.runtimeMode || configuration.mode || configuration.deploymentMode || ''),
  }
}

function agentConsultPlanMessage(data: Record<string, any>): string {
  const vlaPlan = firstRecord(data.vlaPlan, data.plan)
  const runtimeMatch = firstRecord(data.runtimeMatch)
  const planner = firstRecord(vlaPlan.planner, data.aiPlan)
  const plannerSource = String(planner.source || '').trim()
  const plannerModel = String(planner.providerModel || '').trim()
  if (plannerSource !== 'llm') {
    const failureLabel = plannerSource === 'llm_unconfigured'
      ? '未接入大模型'
      : plannerSource === 'llm_timeout'
        ? '大模型规划超时'
        : plannerSource === 'llm_error'
          ? '大模型调用失败'
          : plannerSource === 'llm_parse_error'
            ? '大模型输出无法解析'
            : plannerSource
              ? `大模型未完成规划（${plannerSource}）`
              : '未接入大模型'
    return `${failureLabel}；仅完成基础检查，未生成可启动方案`
  }
  const plannerPrefix = `接入的大模型已理解指令${plannerModel ? ` · ${plannerModel}` : ''}`
  const runtimeMode = String(data.runtimeMode || runtimeMatch.runtimeMode || vlaPlan.runtimeMode || '').trim()
  const usingExistingSsh = runtimeMatch.skipped === true || runtimeMode === 'ssh_existing_instance'
  if (usingExistingSsh && runtimeMatch.readyToStart) {
    return data.readyForConfirmation || vlaPlan.readyToStart
      ? `${plannerPrefix}；可启动`
      : `${plannerPrefix}；已填入参数`
  }
  return `${plannerPrefix}：${data?.readyForConfirmation ? '可启动' : (vlaPlan?.readyToStart ? '待确认资源' : '需补参数')}`
}

export interface Policy {
  name: string
  checkpoint: string
  dataset?: string
  steps?: number
}

export interface TrainingCurvePoint {
  step: string
  ep: number
  epoch: number
  loss: number
}

export interface TrainingCurve {
  job_id: string
  log_path: string
  exists: boolean
  points: TrainingCurvePoint[]
  last_epoch: number | null
  last_loss: number | null
  best_ep: number | null
  best_loss: number | null
  updated_at: number | null
}

export interface CloudBridgeStatus {
  enabled: boolean
  provider: string
  managedBy: string
  userActionRequired: boolean
  message: string
  operatorHint: string
  missingDeploymentFields: string[]
  deploymentMode?: 'managed' | 'ssh' | string
  configurationReady?: boolean
  prepareReady?: boolean
  gpuReady?: boolean
  sshGpuReady?: boolean
  sshConnectionReady?: boolean
  sshGpu?: string
  runtimeEndpoint?: string
  runtimeHost?: string
  runtimePort?: string
  runtimeUser?: string
  configurationWarnings?: string[]
  resourceCatalog?: {
    skuCount?: number | string
    readySkuCount?: number | string
    imageCount?: number | string
    readyImageCount?: number | string
  }
}

export interface CloudGpuSku {
  skuId: string
  displayName?: string
  provider?: string
  gpuSpec?: string
  gpuCount?: string
  hourlyPriceCents?: string
  costHourlyCents?: string
  stockStatus?: 'available' | 'sold_out' | 'unknown' | string
  stockCount?: string
  availableGpuCount?: string
  maxGpuCountPerInstance?: string
  readyToStart?: boolean
  enabled?: boolean
  [key: string]: unknown
}

export interface CloudRuntimeImage {
  imageId: string
  displayName?: string
  provider?: string
  cudaVFrom?: string
  readyToStart?: boolean
  enabled?: boolean
  [key: string]: unknown
}

export interface CloudResourceCatalog {
  provider: string
  skus: CloudGpuSku[]
  images: CloudRuntimeImage[]
  messages?: Record<string, string>
}

export interface CloudRuntimeMatch {
  provider: string
  compatible?: boolean
  readyToStart?: boolean
  matches?: Array<{
    skuId?: string
    imageId?: string
    compatible?: boolean
    readyToStart?: boolean
    score?: number
    reasons?: string[]
    blocking?: string[]
    [key: string]: unknown
  }>
  requirements?: Record<string, unknown>
  message?: string
  [key: string]: unknown
}

export interface CloudSourcePreflight {
  message?: string
  source?: {
    uri?: string
    kind?: string
    strategy?: string
    resolvedPath?: string
    sizeKnown?: boolean
    estimatedBytes?: number | null
    estimatedSize?: string
    requiresUserConfirmation?: boolean
    risks?: string[]
    warnings?: string[]
    missingFields?: string[]
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface AuthConnection {
  id: string
  kind: 'data' | 'model' | 'both'
  provider: string
  label: string
  scope?: string
  sourcePrefixes?: string[]
  configured: boolean
  requiresSecrets?: boolean
}

export interface AuthConnectionInput {
  username: string
  id: string
  kind: 'data' | 'model' | 'both'
  provider: string
  label?: string
  scope?: string
  visibility?: 'user' | 'team'
  sourcePrefixes?: string[]
  secrets: Record<string, string>
}

export interface CloudDatasetSource {
  sourceType:
    | 'platform_dataset'
    | 'public_reference'
    | 'user_object_storage'
    | 'builtin_benchmark'
  datasetId?: string
  uri?: string
  authRef?: string
  format?: string
  benchmark?: string
  suite?: string
  userConfirmed?: boolean
  confirmedAt?: string
  confirmedRisks?: string[]
}

export interface CloudModelSource {
  sourceType:
    | 'ai_resolved'
    | 'builtin_policy'
    | 'evo_studio_checkpoint'
    | 'public_model_repo'
    | 'user_object_storage'
    | 'from_scratch'
  modelFamily?: string
  checkpoint?: string
  uri?: string
  authRef?: string
  format?: string
}

export interface CloudTrainingStartParams {
  username: string
  provider?: string
  dataset_name?: string
  workflow: string
  policy_type?: string
  steps?: number
  device?: string
  task_name?: string
  params: {
    datasetSource: CloudDatasetSource
    modelSource?: CloudModelSource
    policyType?: string
    steps?: number
    device?: string
    [key: string]: unknown
  }
  sku_id?: string
  image_id?: string
  hourly_cost_cents?: number
  waitForSubmit?: boolean
  wait_for_submit?: boolean
}

interface TrainingStore {
  policies: Policy[]
  trainJobMessage: string
  trainPlanMessage: string
  trainPlan: Record<string, unknown> | null
  cloudBridgeStatus: CloudBridgeStatus | null
  cloudResourceCatalog: CloudResourceCatalog | null
  cloudRuntimeMatch: CloudRuntimeMatch | null
  cloudSourcePreflight: CloudSourcePreflight | null
  authConnections: AuthConnection[]
  currentTrainJobId: string
  currentTrainMode: 'local' | 'cloud'
  currentTrainUsername: string
  restartableCloudJobId: string
  trainJobHistory: TrainJobHistoryItem[]
  cloudAutomationMode: CloudAutomationMode
  trainCurve: TrainingCurve | null
  trainingLoading: boolean
  trainingPlanLoading: boolean
  runtimeMatchLoading: boolean
  sourcePreflightLoading: boolean
  trainingStopLoading: boolean
  _runtimeMatchAbort: AbortController | null
  _sourcePreflightAbort: AbortController | null
  _planAbort: AbortController | null
  _startAbort: AbortController | null
  _statusAbort: AbortController | null
  loadPolicies: () => Promise<void>
  loadCloudBridgeStatus: () => Promise<void>
  loadCloudResources: (provider?: string, forceRefresh?: boolean) => Promise<void>
  loadRuntimeMatch: (params: {
    username?: string
    provider?: string
    params?: Record<string, unknown>
    sku_id?: string
    image_id?: string
    force_refresh?: boolean
  }) => Promise<CloudRuntimeMatch | null>
  loadSourcePreflight: (params: {
    username?: string
    provider?: string
    role?: string
    source?: Record<string, unknown>
  }) => Promise<CloudSourcePreflight | null>
  loadAuthConnections: (username?: string) => Promise<void>
  saveAuthConnection: (connection: AuthConnectionInput) => Promise<AuthConnection | null>
  restoreCurrentTrainJob: () => Promise<void>
  doTrainStart: (params: { dataset_name: string; steps?: number; device?: string; policy_type?: string }) => Promise<void>
  doCloudTrainPlan: (params: {
    username: string
    message: string
    workflow?: string
    params: Record<string, unknown>
    provider?: string
    sku_id?: string
    image_id?: string
  }) => Promise<Record<string, unknown> | null>
  applyAgentConsultPlan: (data: Record<string, unknown>) => void
  clearCloudTrainPlan: () => void
  detachCurrentTrainJob: (reason?: string) => void
  cancelCloudChecks: () => void
  cancelCloudStartRequest: () => void
  doCloudTrainStart: (params: CloudTrainingStartParams) => Promise<void>
  handleTrainingWebSocketEvent: (payload: Record<string, unknown>) => void
  repairCloudTraining: (params?: {
    username?: string
    job_id?: string
    task?: string
    automation_mode?: CloudAutomationMode
    user_guidance?: string
    userGuidance?: string
  }) => Promise<Record<string, unknown> | null>
  setCloudAutomationMode: (mode: CloudAutomationMode) => void
  doTrainStop: () => Promise<void>
  fetchTrainStatus: (jobId: string) => Promise<void>
  fetchTrainCurve: (jobId: string) => Promise<void>
  clearTrainCurve: () => void
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  policies: [],
  trainJobMessage: '',
  trainPlanMessage: '',
  trainPlan: null,
  cloudBridgeStatus: null,
  cloudResourceCatalog: null,
  cloudRuntimeMatch: null,
  cloudSourcePreflight: null,
  authConnections: [],
  currentTrainJobId: loadStoredTrainJobId(),
  currentTrainMode: loadStoredTrainMode(),
  currentTrainUsername: loadStoredTrainUsername(),
  restartableCloudJobId: '',
  trainJobHistory: loadStoredTrainJobHistory(),
  cloudAutomationMode: loadStoredCloudAutomationMode(),
  trainCurve: null,
  trainingLoading: false,
  trainingPlanLoading: false,
  runtimeMatchLoading: false,
  sourcePreflightLoading: false,
  trainingStopLoading: false,
  _runtimeMatchAbort: null,
  _sourcePreflightAbort: null,
  _planAbort: null,
  _startAbort: null,
  _statusAbort: null,

  loadPolicies: async () => {
    const response = await api(`${POLICIES}`)
    set({ policies: Array.isArray(response) ? response : response.policies || [] })
  },

  loadCloudBridgeStatus: async () => {
    try {
      const cloudBridgeStatus = await api(`${CLOUD_TRAIN}/bridge`) as CloudBridgeStatus
      set({ cloudBridgeStatus })
    } catch (error) {
      set({
        cloudBridgeStatus: {
          enabled: false,
          provider: '',
          managedBy: 'Evo Studio deployment',
          userActionRequired: false,
          message: error instanceof Error ? error.message : 'Cloud training bridge status is unavailable.',
          operatorHint: 'The backend does not expose cloud training bridge status.',
          missingDeploymentFields: [],
        },
      })
    }
  },

  loadCloudResources: async (provider = '', forceRefresh = true) => {
    try {
      const params = new URLSearchParams()
      if (provider) params.set('provider', provider)
      params.set('include_incomplete', 'true')
      if (forceRefresh) params.set('force_refresh', 'true')
      const query = `?${params.toString()}`
      const cloudResourceCatalog = await api(`${CLOUD_TRAIN}/resources${query}`) as CloudResourceCatalog
      set({ cloudResourceCatalog })
    } catch {
      set({ cloudResourceCatalog: { provider, skus: [], images: [] } })
    }
  },

  loadRuntimeMatch: async (params) => {
    get()._runtimeMatchAbort?.abort()
    const controller = new AbortController()
    set({ _runtimeMatchAbort: controller, runtimeMatchLoading: true })
    try {
      const cloudRuntimeMatch = await postJson(`${TRAIN}/runtime-match`, {
        username: activeUsername(params.username),
        provider: params.provider || '',
        params: params.params || {},
        sku_id: params.sku_id || '',
        image_id: params.image_id || '',
        forceRefresh: params.force_refresh ?? true,
      }, { signal: controller.signal }) as CloudRuntimeMatch
      if (get()._runtimeMatchAbort === controller) set({ cloudRuntimeMatch })
      return cloudRuntimeMatch
    } catch (error) {
      if (isAbortError(error)) return null
      if (get()._runtimeMatchAbort === controller) set({ cloudRuntimeMatch: null })
      return null
    } finally {
      if (get()._runtimeMatchAbort === controller) {
        set({ _runtimeMatchAbort: null, runtimeMatchLoading: false })
      }
    }
  },

  loadSourcePreflight: async (params) => {
    get()._sourcePreflightAbort?.abort()
    const controller = new AbortController()
    set({ _sourcePreflightAbort: controller, sourcePreflightLoading: true })
    try {
      const cloudSourcePreflight = await postJson(`${TRAIN}/source-preflight`, {
        username: activeUsername(params.username),
        provider: params.provider || '',
        role: params.role || 'dataset',
        source: params.source || {},
      }, { signal: controller.signal }) as CloudSourcePreflight
      if (get()._sourcePreflightAbort === controller) set({ cloudSourcePreflight })
      return cloudSourcePreflight
    } catch (error) {
      if (isAbortError(error)) return null
      if (get()._sourcePreflightAbort === controller) set({ cloudSourcePreflight: null })
      return null
    } finally {
      if (get()._sourcePreflightAbort === controller) {
        set({ _sourcePreflightAbort: null, sourcePreflightLoading: false })
      }
    }
  },

  loadAuthConnections: async (username = '') => {
    const active = activeUsername(username)
    try {
      const response = await api(`${CLOUD_TRAIN}/auth-connections?username=${encodeURIComponent(active)}`) as { connections?: AuthConnection[] }
      set({ authConnections: Array.isArray(response.connections) ? response.connections : [] })
    } catch {
      set({ authConnections: [] })
    }
  },

  saveAuthConnection: async (connection) => {
    try {
      const data = await postJson(`${CLOUD_TRAIN}/auth-connections`, {
        ...connection,
        username: activeUsername(connection.username),
      }) as { connection?: AuthConnection }
      await useTrainingStore.getState().loadAuthConnections(connection.username)
      return data.connection || null
    } catch (error) {
      set({ trainPlanMessage: error instanceof Error ? `保存私有连接失败：${error.message}` : '保存私有连接失败' })
      return null
    }
  },

  restoreCurrentTrainJob: async () => {
    const storedJobId = loadStoredTrainJobId()
    const storedMode = loadStoredTrainMode()
    const storedUsername = loadStoredTrainUsername()
    if (storedJobId) {
      try {
        const statusUrl = storedMode === 'cloud'
          ? `${CLOUD_TRAIN}/status/${encodeURIComponent(storedJobId)}?username=${encodeURIComponent(storedUsername)}`
          : `${TRAIN}/status/${encodeURIComponent(storedJobId)}`
        const status = await api(statusUrl)
        const message = dataMessage(status)
        const trainStatus = normalizeTrainStatus(status)
        if (storedMode === 'cloud') {
          maybeStartBackendSupervisorWatch(storedUsername, storedJobId, trainStatus, status)
        }
        const supervisorJobId = storedMode === 'cloud' ? supervisorCurrentJobId(status) : ''
        if (supervisorJobId && supervisorJobId !== storedJobId) {
          storeTrainJobId(supervisorJobId)
          storeTrainMode('cloud')
          storeTrainUsername(storedUsername)
          set({
            currentTrainJobId: supervisorJobId,
            currentTrainMode: 'cloud',
            currentTrainUsername: storedUsername,
            trainJobMessage: message,
          })
          return
        }
        if (
          storedMode === 'cloud' &&
          !isActiveTrainStatus(trainStatus) &&
          shouldKeepCloudFailure(trainStatus, message) &&
          isRuntimeBindingFailure(status) &&
          await bridgeReadyAfterRuntimeRebind()
        ) {
          const historyItem = makeTrainJobHistoryItem(storedJobId, storedMode, storedUsername, status)
          const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
          storeTrainJobId('')
          storeTrainUsername('')
          set({
            trainJobHistory,
            trainJobMessage: '已连接新的云端实例，旧实例失败任务已移到历史。',
            currentTrainJobId: '',
            currentTrainUsername: '',
          })
          return
        }
        if (
          isActiveTrainStatus(trainStatus) ||
          (storedMode === 'cloud' && (shouldKeepCloudFailure(trainStatus, message) || isCompletedTrainStatus(trainStatus, message)))
        ) {
          set({
            currentTrainJobId: storedJobId,
            currentTrainMode: storedMode,
            currentTrainUsername: storedUsername,
            trainJobMessage: message,
          })
          return
        }
        const historyItem = makeTrainJobHistoryItem(storedJobId, storedMode, storedUsername, status)
        const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
        storeTrainJobId('')
        storeTrainUsername('')
        set({ trainJobHistory, trainJobMessage: message, currentTrainJobId: '', currentTrainUsername: '' })
      } catch {
        storeTrainJobId('')
      }
    }

    const current = await api(`${TRAIN}/current`)
    const jobId = typeof current.job_id === 'string' ? current.job_id : ''
    if (jobId && current.running) {
      storeTrainJobId(jobId)
      storeTrainMode('local')
      set({ currentTrainJobId: jobId, currentTrainMode: 'local', currentTrainUsername: '', trainJobMessage: statusMessage(current) })
      return
    }

    try {
      const username = activeUsername(storedUsername)
      const cloudCurrent = await api(`${CLOUD_TRAIN}/current?username=${encodeURIComponent(username)}`)
      const cloudJobId = typeof cloudCurrent.job_id === 'string' ? cloudCurrent.job_id : ''
      if (cloudJobId) {
        const cloudStatus = normalizeTrainStatus(cloudCurrent)
        const suppressedCloudJobId = loadSuppressedCloudJobId()
        if (suppressedCloudJobId === cloudJobId && !isActiveTrainStatus(cloudStatus)) {
          const historyItem = makeTrainJobHistoryItem(cloudJobId, 'cloud', username, cloudCurrent)
          const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
          storeTrainJobId('')
          storeTrainUsername('')
          set({
            trainJobHistory,
            currentTrainJobId: '',
            currentTrainUsername: '',
            trainJobMessage: '已重新绑定云端实例，旧任务已归档。',
          })
          return
        }
        if (!isActiveTrainStatus(cloudStatus)) {
          const historyItem = makeTrainJobHistoryItem(cloudJobId, 'cloud', username, cloudCurrent)
          const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
          const message = dataMessage(cloudCurrent)
          if (shouldKeepCloudFailure(cloudStatus, message) && isRuntimeBindingFailure(cloudCurrent) && await bridgeReadyAfterRuntimeRebind()) {
            storeTrainJobId('')
            storeTrainUsername('')
            set({
              trainJobHistory,
              currentTrainJobId: '',
              currentTrainUsername: '',
              currentTrainMode: 'cloud',
              trainJobMessage: '已连接新的云端实例，旧实例失败任务已移到历史。',
            })
            return
          }
          if (shouldKeepCloudFailure(cloudStatus, message)) {
            storeTrainJobId(cloudJobId)
            storeTrainMode('cloud')
            storeTrainUsername(username)
            set({
              trainJobHistory,
              currentTrainJobId: cloudJobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
              trainJobMessage: message,
            })
            return
          }
          if (isCompletedTrainStatus(cloudStatus, message)) {
            storeTrainJobId(cloudJobId)
            storeTrainMode('cloud')
            storeTrainUsername(username)
            set({
              trainJobHistory,
              currentTrainJobId: cloudJobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
              trainJobMessage: message,
            })
            return
          }
          storeTrainJobId('')
          storeTrainUsername('')
          set({ trainJobHistory, currentTrainJobId: '', currentTrainUsername: '', trainJobMessage: message })
          return
        }
        storeTrainJobId(cloudJobId)
        storeTrainMode('cloud')
        storeTrainUsername(username)
        set({
          currentTrainJobId: cloudJobId,
          currentTrainMode: 'cloud',
          currentTrainUsername: username,
          trainJobMessage: dataMessage(cloudCurrent),
        })
        return
      }
      const currentMessage = dataMessage(cloudCurrent)
      if (currentMessage || cloudCurrent.staleAfterRuntimeRebind || cloudCurrent.archivedPreviousJob) {
        const activeCandidate = get().currentTrainJobId || loadStoredTrainJobId()
        const activeCloudJob = await resolveActiveCloudJob(username, activeCandidate)
        if (activeCloudJob) {
          const activeJobId = String(activeCloudJob.job_id || activeCloudJob.jobId || activeCandidate).trim()
          if (activeJobId) {
            storeTrainJobId(activeJobId)
            storeTrainMode('cloud')
            storeTrainUsername(username)
            set({
              currentTrainJobId: activeJobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
              restartableCloudJobId: '',
              trainJobMessage: dataMessage(activeCloudJob),
            })
            return
          }
        }
        if (shouldIgnoreTransientCloudCurrent(cloudCurrent, activeCandidate, get().trainJobMessage)) {
          return
        }
        const archived = firstRecord(cloudCurrent.archivedPreviousJob)
        const restartableCloudJobId = String(archived.job_id || archived.jobId || '').trim()
        const automationPolicy = cloudAutomationPolicy(get().cloudAutomationMode)
        storeTrainJobId('')
        storeTrainUsername('')
        set({
          currentTrainJobId: '',
          currentTrainMode: 'cloud',
          currentTrainUsername: '',
          restartableCloudJobId,
          trainJobMessage: currentMessage,
        })
        if (restartableCloudJobId && automationPolicy.autoRetrySameRuntime) {
          const restartKey = `${username}:${restartableCloudJobId}:${automationPolicy.mode}`
          if (!runtimeRebindRestartJobKeys.has(restartKey)) {
            runtimeRebindRestartJobKeys.add(restartKey)
            set({ trainJobMessage: '云端实例已恢复，正在自动复用上次任务参数继续运行...' })
            void useTrainingStore.getState().repairCloudTraining({
              username,
              job_id: restartableCloudJobId,
              automation_mode: automationPolicy.mode,
              user_guidance: '当前实例已重新绑定并恢复可用；在同一实例、同一缓存、同一预算内自动继续，不要重新从零开始。',
            }).catch(() => {
              runtimeRebindRestartJobKeys.delete(restartKey)
            })
          }
        }
        return
      }
    } catch {
      // The local training page should still render when the cloud bridge is down.
    }

    storeTrainJobId('')
    set({ currentTrainJobId: '', restartableCloudJobId: '' })
  },

  doTrainStart: async (params) => {
    set({ trainingLoading: true })
    try {
      const data = await postJson(`${TRAIN}/start`, params)
      const jobId = typeof data.job_id === 'string' ? data.job_id : ''
      storeTrainJobId(jobId)
      storeTrainMode('local')
      storeTrainUsername('')
      set({ trainJobMessage: data.message || '', currentTrainJobId: jobId, currentTrainMode: 'local', currentTrainUsername: '' })
    } catch (error) {
      set({ trainJobMessage: error instanceof Error ? `本地训练启动失败：${error.message}` : '本地训练启动失败' })
      throw error
    } finally {
      set({ trainingLoading: false })
    }
  },

  doCloudTrainPlan: async (params) => {
    const username = activeUsername(params.username)
    const automationPolicy = cloudAutomationPolicy(useTrainingStore.getState().cloudAutomationMode)
    get()._planAbort?.abort()
    const controller = new AbortController()
    set({ _planAbort: controller, trainingPlanLoading: true, trainPlanMessage: '正在配置并检查...', trainPlan: null })
    try {
      const data = await postJson(EVO_STUDIO_AGENT, {
        task: params.message,
        mode: 'plan',
        username,
        workflow: params.workflow || '',
        params: params.params,
        automation_policy: automationPolicy,
        context: {
          page: 'training',
          username,
          provider: params.provider || '',
          workflow: params.workflow || '',
          params: params.params,
          automationPolicy,
          sku_id: params.sku_id || '',
          image_id: params.image_id || '',
        },
        provider: params.provider || '',
        sku_id: params.sku_id || '',
        image_id: params.image_id || '',
      }, { signal: controller.signal })
      const normalizedData = normalizeAgentConsultPlan(data)
      if (get()._planAbort === controller) set({
        trainPlan: normalizedData,
        trainPlanMessage: agentConsultPlanMessage(normalizedData),
      })
      return normalizedData
    } catch (error) {
      if (isAbortError(error)) {
        if (get()._planAbort === controller) set({ trainPlanMessage: '已取消检查。' })
        return null
      }
      const message = error instanceof Error ? error.message : 'AI 配置训练失败'
      if (get()._planAbort === controller) set({
        trainPlan: null,
        trainPlanMessage: `AI 配置失败：${message}`,
      })
      return null
    } finally {
      if (get()._planAbort === controller) {
        set({ _planAbort: null, trainingPlanLoading: false })
      }
    }
  },

  applyAgentConsultPlan: (data) => {
    if (!isRecord(data)) return
    const normalizedData = normalizeAgentConsultPlan(data)
    set({
      trainPlan: normalizedData,
      trainPlanMessage: agentConsultPlanMessage(normalizedData),
    })
  },

  clearCloudTrainPlan: () => {
    const state = get()
    state._planAbort?.abort()
    state._runtimeMatchAbort?.abort()
    state._sourcePreflightAbort?.abort()
    set({
      _planAbort: null,
      _runtimeMatchAbort: null,
      _sourcePreflightAbort: null,
      trainPlan: null,
      trainPlanMessage: '',
      cloudRuntimeMatch: null,
      cloudSourcePreflight: null,
      trainingPlanLoading: false,
      runtimeMatchLoading: false,
      sourcePreflightLoading: false,
    })
  },

  detachCurrentTrainJob: (reason = '已重新绑定云端实例，旧任务已移到历史。') => {
    const state = useTrainingStore.getState()
    const jobId = state.currentTrainJobId.trim()
    const mode = state.currentTrainMode
    const username = state.currentTrainUsername
    let trainJobHistory = state.trainJobHistory
    if (jobId) {
      trainJobHistory = upsertTrainJobHistory(trainJobHistory, {
        jobId,
        mode,
        username,
        status: 'detached',
        message: state.trainJobMessage || reason,
        updatedAt: new Date().toISOString(),
      })
      if (mode === 'cloud') {
        storeSuppressedCloudJobId(jobId)
      }
    }
    storeTrainJobId('')
    storeTrainUsername('')
    storeCloudStartParams({})
    set({
      trainJobHistory,
      currentTrainJobId: '',
      currentTrainUsername: '',
      currentTrainMode: 'cloud',
      trainJobMessage: reason,
    })
  },

  cancelCloudChecks: () => {
    const state = get()
    state._runtimeMatchAbort?.abort()
    state._sourcePreflightAbort?.abort()
    state._planAbort?.abort()
    set({
      _runtimeMatchAbort: null,
      _sourcePreflightAbort: null,
      _planAbort: null,
      runtimeMatchLoading: false,
      sourcePreflightLoading: false,
      trainingPlanLoading: false,
      trainPlanMessage: '已取消检查。',
    })
  },

  cancelCloudStartRequest: () => {
    get()._startAbort?.abort()
    set({
      _startAbort: null,
      trainingLoading: false,
      trainJobMessage: '已取消启动等待。若云端已经创建任务，请用“停止云训练”确认停止并释放资源。',
    })
  },

  handleTrainingWebSocketEvent: (payload) => {
    const event = firstRecord(payload)
    const raw = firstRecord(event.raw)
    const merged = { ...raw, ...event }
    const jobId = String(merged.job_id || merged.jobId || '').trim()
    if (!jobId) return
    const status = normalizeTrainStatus(merged)
    const running = merged.running === true || String(merged.running || '').toLowerCase() === 'true'
    const mode = String(merged.mode || get().currentTrainMode || 'cloud') === 'local' ? 'local' : 'cloud'
    const username = get().currentTrainUsername || loadStoredTrainUsername()
    const message = dataMessage({
      ...merged,
      job_id: jobId,
      running,
      status: merged.status || status,
      message: merged.message || raw.message || '',
    })
    const currentJobId = get().currentTrainJobId
    if (currentJobId && currentJobId !== jobId && !sameCloudJobChain(currentJobId, jobId)) return
    if (isActiveTrainStatus(status) || running) {
      storeTrainJobId(jobId)
      storeTrainMode(mode)
      if (username) storeTrainUsername(username)
      set({
        currentTrainJobId: jobId,
        currentTrainMode: mode,
        currentTrainUsername: username,
        trainJobMessage: message,
      })
      return
    }
    if (currentJobId && currentJobId !== jobId && sameCloudJobChain(currentJobId, jobId)) {
      return
    }
    const historyItem = makeTrainJobHistoryItem(jobId, mode, username, {
      ...merged,
      job_id: jobId,
      running: false,
      status: merged.status || status,
      message,
    })
    const trainJobHistory = upsertTrainJobHistory(get().trainJobHistory, historyItem)
    storeTrainJobId('')
    storeTrainUsername('')
    set({
      trainJobHistory,
      trainJobMessage: message || '训练任务已结束。',
      currentTrainJobId: '',
      currentTrainUsername: '',
    })
  },

  doCloudTrainStart: async (params) => {
    const username = activeUsername(params.username)
    const automationPolicy = cloudAutomationPolicy(useTrainingStore.getState().cloudAutomationMode)
    get()._startAbort?.abort()
    const controller = new AbortController()
    set({ _startAbort: controller, trainingLoading: true })
    try {
      const waitForSubmit = params.waitForSubmit ?? params.wait_for_submit ?? true
      const startPayload = {
        ...params,
        username,
        automation_mode: automationPolicy.mode,
        automation_policy: automationPolicy,
        provider: params.provider || '',
        policy_type: params.policy_type || params.params.policyType || 'act',
        steps: params.steps || params.params.steps || 100000,
        device: params.device || params.params.device || 'cuda',
        waitForSubmit,
        wait_for_submit: waitForSubmit,
      }
      storeCloudStartParams(startPayload as Record<string, any>)
      const data = await postJson(`${CLOUD_TRAIN}/start`, startPayload, { signal: controller.signal })
      const jobId = typeof data.job_id === 'string' ? data.job_id : ''
      if (jobId) {
        storeSuppressedCloudJobId('')
      }
      set({ restartableCloudJobId: '' })
      let statusData = firstRecord(data)
      if (jobId) {
        try {
          const verified = await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`)
          statusData = { ...statusData, ...firstRecord(verified) }
        } catch (statusError) {
          statusData = {
            ...statusData,
            status: 'failed',
            running: false,
            error: statusError instanceof Error ? statusError.message : 'status check failed',
            message: '云训练启动后没有确认到任务状态，已交给总控处理。',
          }
        }
      }
      const trainStatus = normalizeTrainStatus(statusData)
      const message = dataMessage(statusData)
      if (jobId) {
        storeTrainJobId(jobId)
        storeTrainMode('cloud')
        storeTrainUsername(username)
      } else {
        storeTrainJobId('')
        storeTrainUsername('')
      }
      if (jobId && !isActiveTrainStatus(trainStatus)) {
        const historyItem = makeTrainJobHistoryItem(jobId, 'cloud', username, statusData)
        const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
        set({
          trainJobHistory,
          trainJobMessage: message,
          currentTrainJobId: jobId,
          currentTrainMode: 'cloud',
          currentTrainUsername: username,
        })
        if (automationPolicy.autoRetrySameRuntime && shouldKeepCloudFailure(trainStatus, message)) {
          const handedToSupervisor = maybeStartBackendSupervisorWatch(username, jobId, trainStatus, statusData)
          if (handedToSupervisor) return
          void useTrainingStore.getState().repairCloudTraining({ username, job_id: jobId })
        }
        return
      }
      set({
        trainJobMessage: message || data.message || statusMessage(data),
        currentTrainJobId: jobId,
        currentTrainMode: 'cloud',
        currentTrainUsername: username,
      })
    } catch (error) {
      if (isAbortError(error)) {
        if (get()._startAbort === controller) {
          set({ trainJobMessage: '已取消启动等待。若云端已经创建任务，请用“停止云训练”确认停止并释放资源。' })
        }
        return
      }
      if (get()._startAbort === controller) {
        set({ trainJobMessage: error instanceof Error ? `云训练启动失败：${error.message}` : '云训练启动失败' })
      }
      throw error
    } finally {
      if (get()._startAbort === controller) {
        set({ _startAbort: null, trainingLoading: false })
      }
    }
  },

  repairCloudTraining: async (params: {
    username?: string
    job_id?: string
    task?: string
    automation_mode?: CloudAutomationMode
    user_guidance?: string
    userGuidance?: string
  } = {}): Promise<Record<string, unknown> | null> => {
    const state = get()
    const username = activeUsername(params.username || state.currentTrainUsername)
    const jobId = (params.job_id || state.currentTrainJobId || '').trim()
    const previous = loadStoredCloudStartParams()
    const previousParams = firstRecord(previous.params)
    const automationMode = normalizeCloudAutomationMode(params.automation_mode || state.cloudAutomationMode)
    const automationPolicy = cloudAutomationPolicy(automationMode)
    const userGuidance = String(params.user_guidance || params.userGuidance || '').trim()
    const task = params.task || [
      '根据当前失败任务和上一份云训练请求，自动诊断、修复并尽量续跑。',
      '优先复用同一台云端实例、已有源码目录、数据/模型缓存和已完成的 bootstrap 阶段；不要重新从零开始。',
      userGuidance ? `用户补充指令：${userGuidance}` : '',
      automationPolicy.autoRetrySameRuntime
        ? '如果修复只发生在同一运行时、同一预算和同一云端缓存内，可以直接续跑；新增费用、换机器或改密钥仍要停下确认。'
        : '只做诊断和修复建议，等待用户确认后才允许再次启动。',
    ].filter(Boolean).join(' ')
    get()._planAbort?.abort()
    const controller = new AbortController()
    set({
      _planAbort: controller,
      trainingPlanLoading: true,
      trainPlanMessage: automationPolicy.autoRetrySameRuntime
        ? '正在分析失败原因并尝试在同一云端实例续跑...'
        : '正在分析失败原因并生成下一步处理计划...',
      trainPlan: null,
    })
    try {
      let currentFailureStatus: Record<string, unknown> = {}
      if (jobId) {
        try {
          currentFailureStatus = firstRecord(await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`))
        } catch {
          currentFailureStatus = {}
        }
        if (isRuntimeBindingFailure(currentFailureStatus)) {
          const bridgeReady = await bridgeReadyAfterRuntimeRebind()
          if (bridgeReady) {
            storeTrainJobId('')
            storeTrainUsername('')
            if (get()._planAbort === controller) {
              set({
                trainPlan: null,
                trainPlanMessage: '已连接新的云端实例，旧实例失败任务已移到历史；当前可以重新启动任务。',
                trainJobMessage: '已连接新的云端实例，旧实例失败任务已移到历史；当前可以重新启动任务。',
                currentTrainJobId: '',
                currentTrainMode: 'cloud',
                currentTrainUsername: '',
              })
            }
            return { kind: 'evo_studio_runtime_rebound/v1', status: currentFailureStatus }
          }
          const message = dataMessage(currentFailureStatus) || '云端实例还没连上，请先重新绑定当前实例的最新 SSH 命令。'
          if (get()._planAbort === controller) {
            set({
              trainPlan: null,
              trainPlanMessage: '云端实例还没连上，不能进入修复规划；请先重新绑定当前实例的最新 SSH 命令。',
              trainJobMessage: message,
              currentTrainJobId: jobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
            })
          }
          return { kind: 'evo_studio_runtime_rebind_required/v1', status: currentFailureStatus }
        }
      }

      if (automationPolicy.autoRetrySameRuntime && jobId) {
        try {
          const supervisorStarted = await postJson(`${CLOUD_TRAIN}/supervisor/repair`, {
            username,
            jobId,
            automationPolicy,
            userGuidance,
          }, { signal: controller.signal })
          const supervisorJobId = typeof supervisorStarted.job_id === 'string' ? supervisorStarted.job_id : ''
          if (supervisorJobId) {
            storeTrainJobId(supervisorJobId)
            storeTrainMode('cloud')
            storeTrainUsername(username)
          }
          if (get()._planAbort === controller) {
            set({
              trainPlan: firstRecord(supervisorStarted),
              trainPlanMessage: '后端总控已复用当前云端实例提交续跑。',
              trainJobMessage: dataMessage(supervisorStarted),
              currentTrainJobId: supervisorJobId || jobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
            })
          }
          return firstRecord(supervisorStarted)
        } catch (supervisorError) {
          if (isAbortError(supervisorError)) throw supervisorError
          if (get()._planAbort === controller) {
            set({ trainPlanMessage: '后端总控需要人工审查，继续生成修复方案。' })
          }
        }
      }

      if (jobId) {
        if (Object.keys(currentFailureStatus).length === 0) {
          try {
            currentFailureStatus = firstRecord(await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`))
          } catch {
            currentFailureStatus = {}
          }
        }
        if (isRuntimeBindingFailure(currentFailureStatus)) {
          const bridgeReady = await bridgeReadyAfterRuntimeRebind()
          if (bridgeReady) {
            storeTrainJobId('')
            storeTrainUsername('')
            if (get()._planAbort === controller) {
              set({
                trainPlan: null,
                trainPlanMessage: '已连接新的云端实例，旧实例失败任务已移到历史；当前可以重新启动任务。',
                trainJobMessage: '已连接新的云端实例，旧实例失败任务已移到历史；当前可以重新启动任务。',
                currentTrainJobId: '',
                currentTrainMode: 'cloud',
                currentTrainUsername: '',
              })
            }
            return { kind: 'evo_studio_runtime_rebound/v1', status: currentFailureStatus }
          }
          const message = dataMessage(currentFailureStatus) || '云端实例还没连上，请先重新绑定当前实例的最新 SSH 命令。'
          if (get()._planAbort === controller) {
            set({
              trainPlan: null,
              trainPlanMessage: '云端实例还没连上，不能进入修复规划；请先重新绑定当前实例的最新 SSH 命令。',
              trainJobMessage: message,
              currentTrainJobId: jobId,
              currentTrainMode: 'cloud',
              currentTrainUsername: username,
            })
          }
          return { kind: 'evo_studio_runtime_rebind_required/v1', status: currentFailureStatus }
        }
      }

      if (automationPolicy.autoRetrySameRuntime && jobId && Object.keys(previousParams).length > 0) {
        const status = currentFailureStatus && Object.keys(currentFailureStatus).length > 0
          ? currentFailureStatus
          : await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`)
        const remediation = firstRecord(firstRecord(status).failureRemediation)
        const autoRepair = firstRecord(remediation.autoRepair)
        const bridgeExistingRuntime = usesExistingRuntime(state.cloudBridgeStatus, firstRecord(status))
        const safeRepair = remediation.code || autoRepair.strategy
          ? autoRepair.safe !== false
          : false
        const agentManagedRepair = automationPolicy.mode === 'full_auto'
        if (bridgeExistingRuntime && (safeRepair || agentManagedRepair)) {
          const previousTaskName = String(previous.task_name || previous.taskName || 'cloud-repair').replace(/-repair-[a-z0-9]+$/i, '')
          const repairStrategy = String(autoRepair.strategy || remediation.code || (
            agentManagedRepair ? 'agent_supervised_same_runtime_repair' : 'auto_repair'
          ))
          set({
            trainPlanMessage: `总控正在处理失败（${repairStrategy}），复用当前云端实例继续跑。`,
          })
          const repairParams: Record<string, any> = {
            ...previousParams,
            repairOfJobId: jobId,
            repairStrategy,
            forceRepairBootstrap: true,
            failureRemediation: remediation,
            failureContext: {
              status: firstRecord(status).status || '',
              error: firstRecord(status).error || '',
              logTail: firstRecord(status).log_tail || firstRecord(status).logTail || '',
              message: firstRecord(status).message || '',
              userGuidance,
            },
            supervisor: {
              mode: automationPolicy.mode,
              sameRuntimeOnly: true,
              inspectLogs: true,
              retryWithoutUserConfirmation: true,
              noRuntimeChange: true,
              noSecretChange: true,
              noBudgetIncrease: true,
              userGuidance,
            },
          }
          const repairBootstrapCommands = Array.isArray((previousParams as Record<string, any>).repairBootstrapCommands)
            ? (previousParams as Record<string, any>).repairBootstrapCommands
            : []
          if (repairBootstrapCommands.length > 0) {
            repairParams.repairBootstrapCommands = repairBootstrapCommands
          }
          for (const staleKey of [
            'bootstrapCommands',
            'bootstrapProfileSpec',
            'healthcheckCommands',
            'preflightCommands',
            'sourceResolutions',
            'command',
          ]) {
            delete repairParams[staleKey]
          }
          await useTrainingStore.getState().doCloudTrainStart({
            ...(previous as CloudTrainingStartParams),
            username,
            provider: String(previous.provider || firstRecord(status).provider || ''),
            workflow: String(previous.workflow || ''),
            sku_id: String(previous.sku_id || ''),
            image_id: String(previous.image_id || ''),
            task_name: `${previousTaskName}-repair-${Date.now().toString(36)}`,
            params: repairParams as unknown as CloudTrainingStartParams['params'],
          })
          return {
            kind: 'evo_studio_supervisor_repair/v1',
            mode: safeRepair ? 'deterministic_safe_repair' : 'agent_supervised_same_runtime_repair',
            status,
            repairStrategy,
            autoStarted: true,
          }
        }
      }

      const data = await postJson(EVO_STUDIO_AGENT, {
        task,
        mode: 'repair',
        username,
        workflow: String(previous.workflow || ''),
        params: previousParams,
        context: {
          page: 'training',
          username,
          provider: previous.provider || '',
          workflow: previous.workflow || '',
          params: previousParams,
          previousStart: previous,
          currentStatus: currentFailureStatus,
          job_id: jobId,
          jobId,
          currentJobMessage: state.trainJobMessage,
          automationPolicy,
          repairPolicy: {
            reuseWorkdir: true,
            reuseSourceCache: true,
            reuseSuccessfulStages: true,
            confirmedStartRequired: automationPolicy.paidStartRequiresConfirmation,
            autoRetrySameRuntime: automationPolicy.autoRetrySameRuntime,
          },
          userGuidance,
        },
        automation_policy: automationPolicy,
        provider: String(previous.provider || ''),
        sku_id: String(previous.sku_id || ''),
        image_id: String(previous.image_id || ''),
        job_id: jobId,
        confirmed: false,
      }, { signal: controller.signal })
      const normalizedData = normalizeAgentConsultPlan(data)
      if (get()._planAbort === controller) {
        set({
          trainPlan: normalizedData,
          trainPlanMessage: automationPolicy.autoRetrySameRuntime
            ? '已完成修复判断，正在确认是否可在同一云端实例内自动续跑。'
            : '已生成修复方案：会复用云端缓存和已完成阶段；确认后才会重新启动任务。',
        })
      }
      const vlaPlan = firstRecord(normalizedData.vlaPlan, normalizedData.plan)
      const nextParams = {
        ...previousParams,
        ...firstRecord(vlaPlan.params),
        repairOfJobId: jobId,
      }
      const hasPreviousRunnableJob = Boolean(
        String(previous.workflow || vlaPlan.workflow || '').trim()
          && Object.keys(previousParams).length > 0,
      )
      const ready = Boolean(normalizedData.readyForConfirmation || vlaPlan.readyToStart)
      const sameExistingRuntime = usesExistingRuntime(
        previousParams,
        nextParams,
        normalizedData.runtimeMatch,
        normalizedData,
        state.cloudBridgeStatus,
      )
      const canRetryExistingRuntime = automationPolicy.autoRetrySameRuntime
        && sameExistingRuntime
        && (ready || hasPreviousRunnableJob)
      if (canRetryExistingRuntime) {
        const previousTaskName = String(previous.task_name || previous.taskName || 'cloud-repair').replace(/-repair-[a-z0-9]+$/i, '')
        set({ trainPlanMessage: '总控正在复用当前云端实例自动修复并续跑。' })
        await useTrainingStore.getState().doCloudTrainStart({
          ...(previous as CloudTrainingStartParams),
          username,
          provider: String(normalizedData.provider || previous.provider || ''),
          workflow: String(vlaPlan.workflow || previous.workflow || ''),
          sku_id: String(vlaPlan.sku_id || previous.sku_id || ''),
          image_id: String(vlaPlan.image_id || previous.image_id || ''),
          task_name: `${previousTaskName}-repair-${Date.now().toString(36)}`,
          params: {
            ...nextParams,
            supervisor: {
              mode: automationPolicy.mode,
              sameRuntimeOnly: true,
              inspectLogs: true,
              retryWithoutUserConfirmation: true,
              noRuntimeChange: true,
              noSecretChange: true,
              noBudgetIncrease: true,
              userGuidance,
            },
          } as unknown as CloudTrainingStartParams['params'],
        })
      } else if (automationPolicy.autoRetrySameRuntime && get()._planAbort === controller) {
        set({
          trainPlanMessage: '这次修复涉及换资源、新费用或关键参数仍不完整，已停下等待你确认。',
        })
      }
      return normalizedData
    } catch (error) {
      if (isAbortError(error)) {
        if (get()._planAbort === controller) set({ trainPlanMessage: '已取消修复检查。' })
        return null
      }
      const message = error instanceof Error ? error.message : '自动修复失败'
      if (get()._planAbort === controller) {
        set({ trainPlan: null, trainPlanMessage: `自动修复失败：${message}` })
      }
      return null
    } finally {
      if (get()._planAbort === controller) {
        set({ _planAbort: null, trainingPlanLoading: false })
      }
    }
  },

  setCloudAutomationMode: (mode) => {
    const normalized = normalizeCloudAutomationMode(mode)
    storeCloudAutomationMode(normalized)
    set({ cloudAutomationMode: normalized })
  },

  doTrainStop: async () => {
    const jobId = useTrainingStore.getState().currentTrainJobId
    const mode = useTrainingStore.getState().currentTrainMode
    const username = useTrainingStore.getState().currentTrainUsername
    if (!jobId) {
      set({ trainJobMessage: 'No active training job id.' })
      return
    }
    set({ trainingStopLoading: true })
    try {
      const data = mode === 'cloud'
        ? await postJson(`${CLOUD_TRAIN}/stop`, { job_id: jobId, username })
        : await postJson(`${TRAIN}/stop`, { job_id: jobId })
      const historyItem = makeTrainJobHistoryItem(jobId, mode, username, {
        ...data,
        status: 'stopped',
        running: false,
      })
      const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
      storeTrainJobId('')
      storeTrainUsername('')
      set({
        trainJobHistory,
        trainJobMessage: data.message || statusMessage(data),
        currentTrainJobId: '',
        currentTrainUsername: '',
      })
    } finally {
      set({ trainingStopLoading: false })
    }
  },

  fetchTrainStatus: async (jobId) => {
    const mode = useTrainingStore.getState().currentTrainMode
    const username = useTrainingStore.getState().currentTrainUsername
    const data = mode === 'cloud'
      ? await api(`${CLOUD_TRAIN}/status/${encodeURIComponent(jobId)}?username=${encodeURIComponent(username)}`)
      : await api(`${TRAIN}/status/${encodeURIComponent(jobId)}`)
    const message = dataMessage(data)
    const trainStatus = normalizeTrainStatus(data)
    if (mode === 'cloud') {
      maybeStartBackendSupervisorWatch(username, jobId, trainStatus, data)
    }
    const supervisorJobId = mode === 'cloud' ? supervisorCurrentJobId(data) : ''
    if (supervisorJobId && supervisorJobId !== jobId) {
      storeTrainJobId(supervisorJobId)
      storeTrainMode('cloud')
      storeTrainUsername(username)
      set({
        trainJobMessage: message,
        currentTrainJobId: supervisorJobId,
        currentTrainMode: 'cloud',
        currentTrainUsername: username,
      })
      return
    }
    if (!isActiveTrainStatus(trainStatus) && useTrainingStore.getState().currentTrainJobId === jobId) {
      const historyItem = makeTrainJobHistoryItem(jobId, mode, username, data)
      const trainJobHistory = upsertTrainJobHistory(useTrainingStore.getState().trainJobHistory, historyItem)
      if (mode === 'cloud' && isRuntimeBindingFailure(data) && await bridgeReadyAfterRuntimeRebind()) {
        storeTrainJobId('')
        storeTrainUsername('')
        set({
          trainJobHistory,
          trainJobMessage: '已连接新的云端实例，旧实例失败任务已移到历史；当前可以重新启动任务。',
          currentTrainJobId: '',
          currentTrainMode: 'cloud',
          currentTrainUsername: '',
        })
        return
      }
      if (mode === 'cloud' && shouldKeepCloudFailure(trainStatus, message)) {
        set({
          trainJobHistory,
          trainJobMessage: message,
          currentTrainJobId: jobId,
          currentTrainMode: 'cloud',
          currentTrainUsername: username,
        })
        return
      }
      if (mode === 'cloud' && isCompletedTrainStatus(trainStatus, message)) {
        set({
          trainJobHistory,
          trainJobMessage: message,
          currentTrainJobId: jobId,
          currentTrainMode: 'cloud',
          currentTrainUsername: username,
        })
        return
      }
      storeTrainJobId('')
      storeTrainUsername('')
      set({
        trainJobHistory,
        trainJobMessage: message,
        currentTrainJobId: '',
        currentTrainUsername: '',
      })
      return
    }
    set({ trainJobMessage: message })
  },

  fetchTrainCurve: async (jobId) => {
    const data = await api(`${TRAIN}/curve/${encodeURIComponent(jobId)}`) as TrainingCurve
    set({ trainCurve: data })
  },

  clearTrainCurve: () => {
    set({ trainCurve: null })
  },
}))

function dataMessage(data: any) {
  const message = typeof data.message === 'string' ? data.message : statusMessage(data)
  const runtime = supervisorRuntime(data)
  const artifactLines = Array.isArray(data?.artifacts)
    ? data.artifacts
      .map((item: any) => {
        const kind = String(item?.kind || 'artifact').replace(/[^a-zA-Z0-9_/-]/g, '_')
        const path = String(item?.path || '').trim()
        return path ? `artifact_${kind}_path: ${path}` : ''
      })
      .filter(Boolean)
    : []
  const metricLines = isRecord(data?.metrics)
    ? Object.entries(data.metrics)
      .map(([key, value]) => `metric_${key}: ${formatStatusValue(value)}`)
      .filter(Boolean)
    : []
  const metricsPath = typeof data?.metricsPath === 'string' && data.metricsPath.trim()
    ? [`artifact_metrics_path: ${data.metricsPath.trim()}`]
    : []
  const enrichedMessage = [
    message,
    ...metricsPath,
    ...artifactLines,
    ...metricLines,
  ].filter(Boolean).join('\n')
  if (!runtime.state && !runtime.message && !runtime.currentJobId) return enrichedMessage
  if (payloadLooksLikeFailedCurrentJob(data, String(runtime.currentJobId || '').trim())) return enrichedMessage
  return [
    enrichedMessage,
    runtime.state ? `supervisor_state: ${runtime.state}` : '',
    runtime.message ? `supervisor_message: ${runtime.message}` : '',
    runtime.currentJobId ? `supervisor_current_job_id: ${runtime.currentJobId}` : '',
  ].filter(Boolean).join('\n')
}

function statusMessage(data: any) {
  return Object.entries(data)
    .map(([key, value]) => `${key}: ${formatStatusValue(value)}`)
    .join('\n')
}

function formatStatusValue(value: any): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}
