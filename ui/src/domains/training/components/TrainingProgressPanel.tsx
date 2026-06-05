import { useEffect, useRef, useState } from 'react'
import { useTrainingStore, type CloudAutomationMode } from '@/domains/training/store/useTrainingStore'
import { useI18n } from '@/i18n'
import { api } from '@/shared/api/client'

type Translate = ReturnType<typeof useI18n.getState>['t']
type CloudArtifact = { kind?: string; name?: string; path?: string; previewable?: boolean }
type CloudArtifactsPayload = {
  artifacts?: CloudArtifact[]
  metrics?: Record<string, unknown>
  metricsPath?: string
  metricsReadError?: string
  metricsParseError?: string
}

const FAILURE_HANDLING_OPTIONS: Array<{ id: CloudAutomationMode; label: string; hint: string }> = [
  {
    id: 'full_auto',
    label: '自动完成',
    hint: '同机同缓存同预算内尽量自己修复续跑',
  },
  {
    id: 'ask',
    label: '先确认',
    hint: '重要改动先问我',
  },
  {
    id: 'safe_auto',
    label: '同机修复',
    hint: '不换机器直接续跑',
  },
]

function sanitizeProgressMessage(message: string) {
  return message
    .replace(/(dataset_path|checkpoint_path):\s*\/root\/[^\n]+/g, '$1: 云端缓存已解析')
    .replace(/\/root\/[^\s'"`]+/g, '云端项目目录')
    .replace(/\/workspace\/outputs[^\s'"`]*/g, '云端产物目录')
}

function parseProgressFields(message: string) {
  const fields: Record<string, string> = {}
  for (const line of message.split('\n')) {
    const index = line.indexOf(':')
    if (index <= 0) continue
    const key = line.slice(0, index).trim()
    const value = line.slice(index + 1).trim()
    if (key && value) fields[key] = value
  }
  return fields
}

function statusLabel(status: string, running: boolean, failed: boolean, t: Translate) {
  const text = status.trim().toLowerCase()
  if (failed) return t('trainingStatusNeedsAction')
  if (running || ['running', 'submitted', 'submitting', 'pending', 'queued', 'starting'].some(item => text.includes(item))) return t('trainingStatusRunning')
  if (text.includes('completed') || text.includes('success')) return t('trainingStatusCompleted')
  if (text.includes('stopped')) return t('trainingStatusStopped')
  return text ? t('trainingStatusProcessing') : t('trainingStatusWaiting')
}

function stageLabel(stage: string, message: string, t: Translate) {
  const text = `${stage} ${message}`.toLowerCase()
  if (text.includes('prepare_code')) return t('trainingStagePrepareCode')
  if (text.includes('setup_env') || text.includes('bootstrap')) return t('trainingStageSetupEnv')
  if (text.includes('resolve_sources')) return t('trainingStageResolveSources')
  if (text.includes('healthcheck') || text.includes('preflight')) return t('trainingStagePreflight')
  if (text.includes('train_vla_rl_backend') || text.includes('run_eval') || text.includes('training')) return t('trainingStageRunning')
  if (text.includes('collect_artifacts') || text.includes('metrics') || text.includes('rollout_summary')) return t('trainingStageCollectArtifacts')
  return stage || message ? t('trainingStatusProcessing') : t('trainingStageWaiting')
}

function formatMetricValue(value: unknown) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (value == null) return ''
  return JSON.stringify(value)
}

function visibleMetricEntries(metrics?: Record<string, unknown>) {
  if (!metrics) return []
  const priority = ['success', 'return', 'reward', 'episode', 'num_trajectories', 'loss']
  return Object.entries(metrics)
    .filter(([, value]) => value == null || ['number', 'string', 'boolean'].includes(typeof value))
    .sort(([left], [right]) => {
      const leftIndex = priority.findIndex((item) => left.toLowerCase().includes(item))
      const rightIndex = priority.findIndex((item) => right.toLowerCase().includes(item))
      return (leftIndex < 0 ? 99 : leftIndex) - (rightIndex < 0 ? 99 : rightIndex)
    })
    .slice(0, 8)
}

function metricEntriesFromProgressFields(fields: Record<string, string>) {
  const metrics: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(fields)) {
    if (!key.startsWith('metric_')) continue
    const metricKey = key.slice('metric_'.length)
    const numeric = Number(value)
    metrics[metricKey] = Number.isFinite(numeric) && value.trim() !== '' ? numeric : value
  }
  return visibleMetricEntries(metrics)
}

function artifactItemsFromProgressFields(fields: Record<string, string>): CloudArtifact[] {
  return Object.entries(fields)
    .filter(([key, value]) => key.startsWith('artifact_') && key.endsWith('_path') && value.trim())
    .map(([key, value]) => {
      const kind = key.slice('artifact_'.length, -'_path'.length) || 'artifact'
      const path = value.trim()
      return {
        kind,
        name: path.split('/').pop() || path,
        path,
        previewable: /\.(json|txt|log)$/i.test(path),
      }
    })
}

function timelineState(
  stepIndex: number,
  currentIndex: number,
  failedIndex: number,
  failed: boolean,
  completed: boolean,
) {
  if (completed) return 'done'
  if (failed && stepIndex === failedIndex) return 'failed'
  if (failed && stepIndex < failedIndex) return 'done'
  if (stepIndex < currentIndex) return 'done'
  if (stepIndex === currentIndex) return 'current'
  return 'pending'
}

export function TrainingProgressPanel() {
  const { t } = useI18n()
  const currentTrainJobId = useTrainingStore((state) => state.currentTrainJobId)
  const currentTrainMode = useTrainingStore((state) => state.currentTrainMode)
  const currentTrainUsername = useTrainingStore((state) => state.currentTrainUsername)
  const trainJobMessage = useTrainingStore((state) => state.trainJobMessage)
  const trainPlan = useTrainingStore((state) => state.trainPlan)
  const trainPlanMessage = useTrainingStore((state) => state.trainPlanMessage)
  const trainingPlanLoading = useTrainingStore((state) => state.trainingPlanLoading)
  const cloudAutomationMode = useTrainingStore((state) => state.cloudAutomationMode)
  const setCloudAutomationMode = useTrainingStore((state) => state.setCloudAutomationMode)
  const restoreCurrentTrainJob = useTrainingStore((state) => state.restoreCurrentTrainJob)
  const fetchTrainStatus = useTrainingStore((state) => state.fetchTrainStatus)
  const repairCloudTraining = useTrainingStore((state) => state.repairCloudTraining)
  const autoRepairKeysRef = useRef<Set<string>>(new Set())
  const cloudArtifactsRequestKeyRef = useRef('')
  const [dismissedFailureDialogKey, setDismissedFailureDialogKey] = useState('')
  const [activeFailureDialogKey, setActiveFailureDialogKey] = useState('')
  const [failureWindowExpanded, setFailureWindowExpanded] = useState(false)
  const [failureInstruction, setFailureInstruction] = useState('')
  const [sessionRunJobIds, setSessionRunJobIds] = useState<Set<string>>(() => new Set())
  const [cloudArtifacts, setCloudArtifacts] = useState<CloudArtifactsPayload | null>(null)
  const [cloudArtifactsLoading, setCloudArtifactsLoading] = useState(false)
  const [cloudArtifactsError, setCloudArtifactsError] = useState('')
  const lowerMessage = trainJobMessage.toLowerCase()
  const cloudConnectionLost = (
    lowerMessage.includes('ssh connection failed') ||
    lowerMessage.includes('error reading ssh protocol banner') ||
    lowerMessage.includes('configured ssh instance is not reachable') ||
    lowerMessage.includes('unable to reach evo_train')
  )
  const cloudRunning = currentTrainMode === 'cloud' && Boolean(currentTrainJobId) && (
    lowerMessage.includes('status: running') ||
    lowerMessage.includes('status: submitted') ||
    lowerMessage.includes('running: true')
  ) && !cloudConnectionLost
  const cloudFailed = currentTrainMode === 'cloud' && Boolean(currentTrainJobId) && (
    cloudConnectionLost ||
    lowerMessage.includes('status: failed') ||
    lowerMessage.includes('status: missing') ||
    lowerMessage.includes('create task failed') ||
    (!cloudRunning && (
      lowerMessage.includes('error:') ||
      lowerMessage.includes('__evo_stage_failed__') ||
      lowerMessage.includes('stage_failed') ||
      lowerMessage.includes('terminated') ||
      lowerMessage.includes('killed') ||
      lowerMessage.includes('connection refused') ||
      lowerMessage.includes('traceback') ||
      lowerMessage.includes('syntaxerror') ||
      lowerMessage.includes('no matching distribution') ||
      lowerMessage.includes('requires a different python')
    ))
  )
  const failureModeLabel = cloudAutomationMode === 'safe_auto'
    ? '同机修复'
    : cloudAutomationMode === 'full_auto'
      ? '全托管'
      : '先确认'
  const repairStatusText = trainingPlanLoading
    ? cloudAutomationMode === 'ask'
      ? '正在分析失败原因并生成下一步处理计划...'
      : '正在分析失败原因并尝试在同一云端实例续跑...'
    : trainPlanMessage
      ? trainPlanMessage
      : cloudAutomationMode === 'ask'
        ? '等待你确认下一步，不会自动改环境或重启。'
        : '总控会读取日志并尝试续跑。'
  const hasRepairPlan = Boolean(trainPlan)
  const progressFields = parseProgressFields(trainJobMessage)
  const messageMetricEntries = metricEntriesFromProgressFields(progressFields)
  const messageArtifactItems = artifactItemsFromProgressFields(progressFields)
  const stableFailureReason = [
    progressFields.status || (cloudFailed ? 'failed' : ''),
    progressFields.stage || '',
    progressFields.error || '',
    lowerMessage.includes('__evo_stage_failed__') ? 'stage_failed' : '',
    lowerMessage.includes('modulenotfounderror') ? 'missing_module' : '',
    lowerMessage.includes('traceback') ? 'traceback' : '',
  ].filter(Boolean).join(':')
  const failureDialogKey = `${currentTrainJobId}:${stableFailureReason}`
  const dialogKey = activeFailureDialogKey || failureDialogKey
  const failureFromThisPageSession = currentTrainJobId ? sessionRunJobIds.has(currentTrainJobId) : false
  const showFailureDialog = !cloudRunning
    && cloudFailed
    && Boolean(activeFailureDialogKey || failureFromThisPageSession)
    && dismissedFailureDialogKey !== dialogKey
  const pythonVersionMismatch = lowerMessage.includes('requires a different python')
  const modelRunReached = /evaluating rollout epochs:\s*100%|eval\/success_|eval\/num_trajectories|eval\/episode_len|success_rate|rollout_summary/i.test(lowerMessage)
  const failedAfterModelRun = cloudFailed && modelRunReached
  const setupFailed = !failedAfterModelRun && (
    lowerMessage.includes('setup_env') ||
    lowerMessage.includes('bootstrap') ||
    lowerMessage.includes('no matching distribution') ||
    lowerMessage.includes('modulenotfounderror') ||
    pythonVersionMismatch
  )
  const nextPlanLines = failedAfterModelRun
    ? [
        '模型运行/评测已经执行到 rollout 或指标阶段。',
        '当前失败更像是结果收集、指标落盘或任务判定失败。',
        cloudAutomationMode === 'ask'
          ? '先检查已生成的指标和产物，再确认是否只修结果收集并续跑。'
          : '复用当前实例、缓存和已有产物，只修结果收集/判定链路后续跑。',
      ]
    : setupFailed
    ? pythonVersionMismatch
      ? [
          '确认失败是 Python 版本不兼容。',
          'RLinf 需要 Python 3.10-3.11；当前云端 Python 3.12 太新，不能直接安装。',
          cloudAutomationMode === 'ask'
            ? '先生成切换到兼容 Python 3.11 环境的修复方案，等你确认后续跑。'
            : '在同一实例内创建/复用兼容 Python 3.11 环境后自动续跑。',
        ]
      : [
          '确认失败发生在环境搭建/依赖安装阶段。',
          '检查日志里的依赖、Python 版本、网络和缓存状态。',
          cloudAutomationMode === 'ask'
            ? '先给出修复方案，等你确认后再续跑。'
            : '只在同一实例、同一缓存、同一预算内自动修复续跑。',
        ]
    : [
        '读取失败日志并归类原因。',
        '判断能否复用当前实例、缓存和预算继续。',
        cloudAutomationMode === 'ask'
          ? '先给出下一步方案，等你确认。'
          : '满足安全条件时自动继续执行。',
      ]
  const progressRows = [
    ['状态', progressFields.status],
    ['阶段', progressFields.stage],
    ['错误', progressFields.error],
  ].filter(([, value]) => value)
  const stageText = `${progressFields.stage || ''} ${lowerMessage}`
  const currentStatusLabel = statusLabel(progressFields.status || '', cloudRunning, cloudFailed, t)
  const currentStageLabel = stageLabel(progressFields.stage || '', lowerMessage, t)
  const isPreparationStage = cloudRunning && (
    /prepare_code|setup_env|bootstrap_runtime|resolve_sources|healthcheck_runtime/i.test(stageText)
  )
  const visibleStageLabel = failedAfterModelRun ? '评测已执行' : currentStageLabel
  const visibleStageHint = failedAfterModelRun
    ? '结果收集或任务判定失败，可基于当前产物修复'
    : cloudFailed
      ? '需要处理后才能继续'
      : isPreparationStage
        ? '正在准备环境，还没进入 GPU 训练'
        : '总控正在跟进任务'
  const resultReady = /completed|success|metrics\.json|rollout_summary|success_rate/i.test(stageText) && !cloudFailed
  const submitted = Boolean(currentTrainJobId)
  const prepareSeen = submitted && /prepare_code|setup_env|bootstrap|healthcheck|preflight|resolve_sources|train_vla|run_eval|metrics|rollout/i.test(stageText)
  const sourceSeen = submitted && /resolve_sources|dataset_path|checkpoint_path|cloud cache|evo_studio\/cache|train_vla|run_eval|metrics|rollout/i.test(stageText)
  const runSeen = submitted && /train_vla_rl_backend|run_eval|training|rollout|success_rate|metrics\.json/i.test(stageText)
  const collectSeen = submitted && /collect_artifacts|metrics\.json|rollout_summary|completed|success/i.test(stageText)
  const timelineCurrentIndex = collectSeen ? 4 : runSeen ? 3 : sourceSeen ? 2 : prepareSeen ? 1 : submitted ? 0 : -1
  const failedTimelineIndex = failedAfterModelRun
    ? 4
    : /setup_env|bootstrap|requires a different python|no matching distribution|modulenotfounderror/i.test(stageText)
    ? 1
    : /resolve_sources|huggingface|network is unreachable|dataset|checkpoint/i.test(stageText)
      ? 2
      : /preflight|train_vla|run_eval|runtimeerror|cuda/i.test(stageText)
        ? 3
        : 0
  const timelineSteps = [
    { label: '提交', detail: '任务进入云端队列' },
    { label: '准备', detail: '代码和环境' },
    { label: '数据/模型', detail: '下载或复用缓存' },
    { label: '运行', detail: '训练或评测' },
    { label: '结果', detail: '指标和产物' },
  ]
  const metricEntries = visibleMetricEntries(cloudArtifacts?.metrics)
  const artifactItems = cloudArtifacts?.artifacts || []
  const displayMetricEntries = metricEntries.length > 0 ? metricEntries : messageMetricEntries
  const displayArtifactItems = artifactItems.length > 0 ? artifactItems : messageArtifactItems

  useEffect(() => {
    if (!cloudRunning || !currentTrainJobId) return
    setSessionRunJobIds((previous) => {
      if (previous.has(currentTrainJobId)) return previous
      const next = new Set(previous)
      next.add(currentTrainJobId)
      return next
    })
  }, [cloudRunning, currentTrainJobId])

  useEffect(() => {
    if (currentTrainMode !== 'cloud' || !currentTrainJobId || !resultReady) {
      cloudArtifactsRequestKeyRef.current = ''
      setCloudArtifacts(null)
      setCloudArtifactsError('')
      setCloudArtifactsLoading(false)
      return
    }

    if (messageMetricEntries.length > 0 || messageArtifactItems.length > 0) {
      setCloudArtifactsLoading(false)
      return
    }

    const requestKey = `${currentTrainUsername || ''}:${currentTrainJobId}`
    if (cloudArtifactsRequestKeyRef.current === requestKey) {
      return
    }
    cloudArtifactsRequestKeyRef.current = requestKey
    let cancelled = false
    setCloudArtifactsLoading(true)
    setCloudArtifactsError('')
    const query = new URLSearchParams({ job_id: currentTrainJobId })
    if (currentTrainUsername) query.set('username', currentTrainUsername)
    api(`/api/train/cloud/artifacts?${query.toString()}`)
      .then((payload) => {
        if (cancelled) return
        setCloudArtifacts(payload as CloudArtifactsPayload)
      })
      .catch((error) => {
        if (cancelled) return
        setCloudArtifactsError(error instanceof Error ? error.message : String(error))
      })
      .finally(() => {
        if (!cancelled) setCloudArtifactsLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [currentTrainJobId, currentTrainMode, currentTrainUsername, resultReady])

  useEffect(() => {
    if (!cloudRunning) return
    setActiveFailureDialogKey('')
    setFailureWindowExpanded(false)
  }, [cloudRunning])

  useEffect(() => {
    if (!cloudFailed || cloudRunning || !failureFromThisPageSession) return
    if (dismissedFailureDialogKey === failureDialogKey) return
    setActiveFailureDialogKey(failureDialogKey)
    setFailureWindowExpanded(false)
  }, [cloudFailed, cloudRunning, dismissedFailureDialogKey, failureDialogKey, failureFromThisPageSession])

  useEffect(() => {
    if (currentTrainJobId) return
    void restoreCurrentTrainJob()
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible' && !useTrainingStore.getState().currentTrainJobId) {
        void restoreCurrentTrainJob()
      }
    }, 10000)

    return () => window.clearInterval(timer)
  }, [currentTrainJobId, restoreCurrentTrainJob])

  useEffect(() => {
    const jobId = currentTrainJobId.trim()
    if (!jobId) {
      return
    }
    if (currentTrainMode === 'cloud' && resultReady) {
      return
    }

    void fetchTrainStatus(jobId)
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void fetchTrainStatus(jobId)
      }
    }, 5000)

    return () => window.clearInterval(timer)
  }, [currentTrainJobId, currentTrainMode, fetchTrainStatus, resultReady])

  useEffect(() => {
    if (!cloudFailed || cloudRunning || !failureFromThisPageSession || cloudAutomationMode === 'ask' || trainingPlanLoading || failureWindowExpanded) return
    const key = failureDialogKey
    if (autoRepairKeysRef.current.has(key)) return
    autoRepairKeysRef.current.add(key)
    void repairCloudTraining({ job_id: currentTrainJobId })
  }, [
    cloudAutomationMode,
    cloudFailed,
    cloudRunning,
    currentTrainJobId,
    failureDialogKey,
    failureFromThisPageSession,
    failureWindowExpanded,
    repairCloudTraining,
    trainingPlanLoading,
  ])

  const handleFailureChoice = (mode: CloudAutomationMode) => {
    setCloudAutomationMode(mode)
    setActiveFailureDialogKey(dialogKey)
  }

  const openFailureDialog = () => {
    setDismissedFailureDialogKey('')
    setActiveFailureDialogKey(failureDialogKey)
    setFailureWindowExpanded(true)
  }

  const confirmFailureHandling = () => {
    setActiveFailureDialogKey(dialogKey)
    setFailureWindowExpanded(false)
    void repairCloudTraining({
      job_id: currentTrainJobId,
      automation_mode: pythonVersionMismatch && cloudAutomationMode === 'ask' ? 'safe_auto' : cloudAutomationMode,
      user_guidance: failureInstruction.trim(),
    })
  }

  if (currentTrainMode === 'cloud' && !currentTrainJobId) {
    return null
  }

  if (!currentTrainJobId && !trainJobMessage.trim()) {
    return null
  }

  return (
    <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-cy">
      {showFailureDialog && (
        <div className={`fixed bottom-4 right-4 z-50 pointer-events-none transition-all duration-300 ease-out ${
          failureWindowExpanded ? 'w-[min(420px,calc(100vw-32px))]' : 'w-[min(320px,calc(100vw-32px))]'
        }`}>
          {failureWindowExpanded ? (
            <div className="pointer-events-auto rounded-xl border border-bd bg-sf shadow-card p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="grid h-6 w-6 place-items-center rounded-full bg-yl/15 text-xs font-bold text-yl">!</span>
                    <div className="text-sm font-bold text-tx">总控暂停了任务</div>
                  </div>
                  <div className="mt-1 text-[11px] text-tx3">
                    {failedAfterModelRun ? '评测已执行，结果收集需要修复。' : setupFailed ? '环境准备失败，需要修复后继续。' : '已停下等待处理。'}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setFailureWindowExpanded(false)}
                  className="shrink-0 rounded-md px-2 py-1 text-[11px] font-semibold text-tx3 hover:bg-bg hover:text-tx"
                >
                  收起
                </button>
              </div>

              <div className="mt-3 rounded-lg border border-bd/50 bg-bg px-3 py-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs font-semibold text-tx">总控看到的问题</div>
                  <div className="text-[11px] text-tx3">可直接补一句</div>
                </div>
                <div className="mt-1 text-xs text-tx3 leading-relaxed">
                  {nextPlanLines[nextPlanLines.length - 1] || repairStatusText}
                </div>
                {trainingPlanLoading && (
                  <div className="mt-2 text-[11px] text-ac">正在检查日志和云端状态...</div>
                )}
              </div>

              <textarea
                value={failureInstruction}
                onChange={(event) => setFailureInstruction(event.target.value)}
                rows={3}
                placeholder="对总控说，例如：不要换机器，先检查现有 conda 环境；如果只是依赖问题就同机修复。"
                className="mt-3 w-full resize-none rounded-lg border border-bd bg-bg px-3 py-2 text-xs text-tx outline-none transition-colors placeholder:text-tx3 focus:border-ac focus:ring-2 focus:ring-ac/10"
              />

              <div className="mt-3 flex items-center justify-between gap-3">
                <div className="text-[11px] font-semibold text-tx3">处理权限</div>
                <div className="flex rounded-lg border border-bd bg-bg p-1">
                  {FAILURE_HANDLING_OPTIONS.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      title={item.hint}
                      onClick={() => handleFailureChoice(item.id)}
                      className={`rounded-md px-2.5 py-1.5 text-[11px] font-semibold transition-colors ${
                        cloudAutomationMode === item.id
                          ? 'bg-ac text-white shadow-glow-ac'
                          : 'text-tx3 hover:bg-sf hover:text-tx'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-3 flex justify-end gap-2">
                <button
                  type="button"
                  disabled={trainingPlanLoading}
                  onClick={confirmFailureHandling}
                  className="px-3 py-1.5 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2
                    transition-all active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {trainingPlanLoading
                    ? '处理中...'
                    : pythonVersionMismatch
                      ? '自动配置环境并续跑'
                      : failureInstruction.trim()
                        ? '发送给总控'
                        : '继续处理'}
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={openFailureDialog}
              className="pointer-events-auto w-full rounded-full border border-yl/40 bg-sf/95 shadow-card px-3 py-2 text-left hover:border-ac/60 transition-colors"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex items-center gap-2">
                  <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-yl/15 text-[10px] font-bold text-yl">!</span>
                  <div className="truncate text-xs font-bold text-tx">总控暂停了任务</div>
                </div>
                <span className="shrink-0 rounded-full bg-ac/10 px-2 py-1 text-[11px] font-semibold text-ac">
                  处理
                </span>
              </div>
            </button>
          )}
        </div>
      )}
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide">
            {currentTrainMode === 'cloud' ? '当前任务' : (t('trainingProgress') || t('trainJobStatus'))}
          </h3>
          {currentTrainMode === 'cloud' && (
            <div className="mt-1 text-xs text-tx3">
              {currentTrainJobId ? '状态、阶段和日志都在这里。' : '任务启动后会显示状态。'}
            </div>
          )}
        </div>
        <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${
          cloudFailed
            ? 'border-yl/40 bg-yl/10 text-yl'
            : cloudRunning
              ? 'border-ac/40 bg-ac/10 text-ac'
              : 'border-bd bg-bg text-tx3'
        }`}>
          {currentTrainMode === 'cloud'
            ? (currentTrainJobId ? currentStatusLabel : t('trainingStatusWaiting'))
            : (currentTrainJobId ? 'Active' : (t('noActiveTraining') || 'No active training'))}
        </span>
      </div>

      <div className="min-h-[160px] rounded-lg bg-bg border border-bd/30 p-3">
        {currentTrainMode === 'cloud' ? (
          currentTrainJobId ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-bd/50 bg-sf px-3 py-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-tx">{visibleStageLabel}</div>
                    <div className="mt-1 truncate text-[11px] text-tx3">
                      {visibleStageHint}
                    </div>
                  </div>
                  <details className="shrink-0 text-[11px] text-tx3">
                    <summary className="cursor-pointer select-none rounded-md border border-bd px-2 py-1 font-semibold text-tx2 hover:border-ac hover:text-ac">
                      详情
                    </summary>
                    <div className="absolute z-10 mt-1 max-w-[360px] rounded-md border border-bd bg-sf px-3 py-2 shadow-card">
                      <div className="font-semibold text-tx2">任务 ID</div>
                      <div className="mt-1 break-all font-mono text-[11px] text-tx3">{currentTrainJobId}</div>
                    </div>
                  </details>
                </div>
              </div>

              <div className="rounded-lg border border-bd/50 bg-sf px-3 py-2">
                <div className="grid grid-cols-5 items-start gap-1.5 max-[760px]:grid-cols-5">
                  {timelineSteps.map((step, index) => {
                    const state = timelineState(index, timelineCurrentIndex, failedTimelineIndex, cloudFailed, resultReady)
                    return (
                      <div
                        key={step.label}
                        className="min-w-0 text-center"
                      >
                        <div className="flex items-center gap-1">
                          <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-[10px] font-bold ${
                            state === 'failed'
                              ? 'bg-yl text-white'
                              : state === 'current'
                                ? 'bg-ac text-white'
                                : state === 'done'
                                  ? 'bg-gn text-white'
                                  : 'bg-bd/50 text-tx3'
                          }`}>
                            {state === 'done' ? '✓' : state === 'failed' ? '!' : index + 1}
                          </span>
                          {index < timelineSteps.length - 1 && (
                            <span className={`h-px flex-1 ${
                              state === 'done' ? 'bg-gn/60' : state === 'failed' ? 'bg-yl/60' : 'bg-bd'
                            }`} />
                          )}
                        </div>
                        <div className="mt-1 truncate text-[11px] font-semibold text-tx2">{step.label}</div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {resultReady && (
                <div className="rounded-lg border border-gn/40 bg-gn/10 px-3 py-2">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="text-xs font-semibold text-tx">云端产物</div>
                      <div className="mt-1 text-[11px] text-tx3">
                        完成后自动读取 metrics.json，并保留云端产物路径。
                      </div>
                    </div>
                    {cloudArtifactsLoading && displayMetricEntries.length === 0 && displayArtifactItems.length === 0 && (
                      <span className="rounded-full border border-bd bg-sf px-2 py-1 text-[11px] font-semibold text-tx3">
                        读取中
                      </span>
                    )}
                  </div>

                  {cloudArtifactsError && (
                    <div className="mt-2 rounded-md border border-yl/40 bg-yl/10 px-2.5 py-2 text-xs text-tx2">
                      读取产物失败：{cloudArtifactsError}
                    </div>
                  )}

                  {cloudArtifacts?.metricsReadError && (
                    <div className="mt-2 rounded-md border border-yl/40 bg-yl/10 px-2.5 py-2 text-xs text-tx2">
                      指标文件暂时无法读取：{cloudArtifacts.metricsReadError}
                    </div>
                  )}

                  {cloudArtifacts?.metricsParseError && (
                    <div className="mt-2 rounded-md border border-yl/40 bg-yl/10 px-2.5 py-2 text-xs text-tx2">
                      指标文件不是合法 JSON：{cloudArtifacts.metricsParseError}
                    </div>
                  )}

                  {displayMetricEntries.length > 0 && (
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      {displayMetricEntries.map(([key, value]) => (
                        <div key={key} className="rounded-md border border-gn/30 bg-sf/80 px-2.5 py-2">
                          <div className="truncate text-[11px] font-semibold text-tx3">{key}</div>
                          <div className="mt-1 font-mono text-sm font-bold text-tx">{formatMetricValue(value)}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {displayArtifactItems.length > 0 && (
                    <details className="mt-2 rounded-md border border-bd/50 bg-sf px-2.5 py-2" open>
                      <summary className="cursor-pointer select-none text-xs font-semibold text-tx2">
                        产物路径
                      </summary>
                      <div className="mt-2 grid gap-2">
                        {displayArtifactItems.map((item) => (
                          <div key={`${item.kind || 'artifact'}:${item.path || item.name}`} className="rounded-md border border-bd/60 bg-bg px-2.5 py-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="rounded-full border border-bd bg-sf px-2 py-0.5 text-[10px] font-semibold text-tx3">
                                {item.kind || 'artifact'}
                              </span>
                              <span className="text-xs font-semibold text-tx2">{item.name || '云端文件'}</span>
                            </div>
                            {item.path && (
                              <div className="mt-1 break-all font-mono text-[11px] text-tx3">{item.path}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}

                  {!cloudArtifactsLoading && !cloudArtifactsError && !cloudArtifacts?.metricsReadError && displayMetricEntries.length === 0 && displayArtifactItems.length === 0 && (
                    <div className="mt-2 text-xs text-tx3">
                      任务完成了，但后端还没有返回可展示的 metrics 或产物路径。
                    </div>
                  )}
                </div>
              )}

              {cloudFailed && (
                <div className="rounded-lg border border-yl/40 bg-yl/10 px-3 py-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="text-xs font-semibold text-tx">
                        {failedAfterModelRun ? '评测已执行，结果待修复' : failureFromThisPageSession ? '任务遇到问题' : '上次任务失败，已记录'}
                      </div>
                      <div className="mt-1 text-xs text-tx3 leading-relaxed">
                        {failedAfterModelRun ? '模型已经跑到 rollout/指标阶段；现在需要修复指标落盘或任务判定，不应从零开始。' : failureFromThisPageSession ? repairStatusText : '这是之前任务的失败记录；不会自动处理，点击后才会继续。'}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={openFailureDialog}
                        className="px-3 py-1.5 rounded-md border border-bd bg-sf text-xs font-semibold text-tx2 hover:border-ac hover:text-ac
                          transition-all active:scale-[0.97]"
                      >
                        处理
                      </button>
                      <button
                        type="button"
                        disabled={trainingPlanLoading}
                        onClick={() => { void repairCloudTraining({ job_id: currentTrainJobId }) }}
                        className="px-3 py-1.5 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2
                          transition-all active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        {trainingPlanLoading ? '处理中...' : '交给总控'}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {progressRows.length > 0 && (
                <details className="rounded-md border border-bd/50 bg-sf px-3 py-2">
                  <summary className="cursor-pointer select-none text-xs font-semibold text-tx2">
                    当前详情
                  </summary>
                  <div className="mt-2 grid gap-2">
                    {progressRows.map(([label, value]) => (
                      <div key={label} className="grid grid-cols-[72px_minmax(0,1fr)] gap-2 rounded-md border border-bd/60 bg-bg px-3 py-2">
                        <div className="text-[11px] font-semibold text-tx3">{label}</div>
                        <div className="text-xs text-tx2 break-words">{sanitizeProgressMessage(value)}</div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              <details className="rounded-md border border-bd/50 bg-sf px-3 py-2">
                <summary className="cursor-pointer select-none text-xs font-semibold text-tx2">
                  查看日志
                </summary>
                <pre className="mt-2 max-h-[360px] overflow-auto text-xs text-tx2 font-mono whitespace-pre-wrap break-all">
                  {sanitizeProgressMessage(trainJobMessage || '暂无日志')}
                </pre>
              </details>
            </div>
          ) : (
            <div className="h-full min-h-[180px] flex items-center justify-center text-sm text-tx3 text-center">
              任务启动后，这里会显示提交、准备、数据/模型、运行和结果。
            </div>
          )
        ) : trainJobMessage ? (
          <div className="space-y-3">
            {isPreparationStage && (
              <div className="rounded-lg border border-yl/40 bg-yl/10 px-3 py-2">
                <div className="text-xs font-semibold text-tx">正在准备环境 · 还没有进入 GPU 训练</div>
              </div>
            )}
            {cloudFailed && (
              <div className="rounded-lg border border-yl/40 bg-yl/10 px-3 py-2">
                <div className="text-xs font-semibold text-tx">
                  {failureFromThisPageSession ? '任务遇到问题，等待确认' : '上次任务失败，已记录'}
                </div>
                <div className="mt-2 rounded-md border border-bd/50 bg-sf/80 px-2.5 py-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold text-tx2">处理方式：{failureModeLabel}</span>
                    {hasRepairPlan && (
                      <span className="rounded-full border border-ac/40 bg-ac/10 px-2 py-0.5 text-[10px] font-semibold text-ac">
                        已有修复上下文
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-xs text-tx3 leading-relaxed">
                    {failureFromThisPageSession ? repairStatusText : '这是之前任务的失败记录；不会自动处理，点击后才会继续。'}
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={openFailureDialog}
                    className="px-3 py-1.5 rounded-md border border-bd bg-sf text-xs font-semibold text-tx2 hover:border-ac hover:text-ac
                      transition-all active:scale-[0.97]"
                  >
                    打开处理窗口
                  </button>
                  <button
                    type="button"
                    disabled={trainingPlanLoading}
                    onClick={() => { void repairCloudTraining({ job_id: currentTrainJobId }) }}
                    className="px-3 py-1.5 rounded-md bg-ac text-white text-xs font-semibold hover:bg-ac2
                      transition-all active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {trainingPlanLoading ? '处理中...' : hasRepairPlan ? '继续处理' : '交给总控继续'}
                  </button>
                </div>
              </div>
            )}
            {progressRows.length > 0 && (
              <div className="grid gap-2">
                {progressRows.map(([label, value]) => (
                  <div key={label} className="grid grid-cols-[72px_minmax(0,1fr)] gap-2 rounded-md border border-bd/60 bg-sf/80 px-3 py-2">
                    <div className="text-[11px] font-semibold text-tx3">{label}</div>
                    <div className="text-xs text-tx2 break-words">{sanitizeProgressMessage(value)}</div>
                  </div>
                ))}
              </div>
            )}
            <details className="rounded-md border border-bd/50 bg-sf px-3 py-2">
              <summary className="cursor-pointer select-none text-xs font-semibold text-tx2">
                查看日志
              </summary>
              <pre className="mt-2 max-h-[360px] overflow-auto text-xs text-tx2 font-mono whitespace-pre-wrap break-all">
                {sanitizeProgressMessage(trainJobMessage)}
              </pre>
            </details>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-sm text-tx3 text-center">
            {t('noTrainingProgress') || 'Training progress will appear here after a job starts.'}
          </div>
        )}
      </div>
    </section>
  )
}
