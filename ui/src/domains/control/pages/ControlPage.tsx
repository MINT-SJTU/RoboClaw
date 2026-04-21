import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessionStore, type SessionState } from '@/domains/session/store/useSessionStore'
import { useHardwareStore, type HardwareCapabilities, type OperationCapability } from '@/domains/hardware/store/useHardwareStore'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'
import { useI18n } from '@/i18n'
import { CameraPreviewPanel } from '@/domains/control/components/CameraPreviewPanel'
import { ServoPanel } from '@/domains/hardware/components/ServoPanel'

const blockedCapability: OperationCapability = { ready: false, missing: [] }

function capabilityOf(
  capabilities: HardwareCapabilities | undefined,
  name: keyof HardwareCapabilities,
): OperationCapability {
  return capabilities?.[name] ?? blockedCapability
}

function canDo(
  state: SessionState,
  capabilities: HardwareCapabilities | undefined,
  useRecordCameras: boolean,
) {
  const canStart = state === 'idle' || state === 'error'
  const tele = state === 'teleoperating'
  const rec = state === 'recording'
  const rep = state === 'replaying'
  const inf = state === 'inferring'
  const recordCapability = capabilityOf(
    capabilities,
    useRecordCameras ? 'record' : 'record_without_cameras',
  )
  const inferCapability = capabilityOf(capabilities, 'infer')
  return {
    teleopStart: canStart && capabilityOf(capabilities, 'teleop').ready,
    teleopStop: tele,
    recStart: canStart && recordCapability.ready,
    recStop: rec,
    saveEp: rec,
    discardEp: rec,
    replayStart: canStart && capabilityOf(capabilities, 'replay').ready,
    replayStop: rep,
    inferStart: canStart && inferCapability.ready,
    inferStop: inf,
  }
}

function ActionBtn({
  children, disabled, onClick, color, title,
}: {
  children: React.ReactNode; disabled?: boolean; onClick?: () => void
  color: 'ac' | 'gn' | 'rd' | 'yl'; title?: string
}) {
  const cls: Record<string, string> = {
    ac: 'bg-ac hover:bg-ac2 shadow-glow-ac',
    gn: 'bg-gn hover:bg-gn/90 shadow-glow-gn',
    rd: 'bg-rd hover:bg-rd/90 shadow-glow-rd',
    yl: 'bg-yl hover:bg-yl/90 shadow-glow-yl',
  }
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      title={title}
      className={`w-full px-4 py-2.5 rounded-lg text-sm font-semibold text-white transition-all
        active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed disabled:shadow-none ${cls[color]}`}
    >
      {children}
    </button>
  )
}

function StatusDot({ color, label }: { color: 'ac' | 'gn' | 'yl'; label: string }) {
  const dotCls: Record<string, string> = { ac: 'bg-ac', gn: 'bg-gn', yl: 'bg-yl' }
  const textCls: Record<string, string> = { ac: 'text-ac', gn: 'text-gn', yl: 'text-yl' }
  return (
    <div className={`mt-2 flex items-center gap-2 text-xs font-medium ${textCls[color]}`}>
      <span className={`w-2 h-2 rounded-full animate-pulse ${dotCls[color]}`} />
      {label}
    </div>
  )
}

export default function ControlPage() {
  const session = useSessionStore((state) => state.session)
  const loading = useSessionStore((state) => state.loading)
  const fetchSessionStatus = useSessionStore((state) => state.fetchSessionStatus)
  const doDismissError = useSessionStore((state) => state.doDismissError)
  const doTeleopStart = useSessionStore((state) => state.doTeleopStart)
  const doTeleopStop = useSessionStore((state) => state.doTeleopStop)
  const doRecordStart = useSessionStore((state) => state.doRecordStart)
  const doRecordStop = useSessionStore((state) => state.doRecordStop)
  const doSaveEpisode = useSessionStore((state) => state.doSaveEpisode)
  const doDiscardEpisode = useSessionStore((state) => state.doDiscardEpisode)
  const doSkipReset = useSessionStore((state) => state.doSkipReset)
  const doReplayStart = useSessionStore((state) => state.doReplayStart)
  const doReplayStop = useSessionStore((state) => state.doReplayStop)
  const doInferStart = useSessionStore((state) => state.doInferStart)
  const doInferStop = useSessionStore((state) => state.doInferStop)
  const hwStatus = useHardwareStore((state) => state.hardwareStatus)
  const fetchHardwareStatus = useHardwareStore((state) => state.fetchHardwareStatus)
  const datasets = useDatasetsStore((state) => state.datasets)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const policies = useTrainingStore((state) => state.policies)
  const loadPolicies = useTrainingStore((state) => state.loadPolicies)
  const { t } = useI18n()
  const navigate = useNavigate()
  const { state, episode_phase: episodePhase, saved_episodes: savedEpisodes, target_episodes: targetEpisodes, embodiment_owner: owner, prepare_stage: prepareStage } = session
  const teleopStopping = loading === 'teleop-stop' || (state === 'stopping' && owner === 'teleop')
  const recordStopping = loading === 'record-stop' || (state === 'stopping' && owner === 'recording')
  const replayStopping = loading === 'replay-stop' || (state === 'stopping' && owner === 'replaying')
  const inferStopping = loading === 'infer-stop' || (state === 'stopping' && owner === 'inferring')

  function translateMissing(msg: string): string {
    if (msg === 'No follower arm configured') return t('hwMissingNoFollower')
    if (msg === 'No leader arm configured') return t('hwMissingNoLeader')
    if (msg.includes('is disconnected') && msg.startsWith('Arm'))
      return `${msg.match(/Arm '(.+?)'/)?.[1] ?? ''} ${t('hwMissingDisconnected')}`
    if (msg.includes('is not calibrated'))
      return `${msg.match(/Arm '(.+?)'/)?.[1] ?? ''} ${t('hwMissingNotCalibrated')}`
    if (msg.includes('is disconnected') && msg.startsWith('Camera'))
      return `${msg.match(/Camera '(.+?)'/)?.[1] ?? ''} ${t('hwMissingCameraDisconnected')}`
    if (msg === 'No cameras configured') return t('hwMissingNoCamera')
    if (msg.includes('mismatch')) return t('hwMissingCountMismatch')
    return msg
  }

  // Merged record/infer card
  type OpMode = 'record' | 'infer'
  const [mode, setMode] = useState<OpMode>('record')
  const [task, setTask] = useState('')
  const [numEp, setNumEp] = useState(10)
  const [episodeTime, setEpisodeTime] = useState(300)
  const [resetTime, setResetTime] = useState(10)
  const [datasetName, setDatasetName] = useState('')
  const [fps, setFps] = useState(30)
  const [useCameras, setUseCameras] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [inferCheckpoint, setInferCheckpoint] = useState('')
  // Replay
  const [replayDataset, setReplayDataset] = useState('')
  const [replayEpisode, setReplayEpisode] = useState(0)
  const hwReady = hwStatus?.ready ?? false
  const capabilities = hwStatus?.capabilities
  const ok = canDo(state, capabilities, useCameras)
  const teleopCapability = capabilityOf(capabilities, 'teleop')
  const recordCapability = capabilityOf(capabilities, useCameras ? 'record' : 'record_without_cameras')
  const replayCapability = capabilityOf(capabilities, 'replay')
  const inferCapability = capabilityOf(capabilities, 'infer')

  useEffect(() => {
    void loadDatasets()
    void loadPolicies()
    void fetchHardwareStatus()
    void fetchSessionStatus()
    const pollInterval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        void fetchHardwareStatus()
      }
    }, 5000)
    return () => clearInterval(pollInterval)
  }, [fetchHardwareStatus, fetchSessionStatus, loadDatasets, loadPolicies])

  const stateLabel: Record<string, string> = {
    preparing: t('hwInitializing'),
    teleoperating: t('stateTeleoperating'),
    recording: t('stateRecording'),
    replaying: t('stateReplaying'),
    inferring: t('stateInferring'),
    stopping: t('stateStopping'),
    calibrating: t('calibrating'),
  }
  const stateBadgeCls: Record<string, string> = {
    preparing: 'bg-yl/15 text-yl border-yl/30',
    teleoperating: 'bg-ac/15 text-ac border-ac/30',
    recording: 'bg-rd/15 text-rd border-rd/30',
    replaying: 'bg-gn/15 text-gn border-gn/30',
    inferring: 'bg-ac/15 text-ac border-ac/30',
    stopping: 'bg-yl/15 text-yl border-yl/30',
    calibrating: 'bg-yl/15 text-yl border-yl/30',
  }

  const busy = state !== 'idle' && state !== 'error'

  // Local elapsed timer — WS events stop after state transitions, so we tick locally
  const [elapsedTick, setElapsedTick] = useState(0)
  useEffect(() => {
    setElapsedTick(Math.round(session.elapsed_seconds) || 0)
  }, [session.elapsed_seconds])
  useEffect(() => {
    if (!busy) return
    const interval = setInterval(() => setElapsedTick(t => t + 1), 1000)
    return () => clearInterval(interval)
  }, [busy])

  const [taskError, setTaskError] = useState(false)

  function handleRecordStart() {
    if (!task.trim()) {
      setTaskError(true)
      setTimeout(() => setTaskError(false), 1500)
      return
    }
    void doRecordStart({
      task: task.trim(),
      num_episodes: numEp,
      episode_time_s: episodeTime,
      reset_time_s: resetTime,
      dataset_name: datasetName.trim() || undefined,
      fps,
      use_cameras: useCameras,
    })
  }
  const busyReason = busy ? `${stateLabel[state] || state}${owner ? ` (${owner})` : ''}` : ''
  const capabilityReason = (capability: OperationCapability) =>
    capability.missing.map(translateMissing).join(' · ')
  const hwAccent = !hwStatus ? 'shadow-inset-ac' : hwStatus.ready ? 'shadow-inset-gn' : 'shadow-inset-yl'
  const camerasExist = hwStatus && hwStatus.cameras.length > 0 && hwStatus.cameras.some((c: any) => c.connected)
  const pct = targetEpisodes > 0 ? Math.round((savedEpisodes / targetEpisodes) * 100) : 0
  const replayDatasets = datasets.filter(
    (dataset) => dataset.capabilities.can_replay && !!dataset.runtime && dataset.stats.total_episodes > 0,
  )

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      {/* Error & hardware warning bars */}
      {session.error && (
        <div className="px-4 py-2 bg-rd/10 border-b border-rd/30 border-l-4 border-l-rd text-rd text-sm font-mono whitespace-pre-wrap flex items-start gap-2">
          <span className="flex-1">{session.error}</span>
          <button
            onClick={() => { void doDismissError() }}
            className="shrink-0 px-2 py-0.5 rounded text-xs font-semibold bg-rd/20 hover:bg-rd/30 transition-colors"
          >
            {t('dismissError')}
          </button>
        </div>
      )}
      {!hwReady && hwStatus && (
        <div className="px-4 py-2.5 bg-yl/8 border-b border-yl/20 text-yl text-sm font-medium">
          {hwStatus.missing.map(translateMissing).join(' · ')}
        </div>
      )}

      <div className="p-4 space-y-3">
        {/* Top row: Hardware status + Teleop + Recording */}
        <div className="flex gap-3 max-[900px]:flex-col">
          {/* Hardware status card — enhanced with embodiment state */}
          <div
            onClick={() => navigate('/settings')}
            className={`w-[200px] max-[900px]:w-full shrink-0 bg-sf rounded-lg p-3.5 cursor-pointer
              ${hwAccent} transition-all hover:shadow-card-hover animate-slide-up stagger-1`}
          >
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('arms')}</span>
                <div className="flex items-center gap-1">
                  {hwStatus?.arms.map(arm => (
                    <span key={arm.alias}
                      className={`w-2.5 h-2.5 rounded-full ring-2 ring-white ${!arm.connected ? 'bg-rd' : !arm.calibrated ? 'bg-yl' : 'bg-gn'}`}
                      title={arm.alias}
                    />
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('cameras')}</span>
                <div className="flex items-center gap-1">
                  {hwStatus?.cameras.map(cam => (
                    <span key={cam.alias}
                      className={`w-2.5 h-2.5 rounded-full ring-2 ring-white ${cam.connected ? 'bg-gn' : 'bg-rd'}`}
                      title={cam.alias}
                    />
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-2 text-2xs text-tx3 font-medium">
              {hwStatus?.ready ? t('hwReady') : `${hwStatus?.missing?.length ?? 0} ${t('warnings')}`}
            </div>

            {/* Embodiment status — local process or cross-process (agent) */}
            {busy && (
              <div className="mt-2 pt-2 border-t border-bd/40">
                <div className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full animate-pulse ${stateBadgeCls[state]?.includes('text-rd') ? 'bg-rd' : stateBadgeCls[state]?.includes('text-yl') ? 'bg-yl' : 'bg-ac'}`} />
                  <span className="text-xs font-semibold text-tx">{stateLabel[state] || state}</span>
                </div>
                <div className="text-2xs text-tx3 mt-0.5 font-mono">
                  {elapsedTick > 0 && `${elapsedTick}s`}
                  {owner && ` · ${t('embodimentSource')}: ${owner}`}
                </div>
              </div>
            )}
            {!busy && owner && owner !== 'unknown' && (
              <div className="mt-2 pt-2 border-t border-bd/40">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full animate-pulse bg-yl" />
                  <span className="text-xs font-semibold text-tx">{owner}</span>
                </div>
                <div className="text-2xs text-tx3 mt-0.5 font-mono">{t('embodimentSource')}</div>
              </div>
            )}

          </div>

          {/* Teleop */}
          <div className="w-[190px] max-[900px]:w-full shrink-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-2">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('teleoperation')}</h3>
            <div className="space-y-2">
              <ActionBtn color="ac" disabled={!ok.teleopStart || !!loading}
                onClick={() => { void doTeleopStart() }}
                title={busy ? busyReason : capabilityReason(teleopCapability) || undefined}>
                {loading === 'teleop' ? t('startingTeleop') : t('startTeleop')}
              </ActionBtn>
              <ActionBtn color="yl" disabled={!ok.teleopStop || !!loading} onClick={() => { void doTeleopStop() }}>
                {teleopStopping ? t('stoppingTeleop') : t('stopTeleop')}
              </ActionBtn>
            </div>
            {(loading === 'teleop' || state === 'teleoperating' || teleopStopping) && (
              <div className={`mt-3 flex items-center gap-2 text-xs font-medium ${teleopStopping ? 'text-yl' : 'text-ac'}`}>
                <span className={`w-2 h-2 rounded-full animate-pulse ${teleopStopping ? 'bg-yl' : 'bg-ac'}`} />
                {loading === 'teleop'
                  ? t('hwInitializing')
                  : teleopStopping
                    ? t('stoppingTeleop')
                    : t('stateTeleoperating')}
              </div>
            )}
          </div>

          {/* Record / Infer — merged card with tab switch */}
          <div className="flex-1 min-w-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-3">
            {/* Tab bar */}
            <div className="flex gap-1.5 mb-3">
              {(['record', 'infer'] as const).map((m) => {
                const label = m === 'record' ? t('recording') : t('inference')
                const active = mode === m
                const locked = (m === 'record' && state === 'inferring') || (m === 'infer' && state === 'recording')
                return (
                  <button key={m} disabled={locked}
                    onClick={() => setMode(m)}
                    className={`px-3 py-1.5 text-xs rounded-md border transition-colors font-medium
                      ${active ? 'border-ac bg-ac/5 text-ac ring-1 ring-ac/20' : 'border-bd/50 text-tx2 hover:border-ac/50'}
                      ${locked ? 'opacity-30 cursor-not-allowed' : ''}`}>
                    {label}
                  </button>
                )
              })}
            </div>

            {/* Mode-specific input */}
            {mode === 'record' ? (
              <input
                value={task}
                onChange={(e) => { setTask(e.target.value); setTaskError(false) }}
                placeholder="Pick up the red block"
                className={`w-full bg-sf2 border text-tx px-3 py-2 rounded-lg text-sm
                  focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3 mb-3
                  ${taskError ? 'border-rd animate-shake' : 'border-bd'}`}
              />
            ) : (
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono mb-3">
                {t('selectCheckpoint')}
                <select value={inferCheckpoint} onChange={(e) => setInferCheckpoint(e.target.value)}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm focus:outline-none focus:border-ac">
                  <option value="">--</option>
                  {policies.map(p => (
                    <option key={p.name} value={p.checkpoint}>
                      {p.name}{p.steps ? ` (${p.steps} steps)` : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}

            {/* Shared + mode-specific params */}
            <div className="flex gap-2 items-end flex-wrap">
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[72px]">
                {t('numEpisodes')}
                <input type="number" value={numEp} onChange={(e) => setNumEp(Number(e.target.value) || 10)} min={1}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
              </label>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[80px]">
                {t('epTime')}
                <input type="number" value={episodeTime} onChange={(e) => setEpisodeTime(Number(e.target.value) || 300)} min={1}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
              </label>
              {mode === 'record' && (
                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[80px]">
                  {t('resetTime')}
                  <input type="number" value={resetTime} onChange={(e) => setResetTime(Number(e.target.value) || 10)} min={0}
                    className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
                </label>
              )}
            </div>

            {/* Collapsible advanced options */}
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-2xs text-tx3 font-mono uppercase tracking-widest
                hover:text-tx2 transition-colors my-2"
            >
              <svg
                className={`w-3 h-3 transition-transform ${showAdvanced ? 'rotate-90' : ''}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
              {t('advancedOptions')}
            </button>

            {showAdvanced && (
              <div className="flex gap-2 items-end flex-wrap mb-3 pl-4 border-l-2 border-bd/40">
                <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1 min-w-[120px]">
                  {t('datasetName')}
                  <input
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder="rec_20260410_..."
                    className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono
                      focus:outline-none focus:border-ac placeholder:text-tx3"
                  />
                </label>
                {mode === 'record' && (
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[72px]">
                    {t('fps')}
                    <input
                      type="number" value={fps}
                      onChange={(e) => setFps(Number(e.target.value) || 30)} min={1} max={120}
                      className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono
                        focus:outline-none focus:border-ac"
                    />
                  </label>
                )}
                <label className="flex items-center gap-2 text-2xs text-tx3 font-mono cursor-pointer self-center pb-1.5">
                  <input
                    type="checkbox" checked={useCameras}
                    onChange={(e) => setUseCameras(e.target.checked)}
                    className="w-4 h-4 rounded border-bd accent-ac"
                  />
                  {t('useCameras')}
                </label>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2 justify-end">
              {mode === 'record' ? (
                <>
                  <ActionBtn color="gn" disabled={!ok.recStart || !!loading} onClick={handleRecordStart}
                    title={busy ? busyReason : capabilityReason(recordCapability) || undefined}>
                    {loading === 'record' ? t('startingRecord') : t('startRecording')}
                  </ActionBtn>
                  <ActionBtn color="rd" disabled={!ok.recStop || !!loading} onClick={() => { void doRecordStop() }}>
                    {recordStopping ? t('stoppingRecord') : t('stopRecording')}
                  </ActionBtn>
                </>
              ) : (
                <>
                  <ActionBtn color="ac" disabled={!ok.inferStart || !!loading || !inferCheckpoint}
                    onClick={() => {
                      void doInferStart({
                        checkpoint_path: inferCheckpoint,
                        num_episodes: numEp,
                        episode_time_s: episodeTime,
                      })
                    }}
                    title={busy ? busyReason : capabilityReason(inferCapability) || undefined}>
                    {loading === 'infer' ? t('startingInference') : t('startInference')}
                  </ActionBtn>
                  <ActionBtn color="yl" disabled={!ok.inferStop || !!loading} onClick={() => { void doInferStop() }}>
                    {inferStopping ? t('stoppingInference') : t('stopInference')}
                  </ActionBtn>
                </>
              )}
            </div>

            {/* Recording progress */}
            {state === 'recording' && (
              <div className="mt-3 pt-3 border-t border-bd/40">
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="font-mono text-tx2">{savedEpisodes} / {targetEpisodes}</span>
                  <span className="font-mono font-bold text-ac">{pct}%</span>
                </div>
                <div className="w-full h-2 bg-sf2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-ac2 to-ac rounded-full transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="flex gap-2 mt-3">
                  <ActionBtn color="gn" disabled={episodePhase !== 'recording'} onClick={() => { void doSaveEpisode() }}>
                    {episodePhase === 'saving' ? t('episodeSaving') : t('saveEpisode')}
                  </ActionBtn>
                  <ActionBtn color="yl" disabled={episodePhase !== 'recording'} onClick={() => { void doDiscardEpisode() }}>
                    {t('discardEpisode')}
                  </ActionBtn>
                  {episodePhase === 'resetting' && (
                    <ActionBtn color="ac" onClick={() => { void doSkipReset() }}>
                      {t('skipReset')}
                    </ActionBtn>
                  )}
                </div>
                <div className="mt-2 flex items-center gap-2 text-xs font-medium">
                  {(episodePhase === 'recording' || !episodePhase) && (
                    <><span className="w-2 h-2 rounded-full bg-ac animate-pulse" /><span className="text-ac">{t('stateRecording')}</span></>
                  )}
                  {episodePhase === 'saving' && (
                    <><span className="w-2 h-2 rounded-full bg-yl animate-pulse" /><span className="text-yl">{t('episodeSaving')}</span></>
                  )}
                  {episodePhase === 'resetting' && (
                    <><span className="w-2 h-2 rounded-full bg-yl animate-pulse" /><span className="text-yl">{t('episodeResetting')}</span></>
                  )}
                </div>
              </div>
            )}
            {recordStopping && <StatusDot color="yl" label={t('stoppingRecord')} />}

            {state === 'preparing' && mode === 'infer' && (
              <StatusDot color="yl" label={prepareStage || t('statePreparing')} />
            )}
            {state === 'inferring' && <StatusDot color="ac" label={t('stateInferring')} />}
            {inferStopping && <StatusDot color="yl" label={t('stoppingInference')} />}
          </div>
        </div>

        <div className="flex gap-3 max-[900px]:flex-col">
          <div className="flex-1 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-4">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('replay')}</h3>
            <div className="flex gap-2 items-end flex-wrap">
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1 min-w-[160px]">
                {t('selectDataset')}
                <select
                  value={replayDataset}
                  onChange={(e) => { setReplayDataset(e.target.value); setReplayEpisode(0) }}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                >
                  <option value="">--</option>
                  {replayDatasets.map(d => (
                    <option key={d.id} value={d.runtime!.name}>
                      {d.label} ({d.stats.total_episodes} ep)
                    </option>
                  ))}
                </select>
              </label>
              {(() => {
                const sel = replayDatasets.find(d => d.runtime?.name === replayDataset)
                const maxEp = (sel?.stats.total_episodes ?? 1) - 1
                return (
                  <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[90px]">
                    {t('episode')} {sel ? `(0-${maxEp})` : ''}
                    <input type="number" value={replayEpisode}
                      onChange={(e) => setReplayEpisode(Math.min(Number(e.target.value) || 0, maxEp))}
                      min={0} max={maxEp}
                      className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
                  </label>
                )
              })()}
              <div className="flex gap-2">
                <ActionBtn color="gn" disabled={!ok.replayStart || !replayDataset || !!loading}
                  onClick={() => { void doReplayStart({ dataset_name: replayDataset, episode: replayEpisode }) }}
                  title={busy ? busyReason : capabilityReason(replayCapability) || undefined}>
                  {loading === 'replay' ? t('startingReplay') : t('startReplay')}
                </ActionBtn>
                <ActionBtn color="yl" disabled={!ok.replayStop || !!loading} onClick={() => { void doReplayStop() }}>
                  {replayStopping ? t('stoppingReplay') : t('stopReplay')}
                </ActionBtn>
              </div>
            </div>
            {state === 'preparing' && owner === 'replaying' && (
              <StatusDot color="yl" label={prepareStage || t('statePreparing')} />
            )}
            {state === 'replaying' && <StatusDot color="gn" label={t('stateReplaying')} />}
            {replayStopping && <StatusDot color="yl" label={t('stoppingReplay')} />}
          </div>
        </div>

        {/* Bottom: Camera + Servo monitoring */}
        <div className="grid grid-cols-2 gap-3 min-h-[240px] max-[900px]:grid-cols-1">
          {camerasExist ? (
            <CameraPreviewPanel cameras={hwStatus!.cameras} busy={busy} />
          ) : (
            <div className="bg-sf rounded-lg p-4 shadow-card flex items-center justify-center text-sm text-tx3">
              {t('noCameraFeed')}
            </div>
          )}
          <ServoPanel state={state} />
        </div>
      </div>
    </div>
  )
}
