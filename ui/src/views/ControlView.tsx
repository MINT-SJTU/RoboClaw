import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDashboard, type SessionState } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'
import { CameraPreviewPanel } from '../components/CameraPreviewPanel'
import { ServoPanel } from '../components/ServoPanel'

function canDo(state: SessionState, hwReady: boolean) {
  const canStart = state === 'idle' || state === 'error'
  const tele = state === 'teleoperating'
  const rec = state === 'recording'
  const rep = state === 'replaying'
  const inf = state === 'inferring'
  return {
    teleopStart: canStart && hwReady,
    teleopStop: tele,
    recStart: (canStart || tele) && hwReady,
    recStop: rec,
    saveEp: rec,
    discardEp: rec,
    replayStart: canStart && hwReady,
    replayStop: rep,
    inferStart: canStart && hwReady,
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

export default function ControlView() {
  const store = useDashboard()
  const { session, datasets, loading, hardwareStatus: hwStatus } = store
  const { state, episode_phase: episodePhase, saved_episodes: savedEpisodes, target_episodes: targetEpisodes, embodiment_owner: owner } = session
  const hwReady = hwStatus?.ready ?? false
  const ok = canDo(state, hwReady)
  const { t } = useI18n()
  const navigate = useNavigate()

  const [task, setTask] = useState('')
  const [numEp, setNumEp] = useState(10)
  const [episodeTime, setEpisodeTime] = useState(300)
  const [resetTime, setResetTime] = useState(10)
  const [replayDataset, setReplayDataset] = useState('')
  const [replayEpisode, setReplayEpisode] = useState(0)
  const [inferCheckpoint, setInferCheckpoint] = useState('')
  const [inferSourceDs, setInferSourceDs] = useState('')
  const [inferEpisodes, setInferEpisodes] = useState(1)

  useEffect(() => {
    store.loadDatasets()
    store.fetchHardwareStatus()
    const hwInterval = setInterval(() => {
      if (document.visibilityState === 'visible') store.fetchHardwareStatus()
    }, 5000)
    return () => clearInterval(hwInterval)
  }, [])

  function handleRecordStart() {
    if (!task.trim()) { store.addLog(t('fillTaskDesc'), 'err'); return }
    store.doRecordStart({
      task: task.trim(),
      num_episodes: numEp,
      episode_time_s: episodeTime,
      reset_time_s: resetTime,
    })
  }

  const stateLabel: Record<string, string> = {
    preparing: t('hwInitializing'),
    teleoperating: t('stateTeleoperating'),
    recording: t('stateRecording'),
    replaying: t('stateReplaying'),
    inferring: t('stateInferring'),
    calibrating: t('calibrating'),
  }
  const stateBadgeCls: Record<string, string> = {
    preparing: 'bg-yl/15 text-yl border-yl/30',
    teleoperating: 'bg-ac/15 text-ac border-ac/30',
    recording: 'bg-rd/15 text-rd border-rd/30',
    replaying: 'bg-gn/15 text-gn border-gn/30',
    inferring: 'bg-ac/15 text-ac border-ac/30',
    calibrating: 'bg-yl/15 text-yl border-yl/30',
  }

  const busy = state !== 'idle' && state !== 'error'
  const busyReason = busy ? `${stateLabel[state] || state}${owner ? ` (${owner})` : ''}` : ''
  const hwAccent = !hwStatus ? 'shadow-inset-ac' : hwStatus.ready ? 'shadow-inset-gn' : 'shadow-inset-yl'
  const camerasExist = hwStatus && hwStatus.cameras.length > 0 && hwStatus.cameras.some((c: any) => c.connected)
  const pct = targetEpisodes > 0 ? Math.round((savedEpisodes / targetEpisodes) * 100) : 0

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Error & hardware warning bars */}
      {session.error && (
        <div className="px-4 py-2 bg-rd/10 border-b border-rd/30 border-l-4 border-l-rd text-rd text-sm font-mono whitespace-pre-wrap flex items-start gap-2">
          <span className="flex-1">{session.error}</span>
          <button
            onClick={store.doDismissError}
            className="shrink-0 px-2 py-0.5 rounded text-xs font-semibold bg-rd/20 hover:bg-rd/30 transition-colors"
          >
            {t('dismissError')}
          </button>
        </div>
      )}
      {!hwReady && hwStatus && (
        <div className="px-4 py-2.5 bg-yl/8 border-b border-yl/20 text-yl text-sm font-medium">
          {hwStatus.missing.join(' · ')}
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

            {/* Embodiment status */}
            {busy && (
              <div className="mt-2 pt-2 border-t border-bd/40">
                <div className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full animate-pulse ${stateBadgeCls[state]?.includes('text-rd') ? 'bg-rd' : stateBadgeCls[state]?.includes('text-yl') ? 'bg-yl' : 'bg-ac'}`} />
                  <span className="text-xs font-semibold text-tx">{stateLabel[state] || state}</span>
                </div>
                <div className="text-2xs text-tx3 mt-0.5 font-mono">
                  {session.elapsed_seconds > 0 && `${Math.round(session.elapsed_seconds)}s`}
                  {owner && ` · ${t('embodimentSource')}: ${owner}`}
                </div>
              </div>
            )}
          </div>

          {/* Teleop */}
          <div className="w-[190px] max-[900px]:w-full shrink-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-2">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('teleoperation')}</h3>
            <div className="space-y-2">
              <ActionBtn color="ac" disabled={!ok.teleopStart || !!loading} onClick={store.doTeleopStart}
                title={busy ? busyReason : undefined}>
                {loading === 'teleop' ? t('startingTeleop') : t('startTeleop')}
              </ActionBtn>
              <ActionBtn color="yl" disabled={!ok.teleopStop || !!loading} onClick={store.doTeleopStop}>
                {t('stopTeleop')}
              </ActionBtn>
            </div>
            {(loading === 'teleop' || state === 'teleoperating') && (
              <div className="mt-3 flex items-center gap-2 text-xs text-ac font-medium">
                <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
                {loading === 'teleop' ? t('hwInitializing') : t('stateTeleoperating')}
              </div>
            )}
          </div>

          {/* Recording */}
          <div className="flex-1 min-w-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-3">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('recording')}</h3>
            <input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="Pick up the red block"
              className="w-full bg-sf2 border border-bd text-tx px-3 py-2 rounded-lg text-sm
                focus:outline-none focus:border-ac focus:shadow-glow-ac placeholder:text-tx3 mb-3"
            />
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
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[80px]">
                {t('resetTime')}
                <input type="number" value={resetTime} onChange={(e) => setResetTime(Number(e.target.value) || 10)} min={0}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
              </label>
              <div className="flex gap-2 ml-auto">
                <ActionBtn color="gn" disabled={!ok.recStart || !!loading} onClick={handleRecordStart}
                  title={busy && state !== 'teleoperating' ? busyReason : undefined}>
                  {loading === 'record' ? t('startingRecord') : t('startRecording')}
                </ActionBtn>
                <ActionBtn color="rd" disabled={!ok.recStop} onClick={store.doRecordStop}>
                  {t('stopRecording')}
                </ActionBtn>
              </div>
            </div>

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
                  <ActionBtn color="gn" disabled={episodePhase !== 'recording'} onClick={store.doSaveEpisode}>
                    {episodePhase === 'saving' ? t('episodeSaving') : t('saveEpisode')}
                  </ActionBtn>
                  <ActionBtn color="yl" disabled={episodePhase !== 'recording'} onClick={store.doDiscardEpisode}>
                    {t('discardEpisode')}
                  </ActionBtn>
                  {episodePhase === 'resetting' && (
                    <ActionBtn color="ac" onClick={store.doSkipReset}>
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
          </div>
        </div>

        {/* Second row: Replay + Inference */}
        <div className="flex gap-3 max-[900px]:flex-col">
          {/* Replay */}
          <div className="w-[220px] max-[900px]:w-full shrink-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-4">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('replay')}</h3>
            <select
              value={replayDataset}
              onChange={(e) => setReplayDataset(e.target.value)}
              className="w-full bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm mb-2
                focus:outline-none focus:border-ac"
            >
              <option value="">{t('selectDataset')}</option>
              {datasets.map(d => (
                <option key={d.name} value={d.name}>{d.name}</option>
              ))}
            </select>
            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono mb-2">
              {t('episode')}
              <input type="number" value={replayEpisode} onChange={(e) => setReplayEpisode(Number(e.target.value) || 0)} min={0}
                className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
            </label>
            <div className="space-y-2">
              <ActionBtn color="gn" disabled={!ok.replayStart || !replayDataset || !!loading}
                onClick={() => store.doReplayStart({ dataset_name: replayDataset, episode: replayEpisode })}
                title={busy ? busyReason : undefined}>
                {loading === 'replay' ? t('startingReplay') : t('startReplay')}
              </ActionBtn>
              <ActionBtn color="yl" disabled={!ok.replayStop} onClick={store.doReplayStop}>
                {t('stopReplay')}
              </ActionBtn>
            </div>
          </div>

          {/* Inference */}
          <div className="flex-1 min-w-0 bg-sf rounded-lg p-3.5 shadow-card animate-slide-up stagger-5">
            <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest mb-3">{t('inference')}</h3>
            <div className="flex gap-2 mb-2">
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('selectCheckpoint')}
                <input value={inferCheckpoint} onChange={(e) => setInferCheckpoint(e.target.value)}
                  placeholder="/path/to/checkpoint"
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm focus:outline-none focus:border-ac placeholder:text-tx3" />
              </label>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('sourceDataset')}
                <select value={inferSourceDs} onChange={(e) => setInferSourceDs(e.target.value)}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm focus:outline-none focus:border-ac">
                  <option value="">--</option>
                  {datasets.map(d => (
                    <option key={d.name} value={d.name}>{d.name}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[72px]">
                {t('numEpisodes')}
                <input type="number" value={inferEpisodes} onChange={(e) => setInferEpisodes(Number(e.target.value) || 1)} min={1}
                  className="bg-sf2 border border-bd text-tx px-2 py-1.5 rounded text-sm font-mono focus:outline-none focus:border-ac" />
              </label>
            </div>
            <div className="flex gap-2">
              <ActionBtn color="ac" disabled={!ok.inferStart || !!loading}
                onClick={() => store.doInferStart({ checkpoint_path: inferCheckpoint, source_dataset: inferSourceDs, num_episodes: inferEpisodes })}
                title={busy ? busyReason : undefined}>
                {loading === 'infer' ? t('startingInference') : t('startInference')}
              </ActionBtn>
              <ActionBtn color="yl" disabled={!ok.inferStop} onClick={store.doInferStop}>
                {t('stopInference')}
              </ActionBtn>
            </div>
            {state === 'inferring' && (
              <div className="mt-2 flex items-center gap-2 text-xs text-ac font-medium">
                <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
                {t('stateInferring')}
              </div>
            )}
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
