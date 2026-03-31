import { useCallback, useEffect, useRef, useState } from 'react'
import { useDataCollection, type RobotState } from '../controllers/datacollection'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'

// ── Servo chart colors ────────────────────────────────────────
const SERVO_COLORS = [
  '#0969da', '#1a7f37', '#cf222e', '#9a6700', '#8250df', '#bc4c00',
]
const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
const MAX_POINTS = 60

interface ServoHistory {
  [motor: string]: number[]
}

function ServoChart({ armAlias, history, busy }: { armAlias: string; history: ServoHistory; busy: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const w = canvas.width
    const h = canvas.height

    ctx.clearRect(0, 0, w, h)

    // Background grid
    ctx.strokeStyle = '#d0d7de'
    ctx.lineWidth = 0.5
    for (let y = 0; y < h; y += h / 4) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }

    // Draw each motor's line
    MOTOR_NAMES.forEach((motor, idx) => {
      const data = history[motor]
      if (!data || data.length < 2) return
      ctx.strokeStyle = SERVO_COLORS[idx]
      ctx.lineWidth = 1.5
      ctx.beginPath()
      const step = w / (MAX_POINTS - 1)
      const offset = MAX_POINTS - data.length
      for (let i = 0; i < data.length; i++) {
        const x = (offset + i) * step
        const y = h - (data[i] / 4096) * h
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    })
  }, [history])

  return (
    <div className="bg-sf border border-bd rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs text-tx2 uppercase tracking-wider font-medium">{armAlias}</h4>
        {busy && <span className="text-2xs text-yl">Serial busy</span>}
      </div>
      <canvas ref={canvasRef} width={400} height={120} className="w-full rounded bg-bg border border-bd" />
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5">
        {MOTOR_NAMES.map((name, idx) => (
          <span key={name} className="text-2xs flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: SERVO_COLORS[idx] }} />
            {name}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Ghost button ──────────────────────────────────────────────
type BtnVariant = 'gn' | 'rd' | 'yl' | 'ac'
const variantCls: Record<BtnVariant, string> = {
  gn: 'border-gn text-gn hover:bg-gn/10',
  rd: 'border-rd text-rd hover:bg-rd/10',
  yl: 'border-yl text-yl hover:bg-yl/10',
  ac: 'border-ac text-ac hover:bg-ac/10',
}

function Btn({
  children,
  disabled,
  onClick,
  variant = 'ac',
}: {
  children: React.ReactNode
  disabled?: boolean
  onClick?: () => void
  variant?: BtnVariant
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`px-3.5 py-1.5 border rounded text-sm bg-bg transition-colors active:scale-[0.97]
        disabled:opacity-30 disabled:cursor-not-allowed ${variantCls[variant]}`}
    >
      {children}
    </button>
  )
}

// ── Camera preview ────────────────────────────────────────────
function CameraPreviewPanel({
  cameras,
  enabled,
  t,
}: {
  cameras: { alias: string; connected: boolean; width: number; height: number }[]
  enabled: boolean
  t: (key: any) => string
}) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (!enabled) return
    const connected = cameras.filter((c) => c.connected)
    if (!connected.length) return
    const timer = setInterval(() => setTick((n) => n + 1), 1500)
    return () => clearInterval(timer)
  }, [cameras, enabled])

  const connected = cameras.filter((c) => c.connected)

  if (!enabled || !connected.length) {
    return (
      <div className="bg-sf p-2 min-h-[100px] flex items-center justify-center border-b border-bd">
        <span className="text-tx2">{!enabled ? t('camerasDisabled') : t('noCameraFeed')}</span>
      </div>
    )
  }

  return (
    <div className="bg-black/5 p-2 flex flex-wrap gap-2 border-b border-bd">
      {connected.map((cam) => (
        <div key={cam.alias} className="flex-1 min-w-[250px] max-w-[520px] relative bg-sf rounded-lg overflow-hidden border border-bd">
          <img
            src={`/api/dashboard/camera-preview/${cam.alias}?t=${tick}`}
            alt={cam.alias}
            className="w-full aspect-video object-contain bg-black"
          />
          <div className="absolute top-1.5 left-2 bg-black/60 text-white text-2xs px-2 py-0.5 rounded">
            {cam.alias}
          </div>
        </div>
      ))}
    </div>
  )
}

function ServoChartPanel({ state, t }: { state: RobotState; t: (key: any) => string }) {
  const [histories, setHistories] = useState<Record<string, ServoHistory>>({})
  const busy = state === 'teleoperating' || state === 'recording'

  const poll = useCallback(async () => {
    if (busy) return
    try {
      const r = await fetch('/api/embodied/servo-positions')
      const data = await r.json()
      if (data.error || !data.arms) return
      setHistories((prev) => {
        const next = { ...prev }
        for (const [alias, positions] of Object.entries(data.arms)) {
          if (typeof positions !== 'object' || (positions as any).error) continue
          const armHistory = { ...(next[alias] || {}) }
          for (const motor of MOTOR_NAMES) {
            const val = (positions as any)[motor]
            if (val == null) continue
            const arr = armHistory[motor] || []
            armHistory[motor] = [...arr.slice(-(MAX_POINTS - 1)), val]
          }
          next[alias] = armHistory
        }
        return next
      })
    } catch { /* ignore */ }
  }, [busy])

  useEffect(() => {
    if (busy) return
    poll()
    const timer = setInterval(poll, 500)
    return () => clearInterval(timer)
  }, [busy, poll])

  const armNames = Object.keys(histories)
  if (!armNames.length && !busy) {
    return (
      <div className="p-4">
        <div className="text-sm text-tx2">{t('servoLoading')}</div>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-3">
      <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium">{t('servoPositions')}</h3>
      {busy && (
        <div className="text-sm text-yl flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-yl animate-pulse" />
          {t('servoBusy')}
        </div>
      )}
      {armNames.map((alias) => (
        <ServoChart key={alias} armAlias={alias} history={histories[alias]} busy={busy} />
      ))}
    </div>
  )
}

function canDo(state: RobotState) {
  const disc = state === 'disconnected'
  const conn = state === 'connected'
  const tele = state === 'teleoperating'
  const rec = state === 'recording'
  const prep = state === 'preparing'
  return {
    connect: disc,
    disconnect: !disc && !prep,
    teleopStart: conn,
    teleopStop: tele,
    recStart: conn || tele,
    recStop: rec,
    saveEp: rec,
    discardEp: rec,
  }
}

// ── Main View ─────────────────────────────────────────────────
export default function DashboardView() {
  const store = useDataCollection()
  const { state, stats, logs, datasets, loading, currentEpisode, totalEpisodes } = store
  const { hardwareStatus: hwStatus, fetchHardwareStatus } = useDashboard()
  const ok = canDo(state)
  const logRef = useRef<HTMLDivElement>(null)
  const { t } = useI18n()

  const stateLabel: Record<RobotState, string> = {
    disconnected: t('stateDisconnected'),
    connected: t('stateConnected'),
    preparing: t('hwInitializing'),
    teleoperating: t('stateTeleoperating'),
    recording: t('stateRecording'),
  }
  const stateBadgeCls: Record<RobotState, string> = {
    disconnected: 'bg-rd/10 text-rd',
    preparing: 'bg-yl/10 text-yl',
    connected: 'bg-gn/10 text-gn',
    teleoperating: 'bg-ac/10 text-ac',
    recording: 'bg-yl/10 text-yl',
  }

  const [camerasEnabled, setCamerasEnabled] = useState(false)

  // Auto-close camera preview BEFORE teleop/record starts (release device for subprocess)
  useEffect(() => {
    if (loading === 'teleop' || loading === 'record') {
      setCamerasEnabled(false)
    }
  }, [loading])

  const [dsName, setDsName] = useState('')
  const [task, setTask] = useState('')
  const [fps, setFps] = useState(30)
  const [numEp, setNumEp] = useState(10)

  useEffect(() => {
    store.connectStatusWs()
    store.loadDatasets()
    store.addLog('RoboClaw UI loaded')
    fetchHardwareStatus()
    const hwInterval = setInterval(() => {
      if (document.visibilityState === 'visible') fetchHardwareStatus()
    }, 5000)
    return () => {
      store.disconnectStatusWs()
      clearInterval(hwInterval)
    }
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [logs])

  function handleRecordStart() {
    if (!dsName.trim()) { store.addLog(t('fillDatasetName'), 'err'); return }
    if (!task.trim()) { store.addLog(t('fillTaskDesc'), 'err'); return }
    store.doRecordStart({
      dataset_name: dsName.trim(),
      task: task.trim(),
      fps,
      num_episodes: numEp,
    })
  }

  return (
    <div className="flex flex-col h-full">
      {/* Stats bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-bd text-sm flex-wrap">
        <span className={`px-2 py-0.5 rounded-sm text-xs font-semibold ${stateBadgeCls[state]}`}>
          {stateLabel[state]}
        </span>
        <span className="text-tx2">Arms: {stats.arms}</span>
        <span className="text-tx2">FPS: {stats.fps}</span>
        <span className="text-tx2">Frames: {stats.frames}</span>
        <span className="text-tx2">Episodes: {stats.episodes}</span>
      </div>

      {/* Main layout */}
      <div className="flex-1 grid grid-cols-[1fr_320px] overflow-hidden max-[900px]:grid-cols-1">
        {/* Left: camera + controls */}
        <div className="flex flex-col overflow-y-auto">
          {/* Camera preview panel */}
          <CameraPreviewPanel cameras={hwStatus?.cameras || []} enabled={camerasEnabled && state !== 'teleoperating' && state !== 'recording'} t={t} />

          {/* Control grid */}
          <div className="grid grid-cols-2 gap-3 p-4 max-[900px]:grid-cols-1">
            {/* Arms card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('arms')}</h3>
              {hwStatus && hwStatus.arms.length > 0 ? (
                <div className="space-y-2 mb-3">
                  {hwStatus.arms.map((arm) => (
                    <div key={arm.alias} className="flex items-center gap-2 text-sm">
                      <span className={`w-2 h-2 rounded-full ${arm.connected ? 'bg-gn' : 'bg-rd'}`} />
                      <span className="font-medium text-tx">{arm.alias}</span>
                      <span className={`px-1.5 py-0.5 rounded-sm text-2xs font-semibold ${
                        arm.role === 'leader' ? 'bg-ac/10 text-ac' : 'bg-gn/10 text-gn'
                      }`}>
                        {arm.role === 'leader' ? t('leader') : t('follower')}
                      </span>
                      {arm.connected && (
                        <span className={`text-2xs ${arm.calibrated ? 'text-gn' : 'text-yl'}`}>
                          {arm.calibrated ? t('hwCalibrated') : t('hwUncalibrated')}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-tx2 mb-3">{t('noArms')}</div>
              )}
              <div className="flex gap-2 flex-wrap">
                <Btn variant="gn" disabled={!ok.connect || loading === 'connect'} onClick={store.doConnect}>
                  {loading === 'connect' ? t('connecting') : t('connect')}
                </Btn>
                <Btn variant="rd" disabled={!ok.disconnect || !!loading} onClick={store.doDisconnect}>
                  {t('disconnect')}
                </Btn>
              </div>
            </div>

            {/* Cameras card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('cameras')}</h3>
              {hwStatus && hwStatus.cameras.length > 0 ? (
                <div className="space-y-2 mb-3">
                  {hwStatus.cameras.map((cam) => (
                    <div key={cam.alias} className="flex items-center gap-2 text-sm">
                      <span className={`w-2 h-2 rounded-full ${cam.connected ? 'bg-gn' : 'bg-rd'}`} />
                      <span className="font-medium text-tx">{cam.alias}</span>
                      {cam.connected && (
                        <span className="text-2xs text-tx2">{cam.width}x{cam.height}</span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-tx2 mb-3">{t('noCameras')}</div>
              )}
              <div className="flex gap-2 flex-wrap">
                <Btn variant="gn" disabled={camerasEnabled} onClick={() => setCamerasEnabled(true)}>
                  {t('enablePreview')}
                </Btn>
                <Btn variant="rd" disabled={!camerasEnabled} onClick={() => setCamerasEnabled(false)}>
                  {t('disablePreview')}
                </Btn>
              </div>
            </div>

            {/* Teleop card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('teleoperation')}</h3>
              <div className="flex gap-2 flex-wrap">
                <Btn variant="ac" disabled={!ok.teleopStart || !!loading} onClick={store.doTeleopStart}>
                  {loading === 'teleop' ? t('startingTeleop') : t('startTeleop')}
                </Btn>
                <Btn variant="yl" disabled={!ok.teleopStop || !!loading} onClick={store.doTeleopStop}>
                  {t('stopTeleop')}
                </Btn>
              </div>
              {(loading === 'teleop' || state === 'teleoperating') && (
                <div className="mt-2 flex items-center gap-2 text-sm text-ac">
                  <span className="inline-block w-2 h-2 rounded-full bg-ac animate-pulse" />
                  {loading === 'teleop' ? t('hwInitializing') : t('stateTeleoperating')}
                </div>
              )}
            </div>

            {/* Recording card (full width) */}
            <div className="bg-sf border border-bd rounded-lg p-4 col-span-2 max-[900px]:col-span-1">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-3 font-medium">{t('recording')}</h3>

              <div className="flex gap-2 flex-wrap mb-3">
                <label className="flex flex-col gap-1 text-xs text-tx2 flex-1 min-w-[120px]">
                  {t('datasetName')}
                  <input
                    value={dsName}
                    onChange={(e) => setDsName(e.target.value)}
                    placeholder="my_dataset"
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-tx2 flex-1 min-w-[120px]">
                  {t('taskDesc')}
                  <input
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder="Pick up the red block"
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
              </div>

              <div className="flex gap-2 flex-wrap mb-3 items-end">
                <label className="flex flex-col gap-1 text-xs text-tx2 w-[80px]">
                  FPS
                  <input
                    type="number"
                    value={fps}
                    onChange={(e) => setFps(Number(e.target.value) || 30)}
                    min={1}
                    max={120}
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-tx2 w-[100px]">
                  {t('numEpisodes')}
                  <input
                    type="number"
                    value={numEp}
                    onChange={(e) => setNumEp(Number(e.target.value) || 10)}
                    min={1}
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
                <div className="flex gap-2 flex-1 justify-end">
                  <Btn variant="gn" disabled={!ok.recStart || !!loading} onClick={handleRecordStart}>
                    {loading === 'record' ? t('startingRecord') : t('startRecording')}
                  </Btn>
                  <Btn variant="rd" disabled={!ok.recStop} onClick={store.doRecordStop}>
                    {t('stopRecording')}
                  </Btn>
                </div>
              </div>

              {state === 'recording' && (
                <>
                  {/* Progress bar */}
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-tx mb-1">
                      <span>Episode {currentEpisode} / {totalEpisodes}</span>
                      <span>{totalEpisodes > 0 ? Math.round(((currentEpisode - 1) / totalEpisodes) * 100) : 0}%</span>
                    </div>
                    <div className="w-full h-2.5 bg-bd rounded-full overflow-hidden">
                      <div
                        className="h-full bg-ac rounded-full transition-all duration-500"
                        style={{ width: `${totalEpisodes > 0 ? ((currentEpisode - 1) / totalEpisodes) * 100 : 0}%` }}
                      />
                    </div>
                  </div>

                  <div className="flex gap-2 flex-wrap mb-3">
                    <Btn variant="gn" onClick={store.doSaveEpisode}>
                      {t('saveEpisode')}
                    </Btn>
                    <Btn variant="yl" onClick={store.doDiscardEpisode}>
                      {t('discardEpisode')}
                    </Btn>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-ac">
                    <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
                    {t('stateRecording')}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Servo position charts */}
          <ServoChartPanel state={state} t={t} />
        </div>

        {/* Right sidebar */}
        <div className="bg-sf border-l border-bd flex flex-col overflow-hidden max-[900px]:border-l-0 max-[900px]:border-t max-[900px]:max-h-[50vh]">
          {/* Datasets */}
          <div className="px-3 py-2.5 border-b border-bd">
            <div className="flex items-center justify-between">
              <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium">{t('datasets')}</h3>
              <button
                onClick={store.loadDatasets}
                className="px-2.5 py-0.5 border border-ac text-ac rounded-sm text-xs hover:bg-ac/10"
              >
                {t('refresh')}
              </button>
            </div>
          </div>
          <div className="overflow-y-auto flex-1 p-2">
            {datasets.length === 0 && (
              <div className="text-tx2 text-center py-4 text-sm">{t('noDatasets')}</div>
            )}
            {datasets.map((d) => (
              <div
                key={d.name}
                className="bg-bg border border-bd rounded mb-1.5 px-3 py-2 flex items-center gap-2 text-sm"
              >
                <span className="flex-1 font-semibold text-tx">{d.name}</span>
                <span className="text-tx2 text-xs whitespace-nowrap">
                  {d.total_episodes != null ? `${d.total_episodes} ep` : ''}
                  {d.total_frames != null ? ` | ${d.total_frames} fr` : ''}
                </span>
                <button
                  onClick={() => {
                    if (confirm(`${t('deleteConfirm')} "${d.name}"?`)) store.deleteDataset(d.name)
                  }}
                  className="px-2 py-0.5 border border-rd text-rd rounded-sm text-xs hover:bg-rd/10"
                >
                  {t('del')}
                </button>
              </div>
            ))}
          </div>

          {/* Log */}
          <div className="px-3 py-2.5 border-t border-bd">
            <div className="flex items-center justify-between">
              <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium">{t('log')}</h3>
              <button
                onClick={store.clearLog}
                className="px-2.5 py-0.5 border border-bd text-tx2 rounded-sm text-xs hover:bg-bd/30"
              >
                {t('clear')}
              </button>
            </div>
          </div>
          <div ref={logRef} className="flex-1 overflow-y-auto px-3 py-1 font-mono text-xs">
            {logs.map((entry, i) => (
              <div
                key={i}
                className={`py-0.5 border-b border-bd/40 ${
                  entry.cls === 'err' ? 'text-rd' : entry.cls === 'ok' ? 'text-gn' : 'text-tx2'
                }`}
              >
                <span className="text-tx2 mr-2 text-2xs">{entry.time}</span>
                {entry.message}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
