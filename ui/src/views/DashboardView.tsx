import { useCallback, useEffect, useRef, useState } from 'react'
import { useDashboard, type SessionState } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'
import { postJson } from '../controllers/api'
import { CalibrationWizard } from '../components/CalibrationWizard'

// ── Servo chart ──────────────────────────────────────────────
const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
const MAX_POINTS = 60

// Each arm gets a base hue, motors within that arm get brightness variations
const BASE_HUES = [210, 140, 0, 270, 30, 330] // blue, green, red, purple, orange, magenta

function getArmPalette(armIndex: number): string[] {
  const hue = BASE_HUES[armIndex % BASE_HUES.length]
  return MOTOR_NAMES.map((_, i) => {
    const lightness = 35 + i * 8 // 35% to 75%
    return `hsl(${hue}, 70%, ${lightness}%)`
  })
}

interface ServoHistory {
  [motor: string]: number[]
}

function UnifiedServoChart({
  histories,
  armNames,
}: {
  histories: Record<string, ServoHistory>
  armNames: string[]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#e1e4e8'
    ctx.lineWidth = 0.5
    for (let y = 0; y <= h; y += h / 4) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }

    // Draw all arms
    armNames.forEach((alias, armIdx) => {
      const history = histories[alias]
      if (!history) return
      const palette = getArmPalette(armIdx)

      MOTOR_NAMES.forEach((motor, motorIdx) => {
        const data = history[motor]
        if (!data || data.length < 2) return
        ctx.strokeStyle = palette[motorIdx]
        ctx.lineWidth = 0.8
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
    })
  }, [histories, armNames])

  return (
    <div className="bg-sf border border-bd rounded-lg p-3">
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '200px' }}
        className="rounded bg-bg border border-bd"
      />
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {armNames.map((alias, armIdx) => (
          <div key={alias} className="flex items-center gap-1.5">
            <span
              className="inline-block w-3 h-1 rounded-full"
              style={{ backgroundColor: getArmPalette(armIdx)[0] }}
            />
            <span className="text-2xs text-tx2 font-medium">{alias}</span>
          </div>
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


function ServoChartPanel({ state, t }: { state: SessionState; t: (key: any) => string }) {
  const [histories, setHistories] = useState<Record<string, ServoHistory>>({})
  const busy = state === 'teleoperating' || state === 'recording' || state === 'preparing'

  const poll = useCallback(async () => {
    if (busy) return
    try {
      const r = await fetch('/api/dashboard/servo-positions')
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
        <div className="text-sm text-yl flex items-center gap-2 mb-2">
          <span className="w-2 h-2 rounded-full bg-yl animate-pulse" />
          {t('servoBusy')}
        </div>
      )}
      <UnifiedServoChart histories={histories} armNames={armNames} />
    </div>
  )
}

function CameraPreviewPanel({ cameras, busy }: { cameras: any[]; busy: boolean }) {
  const [previews, setPreviews] = useState<Record<number, string>>({})
  const [capturing, setCapturing] = useState(false)
  const { t } = useI18n()

  const capture = useCallback(async () => {
    if (busy || capturing) return
    setCapturing(true)
    try {
      await postJson('/api/dashboard/setup/camera-previews')
      const ts = Date.now()
      const map: Record<number, string> = {}
      cameras.forEach((_: any, i: number) => {
        map[i] = `/api/dashboard/setup/camera-preview/${i}?t=${ts}`
      })
      setPreviews(map)
    } catch { /* ignore */ }
    setCapturing(false)
  }, [busy, capturing, cameras])

  return (
    <div className="bg-sf border border-bd rounded-lg p-4 col-span-2 max-[900px]:col-span-1">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium">{t('cameras')}</h3>
        <button
          onClick={capture}
          disabled={busy || capturing}
          className="px-2.5 py-0.5 border border-ac text-ac rounded-sm text-xs hover:bg-ac/10 disabled:opacity-30"
        >
          {capturing ? '...' : t('refresh')}
        </button>
      </div>
      {Object.keys(previews).length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {cameras.filter((c: any) => c.connected).map((_: any, i: number) => (
            <div key={i} className="flex-1 min-w-[200px] max-w-[400px] relative bg-bg rounded-lg overflow-hidden border border-bd">
              <img
                src={previews[i] || ''}
                alt={`Camera ${i}`}
                className="w-full aspect-video object-contain bg-black"
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="text-sm text-tx2">{t('noCameraFeed')}</div>
      )}
    </div>
  )
}

function canDo(state: SessionState, hwReady: boolean) {
  const idle = state === 'idle'
  const tele = state === 'teleoperating'
  const rec = state === 'recording'
  return {
    teleopStart: idle && hwReady,
    teleopStop: tele,
    recStart: (idle || tele) && hwReady,
    recStop: rec,
    saveEp: rec,
    discardEp: rec,
  }
}

// ── Main View ─────────────────────────────────────────────────
export default function DashboardView() {
  const store = useDashboard()
  const { session, logs, datasets, loading, hardwareStatus: hwStatus, startCalibration } = store
  const [showCalibration, setShowCalibration] = useState(false)
  const { state, episode_phase: episodePhase, saved_episodes: savedEpisodes, target_episodes: targetEpisodes } = session
  const hwReady = hwStatus?.ready ?? false
  const ok = canDo(state, hwReady)
  const logRef = useRef<HTMLDivElement>(null)
  const { t } = useI18n()

  const stateLabel: Record<SessionState, string> = {
    idle: t('stateConnected'),
    preparing: t('hwInitializing'),
    teleoperating: t('stateTeleoperating'),
    recording: t('stateRecording'),
  }
  const stateBadgeCls: Record<SessionState, string> = {
    idle: 'bg-gn/10 text-gn',
    preparing: 'bg-yl/10 text-yl',
    teleoperating: 'bg-ac/10 text-ac',
    recording: 'bg-yl/10 text-yl',
  }

  const [task, setTask] = useState('')
  const [numEp, setNumEp] = useState(10)
  const [episodeTime, setEpisodeTime] = useState(300)
  const [resetTime, setResetTime] = useState(10)

  useEffect(() => {
    store.loadDatasets()
    store.addLog('RoboClaw UI loaded')
    store.fetchHardwareStatus()
    const hwInterval = setInterval(() => {
      if (document.visibilityState === 'visible') store.fetchHardwareStatus()
    }, 5000)
    return () => clearInterval(hwInterval)
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [logs])

  function handleRecordStart() {
    if (!task.trim()) { store.addLog(t('fillTaskDesc'), 'err'); return }
    store.doRecordStart({
      task: task.trim(),
      num_episodes: numEp,
      episode_time_s: episodeTime,
      reset_time_s: resetTime,
    })
  }

  return (
    <div className="flex flex-col h-full">
      {/* Stats bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-bd text-sm flex-wrap">
        <span className={`px-2 py-0.5 rounded-sm text-xs font-semibold ${stateBadgeCls[state]}`}>
          {stateLabel[state]}
        </span>
        {state === 'recording' && (
          <span className="text-tx2">{t('savedEpisodes')}: {savedEpisodes} / {targetEpisodes}</span>
        )}
      </div>

      {/* Error banner */}
      {session.error && (
        <div className="px-4 py-2 bg-rd/10 border-b border-rd/30 text-rd text-sm font-mono whitespace-pre-wrap">
          {session.error}
        </div>
      )}

      {/* Hardware readiness warning */}
      {!hwReady && hwStatus && (
        <div className="px-4 py-2 bg-yl/10 border-b border-yl/30 text-yl text-sm">
          {hwStatus.missing.join(' · ')}
        </div>
      )}

      {/* Rerun visualization */}
      {session.rerun_web_port > 0 && (state === 'teleoperating' || state === 'recording') && (
        <div className="border-b border-bd bg-black/5">
          <iframe
            src={`${location.protocol}//${location.hostname}:${session.rerun_web_port}`}
            className="w-full border-0"
            style={{ height: '400px' }}
            title="Rerun Visualization"
          />
        </div>
      )}

      {/* Main layout */}
      <div className="flex-1 grid grid-cols-[1fr_320px] overflow-hidden max-[900px]:grid-cols-1">
        {/* Left: controls */}
        <div className="flex flex-col overflow-y-auto">
          {/* Control grid */}
          <div className="grid grid-cols-2 gap-3 p-4 max-[900px]:grid-cols-1">
            {/* Arms card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('arms')}</h3>
              {hwStatus && hwStatus.arms.length > 0 ? (
                <div className="space-y-2">
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
                        <>
                          <span className={`text-2xs ${arm.calibrated ? 'text-gn' : 'text-yl'}`}>
                            {arm.calibrated ? t('hwCalibrated') : t('hwUncalibrated')}
                          </span>
                          {!arm.calibrated && (
                            <button
                              className="text-2xs text-ac hover:underline ml-1"
                              onClick={() => {
                                startCalibration(arm.alias)
                                setShowCalibration(true)
                              }}
                            >
                              校准
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-tx2">{t('noArms')}</div>
              )}
            </div>

            {/* Cameras card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('cameras')}</h3>
              {hwStatus && hwStatus.cameras.length > 0 ? (
                <div className="space-y-2">
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
                <div className="text-sm text-tx2">{t('noCameras')}</div>
              )}
            </div>

            {/* Camera preview (idle only) */}
            {state === 'idle' && hwStatus && hwStatus.cameras.some((c: any) => c.connected) && (
              <CameraPreviewPanel cameras={hwStatus.cameras} busy={session.state !== 'idle'} />
            )}

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
                <label className="flex flex-col gap-1 text-xs text-tx2 flex-1 min-w-[160px]">
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
                <label className="flex flex-col gap-1 text-xs text-tx2 w-[90px]">
                  {t('numEpisodes')}
                  <input
                    type="number"
                    value={numEp}
                    onChange={(e) => setNumEp(Number(e.target.value) || 10)}
                    min={1}
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-tx2 w-[100px]">
                  Ep time (s)
                  <input
                    type="number"
                    value={episodeTime}
                    onChange={(e) => setEpisodeTime(Number(e.target.value) || 300)}
                    min={1}
                    className="bg-bg border border-bd text-tx px-3 py-1.5 rounded text-sm focus:outline-none focus:border-ac"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-tx2 w-[100px]">
                  Reset (s)
                  <input
                    type="number"
                    value={resetTime}
                    onChange={(e) => setResetTime(Number(e.target.value) || 10)}
                    min={0}
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
                      <span>{t('savedEpisodes')}: {savedEpisodes} / {targetEpisodes}</span>
                      <span>{targetEpisodes > 0 ? Math.round((savedEpisodes / targetEpisodes) * 100) : 0}%</span>
                    </div>
                    <div className="w-full h-2.5 bg-bd rounded-full overflow-hidden">
                      <div
                        className="h-full bg-ac rounded-full transition-all duration-500"
                        style={{ width: `${targetEpisodes > 0 ? (savedEpisodes / targetEpisodes) * 100 : 0}%` }}
                      />
                    </div>
                  </div>

                  <div className="flex gap-2 flex-wrap mb-3">
                    <Btn variant="gn" disabled={episodePhase !== 'recording'} onClick={store.doSaveEpisode}>
                      {episodePhase === 'saving' ? t('episodeSaving') : t('saveEpisode')}
                    </Btn>
                    <Btn variant="yl" disabled={episodePhase !== 'recording'} onClick={store.doDiscardEpisode}>
                      {t('discardEpisode')}
                    </Btn>
                    {episodePhase === 'resetting' && (
                      <Btn variant="ac" onClick={store.doSkipReset}>
                        {t('skipReset')}
                      </Btn>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    {episodePhase === 'recording' && (
                      <>
                        <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
                        <span className="text-ac">{t('stateRecording')}</span>
                      </>
                    )}
                    {episodePhase === 'saving' && (
                      <>
                        <span className="w-2 h-2 rounded-full bg-yl animate-pulse" />
                        <span className="text-yl">{t('episodeSaving')}</span>
                      </>
                    )}
                    {episodePhase === 'resetting' && (
                      <>
                        <span className="w-2 h-2 rounded-full bg-yl animate-pulse" />
                        <span className="text-yl">{t('episodeResetting')}</span>
                      </>
                    )}
                    {!episodePhase && (
                      <>
                        <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
                        <span className="text-ac">{t('stateRecording')}</span>
                      </>
                    )}
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

      {showCalibration && (
        <CalibrationWizard onClose={() => { setShowCalibration(false); store.fetchHardwareStatus() }} />
      )}
    </div>
  )
}
