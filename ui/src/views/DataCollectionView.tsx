import { useEffect, useRef, useState } from 'react'
import { useDataCollection, type RobotState } from '../controllers/datacollection'
import { useI18n } from '../controllers/i18n'

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

function canDo(state: RobotState) {
  const disc = state === 'disconnected'
  const conn = state === 'connected'
  const tele = state === 'teleoperating'
  const rec = state === 'recording'
  return {
    connect: disc,
    disconnect: !disc,
    teleopStart: conn,
    teleopStop: tele,
    recStart: tele,
    recStop: rec,
    saveEp: rec,
    discardEp: rec,
  }
}

// ── Main View ─────────────────────────────────────────────────
export default function DataCollectionView() {
  const store = useDataCollection()
  const { state, stats, episodeNum, logs, datasets } = store
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
    connected: 'bg-gn/10 text-gn',
    preparing: 'bg-yl/10 text-yl',
    teleoperating: 'bg-ac/10 text-ac',
    recording: 'bg-yl/10 text-yl',
  }

  const [dsName, setDsName] = useState('')
  const [task, setTask] = useState('')
  const [fps, setFps] = useState(30)
  const [numEp, setNumEp] = useState(10)

  useEffect(() => {
    store.connectStatusWs()
    store.loadDatasets()
    store.addLog('RoboClaw UI loaded')
    return () => store.disconnectStatusWs()
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [logs])

  function handleRecordStart() {
    if (!dsName.trim() || !task.trim()) return
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
          {/* Camera panel */}
          <div className="bg-sf p-2 min-h-[200px] flex items-center justify-center flex-wrap gap-2 border-b border-bd max-[500px]:min-h-[140px]">
            <span className="text-tx2">{t('noCameraFeed')}</span>
          </div>

          {/* Control grid */}
          <div className="grid grid-cols-2 gap-3 p-4 max-[900px]:grid-cols-1">
            {/* Connection card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('connection')}</h3>
              <div className="flex gap-2 flex-wrap">
                <Btn variant="gn" disabled={!ok.connect} onClick={store.doConnect}>
                  {t('connect')}
                </Btn>
                <Btn variant="rd" disabled={!ok.disconnect} onClick={store.doDisconnect}>
                  {t('disconnect')}
                </Btn>
              </div>
            </div>

            {/* Teleop card */}
            <div className="bg-sf border border-bd rounded-lg p-4">
              <h3 className="text-xs text-tx2 uppercase tracking-wider mb-2 font-medium">{t('teleoperation')}</h3>
              <div className="flex gap-2 flex-wrap">
                <Btn variant="ac" disabled={!ok.teleopStart} onClick={store.doTeleopStart}>
                  {t('startTeleop')}
                </Btn>
                <Btn variant="yl" disabled={!ok.teleopStop} onClick={store.doTeleopStop}>
                  {t('stopTeleop')}
                </Btn>
              </div>
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
                  <Btn variant="gn" disabled={!ok.recStart} onClick={handleRecordStart}>
                    {t('startRecording')}
                  </Btn>
                  <Btn variant="rd" disabled={!ok.recStop} onClick={store.doRecordStop}>
                    {t('stopRecording')}
                  </Btn>
                </div>
              </div>

              {state === 'recording' && (
                <div className="text-center py-4 bg-bg rounded-lg border border-bd">
                  <div className="text-4xl font-bold text-ac leading-tight max-[500px]:text-3xl">
                    {episodeNum}
                  </div>
                  <div className="text-xs text-tx2 uppercase tracking-widest mt-1">
                    {t('episodesRecorded')}
                  </div>
                </div>
              )}
            </div>
          </div>
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
