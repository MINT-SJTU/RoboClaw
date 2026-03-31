import { useState } from 'react'
import { useDashboard } from '../controllers/dashboard'
import type { RecordingState, CompletionSummary } from '../controllers/dashboard'

interface Props {
  hardwareReady: boolean
  recording: RecordingState | null
  completionSummary: CompletionSummary | null
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function IdleForm({ hardwareReady }: { hardwareReady: boolean }) {
  const { startRecording } = useDashboard()
  const [task, setTask] = useState('')
  const [numEpisodes, setNumEpisodes] = useState(10)
  const [episodeTime, setEpisodeTime] = useState(60)
  const [resetTime, setResetTime] = useState(10)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  async function handleStart(e: React.FormEvent) {
    e.preventDefault()
    if (!task.trim() || !hardwareReady) return
    setSubmitting(true)
    setError('')
    try {
      await startRecording({
        task: task.trim(),
        num_episodes: numEpisodes,
        episode_time_s: episodeTime,
        reset_time_s: resetTime,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleStart} className="space-y-4">
      {error && (
        <div className="rounded border border-rd/40 bg-rd/5 p-3 text-sm text-rd">
          {error}
        </div>
      )}

      <label className="block space-y-1">
        <span className="text-sm text-tx2">任务描述</span>
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="描述要采集的任务，例如：把红色方块放到盘子里"
          rows={2}
          className="w-full rounded border border-bd bg-bg px-4 py-2 text-tx resize-none focus:outline-none focus:ring-2 focus:ring-ac"
        />
      </label>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <label className="block space-y-1">
          <span className="text-sm text-tx2">回合数</span>
          <input
            type="number"
            value={numEpisodes}
            onChange={(e) => setNumEpisodes(Number(e.target.value))}
            min={1}
            max={999}
            className="w-full rounded border border-bd bg-bg px-4 py-2 text-tx focus:outline-none focus:ring-2 focus:ring-ac"
          />
        </label>
        <label className="block space-y-1">
          <span className="text-sm text-tx2">每回合时长 (秒)</span>
          <input
            type="number"
            value={episodeTime}
            onChange={(e) => setEpisodeTime(Number(e.target.value))}
            min={5}
            max={600}
            className="w-full rounded border border-bd bg-bg px-4 py-2 text-tx focus:outline-none focus:ring-2 focus:ring-ac"
          />
        </label>
        <label className="block space-y-1">
          <span className="text-sm text-tx2">重置间隔 (秒)</span>
          <input
            type="number"
            value={resetTime}
            onChange={(e) => setResetTime(Number(e.target.value))}
            min={0}
            max={120}
            className="w-full rounded border border-bd bg-bg px-4 py-2 text-tx focus:outline-none focus:ring-2 focus:ring-ac"
          />
        </label>
      </div>

      <button
        type="submit"
        disabled={!hardwareReady || !task.trim() || submitting}
        className="px-3.5 py-1.5 border rounded text-sm bg-bg transition-colors active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed border-ac text-ac hover:bg-ac/10"
      >
        {submitting ? '启动中...' : '开始数采'}
      </button>
    </form>
  )
}

function ActiveRecording({ recording }: { recording: RecordingState }) {
  const { stopRecording } = useDashboard()
  const [stopping, setStopping] = useState(false)

  const progress =
    recording.total_episodes > 0
      ? Math.round((recording.current_episode / recording.total_episodes) * 100)
      : 0

  async function handleStop() {
    setStopping(true)
    await stopRecording()
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-tx2">
        任务: <span className="text-tx">{recording.task || recording.dataset_name}</span>
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm text-tx mb-1">
          <span>
            回合 {recording.current_episode} / {recording.total_episodes}
          </span>
          <span>{progress}%</span>
        </div>
        <div className="w-full h-3 bg-bd rounded-full overflow-hidden">
          <div
            className="h-full bg-ac rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="flex gap-6 text-sm text-tx2">
        <div>
          已用时间: <span className="text-tx">{formatElapsed(recording.elapsed_seconds)}</span>
        </div>
        <div>
          总帧数: <span className="text-tx">{recording.total_frames}</span>
        </div>
      </div>

      {recording.state === 'error' && (
        <div className="rounded border border-rd/40 bg-rd/5 p-3 text-sm text-rd">
          {recording.error_message || '录制过程中发生错误'}
        </div>
      )}

      <button
        onClick={handleStop}
        disabled={stopping}
        className="px-3.5 py-1.5 border rounded text-sm bg-bg transition-colors active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed border-rd text-rd hover:bg-rd/10"
      >
        {stopping ? '正在停止...' : '结束采集'}
      </button>
    </div>
  )
}

function CompletedSummary({ summary }: { summary: CompletionSummary }) {
  const { clearCompletion } = useDashboard()

  return (
    <div className="space-y-4">
      <div className="rounded border border-gn/40 bg-gn/5 p-4">
        <div className="font-semibold text-gn mb-2">采集完成</div>
        <div className="space-y-1 text-sm text-tx2">
          <div>
            数据集: <span className="text-tx">{summary.dataset_name}</span>
          </div>
          <div>
            完成回合: <span className="text-tx">{summary.episodes_completed}</span>
          </div>
          <div>
            总帧数: <span className="text-tx">{summary.total_frames}</span>
          </div>
          {summary.dataset_root && (
            <div>
              存储路径: <span className="text-tx2 text-xs">{summary.dataset_root}</span>
            </div>
          )}
        </div>
      </div>
      <button
        onClick={clearCompletion}
        className="px-3.5 py-1.5 border rounded text-sm bg-bg transition-colors active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed border-ac text-ac hover:bg-ac/10"
      >
        开始新采集
      </button>
    </div>
  )
}

export default function RecordingPanel({ hardwareReady, recording, completionSummary }: Props) {
  let content: React.ReactNode

  if (recording) {
    content = <ActiveRecording recording={recording} />
  } else if (completionSummary) {
    content = <CompletedSummary summary={completionSummary} />
  } else {
    content = <IdleForm hardwareReady={hardwareReady} />
  }

  return (
    <div className="rounded bg-sf border border-bd p-4">
      <h3 className="text-lg font-semibold text-tx mb-4">录制控制</h3>
      {content}
    </div>
  )
}
