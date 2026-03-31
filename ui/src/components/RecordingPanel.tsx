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
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
          {error}
        </div>
      )}

      <label className="block space-y-1">
        <span className="text-sm text-gray-300">任务描述</span>
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="描述要采集的任务，例如：把红色方块放到盘子里"
          rows={2}
          className="w-full rounded-lg border border-gray-600 bg-gray-900 px-4 py-2 text-white resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </label>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <label className="block space-y-1">
          <span className="text-sm text-gray-300">回合数</span>
          <input
            type="number"
            value={numEpisodes}
            onChange={(e) => setNumEpisodes(Number(e.target.value))}
            min={1}
            max={999}
            className="w-full rounded-lg border border-gray-600 bg-gray-900 px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
        <label className="block space-y-1">
          <span className="text-sm text-gray-300">每回合时长 (秒)</span>
          <input
            type="number"
            value={episodeTime}
            onChange={(e) => setEpisodeTime(Number(e.target.value))}
            min={5}
            max={600}
            className="w-full rounded-lg border border-gray-600 bg-gray-900 px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
        <label className="block space-y-1">
          <span className="text-sm text-gray-300">重置间隔 (秒)</span>
          <input
            type="number"
            value={resetTime}
            onChange={(e) => setResetTime(Number(e.target.value))}
            min={0}
            max={120}
            className="w-full rounded-lg border border-gray-600 bg-gray-900 px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
      </div>

      <button
        type="submit"
        disabled={!hardwareReady || !task.trim() || submitting}
        className="rounded-lg bg-blue-600 px-6 py-2.5 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-600 transition-colors"
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
      <div className="text-sm text-gray-400">
        任务: <span className="text-white">{recording.task || recording.dataset_name}</span>
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span>
            回合 {recording.current_episode} / {recording.total_episodes}
          </span>
          <span>{progress}%</span>
        </div>
        <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="flex gap-6 text-sm text-gray-400">
        <div>
          已用时间: <span className="text-white">{formatElapsed(recording.elapsed_seconds)}</span>
        </div>
        <div>
          总帧数: <span className="text-white">{recording.total_frames}</span>
        </div>
      </div>

      {recording.state === 'error' && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
          {recording.error_message || '录制过程中发生错误'}
        </div>
      )}

      <button
        onClick={handleStop}
        disabled={stopping}
        className="rounded-lg bg-red-600 px-6 py-2.5 text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:bg-gray-600 transition-colors"
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
      <div className="rounded-lg border border-green-500/40 bg-green-500/10 p-4">
        <div className="font-semibold text-green-200 mb-2">采集完成</div>
        <div className="space-y-1 text-sm">
          <div>
            数据集: <span className="text-white">{summary.dataset_name}</span>
          </div>
          <div>
            完成回合: <span className="text-white">{summary.episodes_completed}</span>
          </div>
          <div>
            总帧数: <span className="text-white">{summary.total_frames}</span>
          </div>
          {summary.dataset_root && (
            <div>
              存储路径: <span className="text-gray-300 text-xs">{summary.dataset_root}</span>
            </div>
          )}
        </div>
      </div>
      <button
        onClick={clearCompletion}
        className="rounded-lg bg-blue-600 px-6 py-2.5 text-white hover:bg-blue-700 transition-colors"
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
    <div className="rounded-lg bg-gray-800 border border-gray-700 p-4">
      <h3 className="text-lg font-semibold mb-4">录制控制</h3>
      {content}
    </div>
  )
}
