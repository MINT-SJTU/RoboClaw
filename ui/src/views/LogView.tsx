import { useEffect, useRef, useState } from 'react'
import { api, postJson } from '../controllers/api'
import { useI18n } from '../controllers/i18n'

export default function LogView() {
  const { t } = useI18n()
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let active = true

    const fetchLogs = async () => {
      try {
        const data = await api('/api/session/logs')
        if (!active) return
        setLogs(Array.isArray(data.lines) ? data.lines : [])
      } catch {
        if (!active) return
      } finally {
        if (active) setLoading(false)
      }
    }

    fetchLogs()
    const timer = window.setInterval(fetchLogs, 1000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [logs])

  const clearLogs = async () => {
    try {
      await postJson('/api/session/logs/clear')
      setLogs([])
    } catch { /* ignore */ }
  }

  return (
    <div className="page-enter flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 bg-sf border-b border-bd/50">
        <h2 className="text-xl font-bold tracking-tight">{t('logs')}</h2>
        <button
          onClick={clearLogs}
          className="px-3 py-1.5 text-tx3 rounded-lg text-xs hover:text-tx2 hover:bg-bd/30 transition-colors"
        >
          {t('clear')}
        </button>
      </div>

      <div ref={logRef} className="flex-1 overflow-y-auto px-6 py-3 font-mono text-sm bg-bg">
        {!loading && logs.length === 0 && (
          <div className="text-tx3 text-center py-12 text-sm">{t('noLogs') || 'No logs yet'}</div>
        )}
        {loading && (
          <div className="text-tx3 text-center py-12 text-sm">Loading logs...</div>
        )}
        {logs.map((line, i) => (
          <div
            key={`${i}-${line.slice(0, 24)}`}
            className={`py-1 border-b border-bd/15 whitespace-pre-wrap break-words ${
              /traceback|exception|error/i.test(line)
                ? 'text-rd'
                : /success|saved|started|connected/i.test(line)
                  ? 'text-gn'
                  : 'text-tx3'
            }`}
          >
            <span className="text-tx3/50 mr-3 text-xs">{String(i + 1).padStart(4, '0')}</span>
            {line}
          </div>
        ))}
      </div>
    </div>
  )
}
