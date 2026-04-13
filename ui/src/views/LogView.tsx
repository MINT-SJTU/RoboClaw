import { useEffect, useRef } from 'react'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'

export default function LogView() {
  const { logs, clearLog } = useDashboard()
  const { t } = useI18n()
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [logs])

  return (
    <div className="page-enter flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 bg-sf border-b border-bd/50">
        <h2 className="text-xl font-bold tracking-tight">{t('logs')}</h2>
        <button
          onClick={clearLog}
          className="px-3 py-1.5 text-tx3 rounded-lg text-xs hover:text-tx2 hover:bg-bd/30 transition-colors"
        >
          {t('clear')}
        </button>
      </div>

      <div ref={logRef} className="flex-1 overflow-y-auto px-6 py-3 font-mono text-sm bg-bg">
        {logs.length === 0 && (
          <div className="text-tx3 text-center py-12 text-sm">{t('noLogs') || 'No logs yet'}</div>
        )}
        {logs.map((entry, i) => (
          <div
            key={i}
            className={`py-1 border-b border-bd/15 ${
              entry.cls === 'err' ? 'text-rd' : entry.cls === 'ok' ? 'text-gn' : 'text-tx3'
            }`}
          >
            <span className="text-tx3/50 mr-3 text-xs">{entry.time}</span>
            {entry.message}
          </div>
        ))}
      </div>
    </div>
  )
}
