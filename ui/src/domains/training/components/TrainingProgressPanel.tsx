import { useEffect } from 'react'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'
import { useI18n } from '@/i18n'

export function TrainingProgressPanel() {
  const { t } = useI18n()
  const currentTrainJobId = useTrainingStore((state) => state.currentTrainJobId)
  const trainJobMessage = useTrainingStore((state) => state.trainJobMessage)
  const fetchTrainStatus = useTrainingStore((state) => state.fetchTrainStatus)

  useEffect(() => {
    const jobId = currentTrainJobId.trim()
    if (!jobId) {
      return
    }

    void fetchTrainStatus(jobId)
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void fetchTrainStatus(jobId)
      }
    }, 5000)

    return () => window.clearInterval(timer)
  }, [currentTrainJobId, fetchTrainStatus])

  return (
    <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-cy">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className="text-sm font-bold text-tx uppercase tracking-wide">
          {t('trainingProgress') || t('trainJobStatus')}
        </h3>
        <span className="text-[11px] font-mono text-tx3">
          {currentTrainJobId || (t('noActiveTraining') || 'No active training')}
        </span>
      </div>

      <div className="min-h-[220px] rounded-lg bg-bg border border-bd/30 p-3">
        {trainJobMessage ? (
          <pre className="text-xs text-tx2 font-mono whitespace-pre-wrap break-all">
            {trainJobMessage}
          </pre>
        ) : (
          <div className="h-full flex items-center justify-center text-sm text-tx3 text-center">
            {t('noTrainingProgress') || 'Training progress will appear here after a job starts.'}
          </div>
        )}
      </div>
    </section>
  )
}
