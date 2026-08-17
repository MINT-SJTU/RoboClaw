import { useMemo } from 'react'
import { useToast } from '@/app/shell/ToastOutlet'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useI18n } from '@/i18n'

export function AutoCalibrationPanel({ onRefresh }: { onRefresh: () => Promise<void> }) {
  const { t } = useI18n()
  const toast = useToast((state) => state.add)
  const session = useSessionStore((state) => state.session)
  const loading = useSessionStore((state) => state.loading)
  const doAutoCalibrationStart = useSessionStore((state) => state.doAutoCalibrationStart)
  const doAutoCalibrationStop = useSessionStore((state) => state.doAutoCalibrationStop)

  const isRunning = session.calibration_mode === 'auto' && (
    session.state === 'calibrating' || session.state === 'stopping'
  )
  const hasResults = session.calibration_mode === 'auto' && session.calibration_results.length > 0

  const summary = useMemo(() => {
    const counts = { success: 0, skipped: 0, failed: 0 }
    for (const item of session.calibration_results) {
      if (item.status === 'success') counts.success += 1
      if (item.status === 'skipped') counts.skipped += 1
      if (item.status === 'failed') counts.failed += 1
    }
    return counts
  }, [session.calibration_results])

  const renderReasonLabel = (reason: string) => {
    if (!reason) return ''
    if (reason === 'unsupported_arm_type') return t('autoCalReasonUnsupported')
    if (reason === 'disconnected') return t('autoCalReasonDisconnected')
    if (reason === 'manual_calibration_required') return t('autoCalReasonManualRequired')
    if (reason === 'batch_stopped') return t('autoCalReasonBatchStopped')
    if (reason === 'stopped') return t('autoCalReasonStopped')
    return reason
  }

  const renderStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      pending: 'border-bd/40 text-tx3 bg-white',
      running: 'border-ac/30 text-ac bg-ac/5',
      success: 'border-gn/30 text-gn bg-gn/5',
      skipped: 'border-yl/30 text-yl bg-yl/5',
      failed: 'border-rd/30 text-rd bg-rd/5',
    }
    const labels: Record<string, string> = {
      pending: t('autoCalStatusPending'),
      running: t('autoCalStatusRunning'),
      success: t('autoCalStatusSuccess'),
      skipped: t('autoCalStatusSkipped'),
      failed: t('autoCalStatusFailed'),
    }
    return (
      <span className={`rounded-full border px-2 py-0.5 text-2xs font-semibold ${styles[status] || styles.pending}`}>
        {labels[status] || status}
      </span>
    )
  }

  const handleStart = async () => {
    try {
      await doAutoCalibrationStart()
    } catch (error) {
      toast(error instanceof Error ? error.message : t('autoCalStartFailed'), 'e')
    }
  }

  const handleStop = async () => {
    try {
      await doAutoCalibrationStop()
      await onRefresh()
    } catch (error) {
      toast(error instanceof Error ? error.message : t('autoCalStopFailed'), 'e')
    }
  }

  return (
    <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
            {t('autoCalibrateAll')}
          </h3>
          <p className="mt-2 text-sm text-tx3">{t('autoCalibrateAllDesc')}</p>
        </div>
        {isRunning ? (
          <button
            type="button"
            onClick={handleStop}
            disabled={loading === 'auto-calibration-stop'}
            className="rounded-full border border-rd/30 bg-white px-4 py-2 text-sm font-semibold text-rd transition-all hover:bg-rd/5 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === 'auto-calibration-stop' ? t('autoCalStopping') : t('stop')}
          </button>
        ) : (
          <button
            type="button"
            onClick={handleStart}
            disabled={loading === 'auto-calibration'}
            className="rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white shadow-glow-ac transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === 'auto-calibration' ? t('autoCalStarting') : t('autoCalibrateAll')}
          </button>
        )}
      </div>

      {hasResults || isRunning ? (
        <div className="mt-5 space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-bd/30 bg-white px-4 py-3">
              <div className="text-2xs uppercase tracking-[0.18em] text-tx3">{t('autoCalProgress')}</div>
              <div className="mt-2 text-lg font-semibold text-tx">
                {session.calibration_index} / {session.calibration_total}
              </div>
            </div>
            <div className="rounded-xl border border-bd/30 bg-white px-4 py-3">
              <div className="text-2xs uppercase tracking-[0.18em] text-tx3">{t('autoCalCurrentArm')}</div>
              <div className="mt-2 text-lg font-semibold text-tx">
                {session.calibration_current_arm || t('autoCalNoCurrentArm')}
              </div>
            </div>
            <div className="rounded-xl border border-bd/30 bg-white px-4 py-3">
              <div className="text-2xs uppercase tracking-[0.18em] text-tx3">{t('autoCalPhase')}</div>
              <div className="mt-2 text-lg font-semibold capitalize text-tx">
                {session.calibration_phase || t('autoCalIdle')}
              </div>
            </div>
          </div>

          {(summary.success || summary.skipped || summary.failed) > 0 && (
            <div className="flex flex-wrap gap-2 text-2xs">
              <span className="rounded-full border border-gn/30 bg-gn/5 px-2.5 py-1 font-semibold text-gn">
                {t('autoCalStatusSuccess')}: {summary.success}
              </span>
              <span className="rounded-full border border-yl/30 bg-yl/5 px-2.5 py-1 font-semibold text-yl">
                {t('autoCalStatusSkipped')}: {summary.skipped}
              </span>
              <span className="rounded-full border border-rd/30 bg-rd/5 px-2.5 py-1 font-semibold text-rd">
                {t('autoCalStatusFailed')}: {summary.failed}
              </span>
            </div>
          )}

          <div className="space-y-2">
            {session.calibration_results.map((item) => (
              <div
                key={item.alias}
                className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-bd/30 bg-white px-4 py-3"
              >
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-tx">{item.alias}</div>
                  {item.reason && (
                    <div className="mt-1 text-xs text-tx3">
                      {renderReasonLabel(item.reason)}
                    </div>
                  )}
                </div>
                {renderStatusBadge(item.status)}
              </div>
            ))}
          </div>

          {session.calibration_error && (
            <div className="rounded-xl border border-rd/30 bg-rd/5 px-4 py-3 text-sm text-rd">
              {session.calibration_error}
            </div>
          )}
        </div>
      ) : (
        <p className="mt-5 text-sm text-tx3">{t('autoCalibrateAllHint')}</p>
      )}
    </section>
  )
}

export default AutoCalibrationPanel
