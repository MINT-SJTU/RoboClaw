import { useState, useEffect, useMemo } from 'react'
import { useTrainingStore, type TrainingCurve } from '@/domains/training/store/useTrainingStore'
import { useI18n } from '@/i18n'

function useChartData(curve: TrainingCurve | null) {
  return useMemo(() => {
    const pts = curve?.points ?? []
    if (!pts.length) {
      return {
        has: false, polyline: '',
        xTicks: ['0', '25', '50', '75', '100'],
        yTicks: ['1.00', '0.70', '0.40', '0.10'],
      }
    }
    const eps = pts.map(p => p.ep)
    const losses = pts.map(p => p.loss)
    const xMin = Math.min(...eps), xMax = Math.max(...eps)
    const rawYMin = Math.min(...losses), rawYMax = Math.max(...losses)
    const yPad = Math.max((rawYMax - rawYMin) * 0.15, 0.05)
    const yMin = Math.max(0, rawYMin - yPad), yMax = rawYMax + yPad
    const xSpan = Math.max(xMax - xMin, 1), ySpan = Math.max(yMax - yMin, 0.1)

    const polyline = pts.map(p => {
      const x = 8 + ((p.ep - xMin) / xSpan) * 84
      const y = 88 - ((p.loss - yMin) / ySpan) * 74
      return `${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')

    const xTicks = Array.from({ length: 5 }, (_, i) => String(Math.round(xMin + xSpan * (i / 4))))
    const yTicks = Array.from({ length: 4 }, (_, i) => (yMin + ySpan * (1 - i / 3)).toFixed(2))

    return { has: true, polyline, xTicks, yTicks }
  }, [curve?.points])
}

export function LossCurvePanel() {
  const { t } = useI18n()
  const trainCurve = useTrainingStore((state) => state.trainCurve)
  const fetchTrainCurve = useTrainingStore((state) => state.fetchTrainCurve)
  const clearTrainCurve = useTrainingStore((state) => state.clearTrainCurve)
  const [jobId, setJobId] = useState('')

  useEffect(() => {
    const id = jobId.trim()
    clearTrainCurve()
    if (!id) return
    fetchTrainCurve(id)
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') fetchTrainCurve(id)
    }, 10_000)
    return () => clearInterval(timer)
  }, [jobId])

  const chart = useChartData(trainCurve)

  return (
    <section className="bg-sf rounded-lg p-4 shadow-card flex flex-col animate-slide-up stagger-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('lossCurve')}</h3>
        <div className="text-right text-[11px] font-mono text-tx3">
          {chart.has ? (
            <>
              <div>{t('latestLoss')}: {trainCurve?.last_loss?.toFixed(3)}</div>
              <div>{t('latestEpoch')}: {trainCurve?.points[trainCurve.points.length - 1]?.ep}</div>
              <div className="mt-1">{t('bestLoss')}: {trainCurve?.best_loss?.toFixed(3)}</div>
              <div>{t('bestEpoch')}: {trainCurve?.best_ep}</div>
            </>
          ) : (
            <div className="px-2 py-1 rounded-full bg-ac/10 text-ac font-semibold">Live</div>
          )}
        </div>
      </div>

      <label className="mt-3 flex flex-col gap-1 text-2xs text-tx3 font-mono">
        {t('trainingId')}
        <input
          value={jobId}
          onChange={e => setJobId(e.target.value)}
          placeholder={t('trainingIdPlaceholder')}
          className="bg-sf2 border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac placeholder:text-tx3"
        />
      </label>

      <div className="mt-4 h-[420px] rounded-xl border border-dashed border-bd2/80 bg-gradient-to-br from-sf2/80 via-white to-ac/5 p-4 max-[900px]:h-[340px]">
        <div className="h-full flex gap-3">
          {/* Y-axis label */}
          <div className="w-8 shrink-0 flex items-center justify-center">
            <span className="text-xs font-mono uppercase tracking-[0.25em] text-tx3 [writing-mode:vertical-rl] rotate-180">
              {t('loss')}
            </span>
          </div>

          <div className="flex-1 min-w-0 flex flex-col">
            {/* Chart area */}
            <div className="relative flex-1 rounded-lg border border-bd/60 bg-white/70 overflow-hidden">
              <div
                className="absolute inset-0 opacity-80"
                style={{
                  backgroundImage: 'linear-gradient(to right, rgba(156,163,175,0.18) 1px, transparent 1px), linear-gradient(to bottom, rgba(156,163,175,0.18) 1px, transparent 1px)',
                  backgroundSize: '20% 100%, 100% 25%',
                }}
              />
              <div className="absolute left-0 top-0 bottom-0 w-px bg-tx2/25" />
              <div className="absolute left-0 right-0 bottom-0 h-px bg-tx2/25" />

              <svg className="absolute inset-0 w-full h-full text-ac/70" viewBox="0 0 100 100" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="loss-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="currentColor" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="currentColor" stopOpacity="0.9" />
                  </linearGradient>
                </defs>
                {chart.has ? (
                  <polyline points={chart.polyline} fill="none" stroke="url(#loss-grad)"
                    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                ) : (
                  <path d="M8 78 C18 65, 24 56, 34 54 S50 36, 60 40 S74 24, 92 18"
                    fill="none" stroke="url(#loss-grad)" strokeWidth="1.6" strokeLinecap="round" />
                )}
              </svg>

              {!chart.has && (
                <div className="absolute inset-x-0 bottom-5 text-center px-6">
                  <div className="text-sm font-semibold text-tx">{t('lossCurve')}</div>
                  <div className="mt-1 text-sm text-tx3">{t('lossCurvePlaceholder')}</div>
                </div>
              )}
            </div>

            {/* X ticks */}
            <div className="mt-3 px-1 flex items-center justify-between text-[11px] font-mono text-tx3">
              {chart.xTicks.map(tick => <span key={tick}>{tick}</span>)}
            </div>
            <div className="mt-1 text-center text-xs font-mono uppercase tracking-[0.25em] text-tx3">
              {t('epoch')}
            </div>
          </div>

          {/* Y ticks */}
          <div className="w-8 shrink-0 flex flex-col justify-between text-[11px] font-mono text-tx3 py-1">
            {chart.yTicks.map(tick => <span key={tick}>{tick}</span>)}
          </div>
        </div>
      </div>
    </section>
  )
}
