import { useState } from 'react'
import { useI18n } from '@/i18n'

interface CalibrationSafetyCardProps {
  mode: 'auto' | 'manual'
  armAlias?: string
  busy?: boolean
  onConfirm: () => void | Promise<void>
  onCancel: () => void
}

export default function CalibrationSafetyCard({
  mode,
  armAlias,
  busy = false,
  onConfirm,
  onCancel,
}: CalibrationSafetyCardProps) {
  const { t } = useI18n()
  const [acknowledged, setAcknowledged] = useState(false)

  const title = mode === 'auto' ? t('calSafetyAutoTitle') : t('calSafetyManualTitle')
  const description = mode === 'auto' ? t('calSafetyAutoDesc') : t('calSafetyManualDesc')
  const confirmLabel = mode === 'auto' ? t('calSafetyAutoConfirm') : t('calSafetyManualConfirm')

  return (
    <div className="rounded-2xl border border-yl/30 bg-yl/5 p-4 text-tx shadow-card">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full bg-yl" />
        <div className="min-w-0">
          <h4 className="text-sm font-semibold text-tx">{title}</h4>
          <p className="mt-2 text-sm leading-6 text-tx2">{description}</p>
          {armAlias && (
            <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-yl/30 bg-white/80 px-3 py-1 text-2xs font-semibold text-tx2">
              <span className="uppercase tracking-[0.18em] text-tx3">{t('calSafetyArm')}</span>
              <span className="text-tx">{armAlias}</span>
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-white/70 bg-white/90 p-4">
        <div className="text-2xs font-semibold uppercase tracking-[0.18em] text-tx3">
          {t('calSafetyChecklist')}
        </div>
        <ul className="mt-3 space-y-2 text-sm leading-6 text-tx2">
          <li>{t('calSafetyZeroPose')}</li>
          <li>{t('calSafetyClearance')}</li>
          <li>{t('calSafetyCables')}</li>
        </ul>
        <label className="mt-4 flex cursor-pointer items-start gap-3 rounded-xl border border-bd/40 bg-sf px-3 py-3 text-sm text-tx">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(event) => setAcknowledged(event.target.checked)}
            className="mt-0.5 h-4 w-4 rounded border-bd/50 text-ac focus:ring-ac"
          />
          <span>{t('calSafetyAcknowledge')}</span>
        </label>
      </div>

      <div className="mt-4 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="rounded-full border border-bd/40 bg-white px-4 py-2 text-sm font-semibold text-tx2 transition-all hover:border-bd hover:text-tx disabled:cursor-not-allowed disabled:opacity-60"
        >
          {t('cancel')}
        </button>
        <button
          type="button"
          onClick={() => { void onConfirm() }}
          disabled={!acknowledged || busy}
          className="rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white shadow-glow-ac transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {confirmLabel}
        </button>
      </div>
    </div>
  )
}
