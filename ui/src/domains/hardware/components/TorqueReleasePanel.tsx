import { useState } from 'react'
import { useToast } from '@/app/shell/ToastOutlet'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'
import { postJson } from '@/shared/api/client'

const HARDWARE = '/api/hardware'

interface ReleaseTorqueResponse {
  released: string[]
  skipped: Array<{ alias: string; reason: string }>
  failed: Array<{ alias: string; reason: string }>
}

export default function TorqueReleasePanel() {
  const { t } = useI18n()
  const toast = useToast((state) => state.add)
  const hardwareStatus = useHardwareStore((state) => state.hardwareStatus)
  const fetchHardwareStatus = useHardwareStore((state) => state.fetchHardwareStatus)
  const [releasing, setReleasing] = useState(false)

  const connectedArms = hardwareStatus?.arms.filter((arm) => arm.connected).length ?? 0
  const disconnectedArms = hardwareStatus?.arms.filter((arm) => !arm.connected).length ?? 0
  const sessionBusy = hardwareStatus?.session_busy ?? false

  const handleRelease = async () => {
    setReleasing(true)
    try {
      const result = await postJson(`${HARDWARE}/release-torque`) as ReleaseTorqueResponse
      await fetchHardwareStatus()
      if (result.failed.length > 0) {
        toast(t('releaseTorquePartial', {
          released: String(result.released.length),
          skipped: String(result.skipped.length),
          failed: String(result.failed.length),
        }), 'e')
        return
      }
      toast(t('releaseTorqueSuccess', { count: String(result.released.length) }), 's')
    } catch (error) {
      toast(error instanceof Error ? error.message : t('releaseTorqueFailed'), 'e')
    } finally {
      setReleasing(false)
    }
  }

  const disabled = releasing || sessionBusy || connectedArms === 0
  const helper = sessionBusy
    ? t('sessionBusy')
    : connectedArms === 0
      ? t('releaseTorqueNoConnected')
      : disconnectedArms > 0
        ? t('releaseTorqueDisconnectedHint', { count: String(disconnectedArms) })
        : t('releaseTorqueHint', { count: String(connectedArms) })

  return (
    <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
            {t('releaseTorque')}
          </h3>
          <p className="mt-2 text-sm text-tx3">{t('releaseTorqueDesc')}</p>
        </div>
        <button
          type="button"
          onClick={() => { void handleRelease() }}
          disabled={disabled}
          className="rounded-full border border-ac/25 bg-white px-4 py-2 text-sm font-semibold text-ac transition-all hover:border-ac/40 hover:bg-ac/5 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {releasing ? t('releasingTorque') : t('releaseTorque')}
        </button>
      </div>
      <p className="mt-5 text-sm text-tx3">{helper}</p>
    </section>
  )
}
