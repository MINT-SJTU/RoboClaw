import { useEffect, useMemo } from 'react'
import { useToast } from '@/app/shell/ToastOutlet'
import { useRecoveryStore, type RecoveryFault } from '@/domains/recovery/store/useRecoveryStore'
import { useI18n } from '@/i18n'

const FAULT_I18N: Record<string, { title: string; desc: string }> = {
  arm_disconnected: {
    title: 'troubleshootArmDisconnectedTitle',
    desc: 'troubleshootArmDisconnectedDesc',
  },
  arm_timeout: {
    title: 'troubleshootArmTimeoutTitle',
    desc: 'troubleshootArmTimeoutDesc',
  },
  arm_not_calibrated: {
    title: 'troubleshootArmNotCalibratedTitle',
    desc: 'troubleshootArmNotCalibratedDesc',
  },
  camera_disconnected: {
    title: 'troubleshootCameraDisconnectedTitle',
    desc: 'troubleshootCameraDisconnectedDesc',
  },
  camera_frame_drop: {
    title: 'troubleshootCameraFrameDropTitle',
    desc: 'troubleshootCameraFrameDropDesc',
  },
  record_crashed: {
    title: 'troubleshootRecordCrashedTitle',
    desc: 'troubleshootRecordCrashedDesc',
  },
}

function buildStepKeys(fault: RecoveryFault, stepCount: number): string[] {
  const prefix = FAULT_I18N[fault.fault_type]?.title.replace('Title', 'Step')
  if (!prefix) {
    return []
  }
  return Array.from({ length: stepCount }, (_, index) => `${prefix}${index + 1}`)
}

export default function RecoveryCenterPage() {
  const { t } = useI18n()
  const toast = useToast((state) => state.add)
  const faults = useRecoveryStore((state) => state.faults)
  const guides = useRecoveryStore((state) => state.guides)
  const rechecking = useRecoveryStore((state) => state.rechecking)
  const restarting = useRecoveryStore((state) => state.restarting)
  const fetchFaults = useRecoveryStore((state) => state.fetchFaults)
  const fetchGuides = useRecoveryStore((state) => state.fetchGuides)
  const recheckHardware = useRecoveryStore((state) => state.recheckHardware)
  const restartDashboard = useRecoveryStore((state) => state.restartDashboard)

  useEffect(() => {
    void fetchFaults()
    void fetchGuides()
  }, [fetchFaults, fetchGuides])

  const visibleFaults = useMemo(
    () => faults.slice().sort((left, right) => right.timestamp - left.timestamp),
    [faults],
  )

  async function handleRecheck(): Promise<void> {
    try {
      const nextFaults = await recheckHardware()
      if (nextFaults.length === 0) {
        toast(t('troubleshootRecovered'), 's')
        return
      }
      toast(t('troubleshootStillFailing'), 'e')
    } catch (error) {
      toast(error instanceof Error ? error.message : t('troubleshootRecheckFailed'), 'e')
    }
  }

  async function handleRestart(): Promise<void> {
    try {
      await restartDashboard()
    } catch (error) {
      toast(error instanceof Error ? error.message : t('recoveryRestartFailed'), 'e')
    }
  }

  return (
    <div className="page-enter flex h-full flex-col overflow-y-auto">
      <div className="border-b border-bd/50 bg-sf">
        <div className="w-full px-6 py-5 2xl:px-10">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="text-2xs font-semibold uppercase tracking-[0.22em] text-tx3">
                {t('recoveryNav')}
              </div>
              <h2 className="mt-2 text-2xl font-bold tracking-tight text-tx">{t('recoveryTitle')}</h2>
              <p className="mt-2 max-w-3xl text-sm text-tx3">{t('recoveryDesc')}</p>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => { void handleRecheck() }}
                disabled={rechecking || restarting}
                className="rounded-full border border-bd/40 bg-white px-4 py-2 text-sm font-semibold text-tx2 transition-all hover:border-ac/30 hover:text-ac disabled:cursor-not-allowed disabled:opacity-50"
              >
                {rechecking ? t('troubleshootRechecking') : t('recoveryRecheckAll')}
              </button>
              <button
                type="button"
                onClick={() => { void handleRestart() }}
                disabled={restarting}
                className="rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white shadow-glow-ac transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {restarting ? t('recoveryRestarting') : t('recoveryRestartDashboard')}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 w-full px-6 py-6 2xl:px-10">
        <div className="space-y-6">
          <section className="rounded-2xl border border-ac/20 bg-ac/5 p-5 shadow-card">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-2xs font-semibold uppercase tracking-[0.18em] text-ac">
                  {t('recoveryPrimaryAction')}
                </div>
                <h3 className="mt-2 text-lg font-semibold text-tx">{t('recoveryRestartCardTitle')}</h3>
                <p className="mt-2 max-w-2xl text-sm text-tx3">{t('recoveryRestartCardDesc')}</p>
              </div>
            </div>
          </section>

          {visibleFaults.length === 0 ? (
            <section className="rounded-2xl border border-gn/20 bg-gn/5 p-6 shadow-card">
              <div className="text-sm font-semibold text-gn">{t('recoveryNoFaultsTitle')}</div>
              <p className="mt-2 text-sm text-tx3">{t('recoveryNoFaultsDesc')}</p>
            </section>
          ) : (
            <section className="space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                    {t('recoveryActiveFaults')}
                  </h3>
                  <p className="mt-2 text-sm text-tx3">
                    {t('recoveryFaultCount', { count: String(visibleFaults.length) })}
                  </p>
                </div>
              </div>

              {visibleFaults.map((fault) => {
                const guide = guides?.[fault.fault_type]
                const labels = FAULT_I18N[fault.fault_type]
                const title = labels ? t(labels.title as never) : fault.fault_type
                const description = labels ? t(labels.desc as never) : fault.message
                const steps = buildStepKeys(fault, guide?.step_count ?? 0).flatMap((stepKey) => {
                  const step = t(stepKey as never)
                  return step === stepKey ? [] : [step]
                })

                return (
                  <article key={`${fault.fault_type}:${fault.device_alias}`} className="rounded-2xl border border-yl/20 bg-white p-5 shadow-card">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="rounded-full border border-yl/30 bg-yl/10 px-2.5 py-1 text-2xs font-semibold uppercase tracking-[0.18em] text-yl">
                            {fault.device_alias}
                          </span>
                          <span className="text-sm font-semibold text-tx">{title}</span>
                        </div>
                        <p className="mt-2 text-sm text-tx3">{description}</p>
                        <p className="mt-2 font-mono text-xs text-tx3">{fault.message}</p>
                      </div>
                    </div>

                    {steps.length > 0 && (
                      <ol className="mt-4 space-y-2 text-sm text-tx2">
                        {steps.map((step, index) => (
                          <li key={`${fault.fault_type}:${fault.device_alias}:${index}`} className="flex gap-2">
                            <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-yl/10 text-2xs font-semibold text-yl">
                              {index + 1}
                            </span>
                            <span>{step}</span>
                          </li>
                        ))}
                      </ol>
                    )}

                    {guide?.can_recheck && (
                      <div className="mt-4">
                        <span className="rounded-full border border-ac/20 bg-ac/5 px-3 py-1.5 text-xs font-semibold text-ac">
                          {t('recoveryRecheckHint')}
                        </span>
                      </div>
                    )}
                  </article>
                )
              })}
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
