import { useEffect, useMemo } from 'react'
import { useToast } from '@/app/shell/ToastOutlet'
import { useRecoveryStore } from '@/domains/recovery/store/useRecoveryStore'
import { useSetup } from '@/domains/hardware/setup/store/useSetupStore'
import { useI18n } from '@/i18n'

export default function RecoveryCenterPage() {
  const { t } = useI18n()
  const toast = useToast((state) => state.add)
  const faults = useRecoveryStore((state) => state.faults)
  const hasCheckedHardware = useRecoveryStore((state) => state.hasCheckedHardware)
  const checkingHardware = useRecoveryStore((state) => state.checkingHardware)
  const restarting = useRecoveryStore((state) => state.restarting)
  const checkHardware = useRecoveryStore((state) => state.checkHardware)
  const restartDashboard = useRecoveryStore((state) => state.restartDashboard)
  const devices = useSetup((state) => state.devices)
  const loadDevices = useSetup((state) => state.loadDevices)

  useEffect(() => {
    void loadDevices()
  }, [loadDevices])

  const hardwareRows = useMemo(
    () => [
      ...devices.arms.map((arm) => ({
        key: `arm:${arm.alias}`,
        kind: 'arm' as const,
        alias: arm.alias,
        badge: arm.type,
      })),
      ...devices.cameras.map((camera) => ({
        key: `camera:${camera.alias}`,
        kind: 'camera' as const,
        alias: camera.alias,
        badge: camera.port,
      })),
    ],
    [devices.arms, devices.cameras],
  )
  const faultMap = useMemo(
    () => new Map(faults.map((fault) => [`${fault.fault_type}:${fault.device_alias}`, fault])),
    [faults],
  )

  async function handleRestart(): Promise<void> {
    try {
      await restartDashboard()
    } catch (error) {
      toast(error instanceof Error ? error.message : t('recoveryRestartFailed'), 'e')
    }
  }

  async function handleHardwareCheck(): Promise<void> {
    try {
      await checkHardware()
    } catch (error) {
      toast(error instanceof Error ? error.message : t('recoveryCheckHardwareFailed'), 'e')
    }
  }

  function faultFor(faultType: string, alias: string) {
    return faultMap.get(`${faultType}:${alias}`)
  }

  function statusText(ok: boolean): string {
    return ok ? t('recoveryStatusNormal') : t('recoveryStatusAbnormal')
  }

  function statusTone(ok: boolean): string {
    return ok ? 'text-gn' : 'text-rd'
  }

  function motorStatus(alias: string): { text: string; tone: string } {
    if (!hasCheckedHardware) {
      return { text: '--', tone: 'text-tx3' }
    }
    const fault = faultFor('arm_motor_disconnected', alias)
    if (!fault) {
      return { text: t('recoveryStatusNormal'), tone: 'text-gn' }
    }
    return {
      text: t('recoveryMotorFaultDetail' as never, { motors: fault.message } as never),
      tone: 'text-rd',
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

          <section className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                  {t('recoveryActiveFaults')}
                </h3>
                <p className="mt-2 text-sm text-tx3">
                  {t('recoveryFaultCount', { count: String(faults.length) })}
                </p>
              </div>
            </div>
            <div>
              <button
                type="button"
                onClick={() => { void handleHardwareCheck() }}
                disabled={checkingHardware}
                className="rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white shadow-glow-ac transition-all hover:bg-ac2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {checkingHardware ? t('recoveryCheckingHardware') : t('recoveryCheckHardware')}
              </button>
            </div>

            {hardwareRows.length === 0 ? (
              <section className="rounded-2xl border border-bd/50 bg-white p-6 shadow-card">
                <div className="text-sm text-tx3">{t('noConfiguredDevices')}</div>
              </section>
            ) : (
              <section className="rounded-2xl border border-bd/50 bg-white p-5 shadow-card">
                <div className="space-y-3">
                  {hardwareRows.map((device) => {
                    if (!hasCheckedHardware) {
                      return (
                        <div
                          key={device.key}
                          className="rounded-xl border border-bd/40 bg-sf px-4 py-3"
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold text-tx">{device.alias}</span>
                            <span className="rounded bg-white px-1.5 py-0.5 text-2xs font-mono text-tx2 border border-bd/40">
                              {device.badge}
                            </span>
                          </div>
                          <div className="mt-3 grid gap-2 text-sm text-tx2 md:grid-cols-3">
                            <div><span className="text-tx3">{t('recoverySerialConnection')}：</span><span className="text-tx3">--</span></div>
                            {device.kind === 'arm' ? (
                              <>
                                <div><span className="text-tx3">{t('recoveryCalibrationStatus')}：</span><span className="text-tx3">--</span></div>
                                <div><span className="text-tx3">{t('recoveryMotorWiring')}：</span><span className="text-tx3">--</span></div>
                              </>
                            ) : (
                              <div><span className="text-tx3">{t('camera')}：</span><span className="text-tx3">--</span></div>
                            )}
                          </div>
                        </div>
                      )
                    }

                    const serialOk = !faultFor(
                      device.kind === 'arm' ? 'arm_disconnected' : 'camera_disconnected',
                      device.alias,
                    )
                    const serialText = statusText(serialOk)
                    const serialTone = statusTone(serialOk)

                    return (
                      <div
                        key={device.key}
                        className="rounded-xl border border-bd/40 bg-sf px-4 py-3"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-tx">{device.alias}</span>
                          <span className="rounded bg-white px-1.5 py-0.5 text-2xs font-mono text-tx2 border border-bd/40">
                            {device.badge}
                          </span>
                        </div>
                        <div className="mt-3 grid gap-2 text-sm text-tx2 md:grid-cols-3">
                          <div>
                            <span className="text-tx3">{t('recoverySerialConnection')}：</span>
                            <span className={serialTone}>{serialText}</span>
                          </div>

                          {device.kind === 'arm' ? (
                            <>
                              <div>
                                <span className="text-tx3">{t('recoveryCalibrationStatus')}：</span>
                                <span className={faultFor('arm_not_calibrated', device.alias) ? 'text-rd' : 'text-gn'}>
                                  {faultFor('arm_not_calibrated', device.alias) ? t('hwUncalibrated') : t('hwCalibrated')}
                                </span>
                              </div>
                              <div>
                                <span className="text-tx3">{t('recoveryMotorWiring')}：</span>
                                <span className={motorStatus(device.alias).tone}>{motorStatus(device.alias).text}</span>
                              </div>
                            </>
                          ) : (
                            <div>
                              <span className="text-tx3">{t('camera')}：</span>
                              <span className={serialTone}>{serialText}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>

                {hasCheckedHardware && faults.length === 0 && (
                  <div className="mt-4 rounded-xl border border-gn/20 bg-gn/5 px-4 py-3 text-sm text-gn">
                    {t('recoveryNoFaultsDesc')}
                  </div>
                )}
              </section>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
