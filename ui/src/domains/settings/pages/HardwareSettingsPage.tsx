import { useEffect, useMemo, useState } from 'react'
import { useSetup } from '@/domains/hardware/setup/store/useSetupStore'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useI18n } from '@/i18n'
import { postJson } from '@/shared/api/client'
import DeviceList from '@/domains/hardware/setup/components/DeviceList'
import DiscoveryWizard from '@/domains/hardware/setup/components/DiscoveryWizard'
import PermissionPanel from '@/domains/hardware/setup/components/PermissionPanel'
import { TemperatureHeatMap } from '@/domains/hardware/components/TemperatureHeatMap'
import { CalibrationPanel } from '@/domains/hardware/components/CalibrationPanel'
import AutoCalibrationPanel from '@/domains/hardware/components/AutoCalibrationPanel'
import { ServoPollingToggle } from '@/domains/hardware/components/ServoPollingToggle'
import SettingsPageFrame from '@/domains/settings/components/SettingsPageFrame'

function SummaryTile({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent: 'ac' | 'yl' | 'gn'
}) {
  const accents = {
    ac: 'border-ac/20 bg-ac/5 text-ac',
    yl: 'border-yl/20 bg-yl/5 text-yl',
    gn: 'border-gn/20 bg-gn/5 text-gn',
  }

  return (
    <div className="rounded-2xl border border-bd/30 bg-white p-4 shadow-card">
      <div className="text-2xs uppercase tracking-[0.18em] text-tx3">{label}</div>
      <div className="mt-3 flex items-center gap-3">
        <span className={`rounded-full px-2.5 py-1 text-2xs font-semibold ${accents[accent]}`}>
          {value}
        </span>
      </div>
    </div>
  )
}

export default function HardwareSettingsPage() {
  const { t } = useI18n()
  const {
    wizardActive,
    startWizard,
    cancelWizard,
    loadDevices,
    loadCatalog,
    checkPermissions,
    devices,
  } = useSetup()
  const fetchHardwareStatus = useHardwareStore((state) => state.fetchHardwareStatus)
  const fetchSessionStatus = useSessionStore((state) => state.fetchSessionStatus)
  const sessionState = useSessionStore((state) => state.session.state)
  const sessionCalArm = useSessionStore((state) => state.session.calibration_arm)
  const sessionCalMode = useSessionStore((state) => state.session.calibration_mode)
  const hardwareStatus = useHardwareStore((state) => state.hardwareStatus)
  const [calibratingArm, setCalibratingArm] = useState<string | null>(null)

  useEffect(() => {
    const bootstrap = async () => {
      await loadCatalog()
      await loadDevices()
      await fetchSessionStatus()
    }
    void bootstrap()
    void fetchHardwareStatus()
    void checkPermissions()

    const hwInterval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        void fetchHardwareStatus()
      }
    }, 5000)

    return () => clearInterval(hwInterval)
  }, [checkPermissions, fetchHardwareStatus, fetchSessionStatus, loadCatalog, loadDevices])

  useEffect(() => {
    if (
      sessionCalMode === 'manual'
      && sessionState === 'calibrating'
      && sessionCalArm
      && !calibratingArm
    ) {
      setCalibratingArm(sessionCalArm)
    }
  }, [calibratingArm, sessionCalArm, sessionCalMode, sessionState])

  const uncalibratedArms = useMemo(
    () => devices.arms.filter((arm) => !arm.calibrated).length,
    [devices.arms],
  )

  const warningsCount = hardwareStatus?.missing.length ?? 0
  const calibrationSummary = devices.arms.length === 0
    ? t('settingsNoDevices')
    : uncalibratedArms === 0
      ? t('settingsAllCalibrated')
      : t('settingsUncalibratedCount', { count: String(uncalibratedArms) })
  const healthSummary = warningsCount === 0
    ? t('settingsStatusReady')
    : t('settingsWarningsCount', { count: String(warningsCount) })

  const action = wizardActive ? (
    <button
      type="button"
      onClick={() => { void cancelWizard() }}
      className="rounded-full border border-bd/40 bg-white px-4 py-2 text-sm font-semibold text-tx2 transition-all hover:border-rd/30 hover:text-rd"
    >
      {t('cancel')}
    </button>
  ) : (
    <button
      type="button"
      onClick={startWizard}
      className="rounded-full bg-ac px-4 py-2 text-sm font-semibold text-white shadow-glow-ac transition-all hover:bg-ac2"
    >
      {t('addDevice')}
    </button>
  )

  return (
    <SettingsPageFrame
      title={t('settingsHardware')}
      description={t('settingsHardwareDesc')}
      actions={action}
    >
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <SummaryTile
            label={t('configuredArms')}
            value={String(devices.arms.length)}
            accent="ac"
          />
          <SummaryTile
            label={t('configuredCameras')}
            value={String(devices.cameras.length)}
            accent="gn"
          />
          <SummaryTile
            label={t('settingsHardwareHealth')}
            value={warningsCount === 0 ? calibrationSummary : healthSummary}
            accent={warningsCount === 0 ? 'gn' : 'yl'}
          />
        </div>
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.95fr)]">
          <div className="space-y-6">
            <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                    {t('configuredDevices')}
                  </h3>
                  <p className="mt-2 text-sm text-tx3">{t('settingsHardwareListDesc')}</p>
                </div>
                {!wizardActive && (
                  <button
                    type="button"
                    onClick={startWizard}
                    className="shrink-0 rounded-full border border-ac/25 bg-white px-4 py-2 text-sm font-semibold text-ac transition-all hover:border-ac/40 hover:bg-ac/5"
                  >
                    {t('addDevice')}
                  </button>
                )}
              </div>

              <div className="mt-5">
                <DeviceList onManualCalibrate={async (alias) => {
                  setCalibratingArm(alias)
                  await postJson('/api/calibration/start', { arm_alias: alias })
                }} />
              </div>
            </section>

            {wizardActive && (
              <section className="rounded-2xl border border-ac/20 bg-sf p-5 shadow-card">
                <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
                  <div className="min-w-0">
                    <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                      {t('setupWizard')}
                    </h3>
                    <p className="mt-2 text-sm text-tx3">{t('settingsSetupDesc')}</p>
                  </div>
                </div>
                <DiscoveryWizard />
              </section>
            )}
          </div>

          <div className="space-y-6">
            <PermissionPanel onFixed={() => { void checkPermissions() }} />

            <AutoCalibrationPanel onRefresh={async () => {
              await fetchHardwareStatus()
              await loadDevices()
            }} />

            {calibratingArm ? (
              <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
                <CalibrationPanel
                  armAlias={calibratingArm}
                  onClose={() => {
                    setCalibratingArm(null)
                    void fetchHardwareStatus()
                    void loadDevices()
                  }}
                />
              </section>
            ) : (
              <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
                <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                  {t('manualCalibrate')}
                </h3>
                <p className="mt-2 text-sm text-tx3">{t('settingsCalibrationHint')}</p>
              </section>
            )}

            <section className="rounded-2xl border border-bd/30 bg-sf p-5 shadow-card">
              <div className="mb-4 flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-[0.18em] text-tx">
                    {t('servoTemperature')}
                  </h3>
                  <p className="mt-2 text-sm text-tx3">{t('settingsTemperatureDesc')}</p>
                </div>
                <ServoPollingToggle />
              </div>
              <TemperatureHeatMap />
            </section>
          </div>
        </div>
      </div>
    </SettingsPageFrame>
  )
}
