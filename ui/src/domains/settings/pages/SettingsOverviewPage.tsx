import { useEffect, useState } from 'react'
import SettingsPageFrame from '@/domains/settings/components/SettingsPageFrame'
import SettingsSummaryCard from '@/domains/settings/components/SettingsSummaryCard'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { fetchHfConfig, classifyHfEndpoint } from '@/domains/hub/api/hubConfigApi'
import { useI18n } from '@/i18n'
import {
  fetchProviderStatus,
  type ProviderStatusResponse,
} from '@/domains/provider/api/providerApi'
import { useSetup } from '@/domains/hardware/setup/store/useSetupStore'

function HardwareIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 18V8a2 2 0 0 1 2-2h1" />
      <path d="M15 10V5a2 2 0 0 1 2-2h1" />
      <path d="M9 6h6" />
      <path d="M6 18h12" />
      <path d="M11 18v-4a1 1 0 0 1 1-1h0a1 1 0 0 1 1 1v4" />
    </svg>
  )
}

function ProviderIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M7 9h10" />
      <path d="M7 13h4" />
    </svg>
  )
}

function HubIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 9a6 6 0 0 1 12 0c0 7-6 11-6 11S6 16 6 9Z" />
      <circle cx="12" cy="9" r="2.5" />
    </svg>
  )
}

export default function SettingsOverviewPage() {
  const { t } = useI18n()
  const { loadDevices, devices } = useSetup()
  const fetchHardwareStatus = useHardwareStore((state) => state.fetchHardwareStatus)
  const hardwareStatus = useHardwareStore((state) => state.hardwareStatus)
  const [providerStatus, setProviderStatus] = useState<ProviderStatusResponse | null>(null)
  const [hubSummary, setHubSummary] = useState({ endpoint: '', maskedToken: '', proxy: '' })

  useEffect(() => {
    loadDevices()
    fetchHardwareStatus()
    fetchProviderStatus().then(setProviderStatus).catch((error) => {
      console.warn('Failed to load provider summary', error)
    })
    fetchHfConfig().then((config) => {
      setHubSummary({
        endpoint: config.endpoint || '',
        maskedToken: config.masked_token || '',
        proxy: config.proxy || '',
      })
    }).catch((error) => {
      console.warn('Failed to load HF summary', error)
    })
  }, [fetchHardwareStatus, loadDevices])

  const warningCount = hardwareStatus?.missing.length ?? 0
  const uncalibratedCount = devices.arms.filter((arm) => !arm.calibrated).length
  const providerLabel = providerStatus?.active_provider
    ? providerStatus.providers.find((provider) => provider.name === providerStatus.active_provider)?.label || providerStatus.active_provider
    : t('providerNotConfigured')

  const hubEndpointMode = classifyHfEndpoint(hubSummary.endpoint)
  const hubEndpointLabel = hubEndpointMode === 'default'
    ? t('hfDefault')
    : hubEndpointMode === 'mirror'
      ? t('hfMirror')
      : t('hfCustomEndpoint')

  return (
    <SettingsPageFrame
      title={t('settingsTitle')}
      description={t('settingsOverviewDesc')}
    >
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        <SettingsSummaryCard
          to="/settings/hardware"
          title={t('settingsHardware')}
          description={t('settingsHardwareDesc')}
          actionLabel={t('manageHardware')}
          status={warningCount === 0 ? t('settingsStatusReady') : t('settingsStatusNeedsAttention')}
          accent={warningCount === 0 ? 'gn' : 'yl'}
          icon={<HardwareIcon />}
          metrics={[
            { label: t('configuredArms'), value: String(devices.arms.length) },
            { label: t('configuredCameras'), value: String(devices.cameras.length) },
            { label: t('hwUncalibrated'), value: String(uncalibratedCount) },
          ]}
        />

        <SettingsSummaryCard
          to="/settings/provider"
          title={t('settingsProvider')}
          description={t('settingsProviderDesc')}
          actionLabel={t('manageProvider')}
          status={providerStatus?.active_provider ? t('settingsStatusReady') : t('settingsNotConfigured')}
          accent={providerStatus?.active_provider ? 'ac' : 'yl'}
          icon={<ProviderIcon />}
          metrics={[
            { label: t('currentProvider'), value: providerLabel },
            { label: t('settingsDefaultModel'), value: providerStatus?.default_model || t('settingsNoModel') },
            { label: t('savedStatus'), value: providerStatus?.active_provider_configured ? t('saved') : t('notSaved') },
          ]}
        />

        <SettingsSummaryCard
          to="/settings/hub"
          title={t('hfConfig')}
          description={t('settingsHubDesc')}
          actionLabel={t('manageHub')}
          status={hubSummary.maskedToken ? t('saved') : t('settingsNotConfigured')}
          accent={hubSummary.maskedToken ? 'gn' : 'ac'}
          icon={<HubIcon />}
          metrics={[
            { label: t('hfEndpoint'), value: hubEndpointLabel },
            { label: t('hfToken'), value: hubSummary.maskedToken ? t('saved') : t('settingsNotConfigured') },
            { label: t('hfProxy'), value: hubSummary.proxy || t('settingsNotConfigured') },
          ]}
        />
      </div>
    </SettingsPageFrame>
  )
}
