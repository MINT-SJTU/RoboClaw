import { useI18n } from '@/i18n'
import type {
  ScannedPort,
  ScannedCamera,
} from '@/domains/hardware/setup/store/useSetupStore'
import DeviceNode from './DeviceNode'
import {
  presentCameraLabel,
  presentCameraStream,
  presentPortLabel,
} from '@/domains/hardware/setup/lib/devicePresentation'

interface Props {
  ports: ScannedPort[]
  cameras: ScannedCamera[]
  scanning: boolean
}

export default function ScanArea({ ports, cameras, scanning }: Props) {
  const { t } = useI18n()

  if (scanning) {
    return (
      <div className="min-h-[120px] rounded-lg border border-dashed border-ac/20 bg-ac/[0.02] flex items-center justify-center gap-3">
        <div className="w-5 h-5 rounded-full border-2 border-ac border-t-transparent animate-spin" />
        <span className="text-sm text-ac">{t('scanning')}</span>
      </div>
    )
  }

  if (ports.length === 0 && cameras.length === 0) {
    return (
      <div className="min-h-[120px] rounded-lg border border-dashed border-bd/50 bg-sf/30 flex items-center justify-center">
        <span className="text-tx2 text-sm">{t('noDevicesScanned')}</span>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-bd/30 bg-sf/30 p-5">
      <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium mb-3">
        {t('discoveredDevices')}
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {ports.map((port) => (
          <DeviceNode
            key={port.stable_id}
            id={`port:${port.stable_id}`}
            kind="port"
            label={presentPortLabel(port)}
            sublabel={`${port.motor_ids.length} ${t('motorsFound')}`}
            moved={port.moved}
          />
        ))}
        {cameras.map((cam) => (
          <DeviceNode
            key={cam.stable_id}
            id={`camera:${cam.stable_id}`}
            kind="camera"
            label={presentCameraLabel(cam, cameras, t)}
            sublabel={presentCameraStream(cam, t)}
            previewUrl={cam.preview_url}
          />
        ))}
      </div>
    </div>
  )
}
