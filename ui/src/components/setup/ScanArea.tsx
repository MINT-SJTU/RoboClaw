import { useI18n } from '../../controllers/i18n'
import type { ScannedPort, ScannedCamera } from '../../controllers/setup'
import DeviceNode from './DeviceNode'

interface Props {
  ports: ScannedPort[]
  cameras: ScannedCamera[]
  scanning: boolean
}

export default function ScanArea({ ports, cameras, scanning }: Props) {
  const { t } = useI18n()

  if (scanning) {
    return (
      <div className="flex-1 min-h-[300px] rounded-xl border-2 border-dashed border-ac/30 flex items-center justify-center">
        <div className="relative">
          <div className="w-16 h-16 rounded-full border-2 border-ac animate-pulse-ring" />
          <div className="absolute inset-0 w-16 h-16 rounded-full border-2 border-ac animate-pulse-ring" style={{ animationDelay: '0.5s' }} />
          <div className="absolute inset-0 flex items-center justify-center text-ac text-sm font-medium">
            {t('scanning')}
          </div>
        </div>
      </div>
    )
  }

  if (ports.length === 0 && cameras.length === 0) {
    return (
      <div className="flex-1 min-h-[300px] rounded-xl border-2 border-dashed border-bd flex items-center justify-center">
        <span className="text-tx2 text-sm">{t('noDevicesScanned')}</span>
      </div>
    )
  }

  return (
    <div className="flex-1 min-h-[300px] rounded-xl border border-bd bg-sf/30 p-4">
      <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium mb-3">
        {t('discoveredDevices')}
      </h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {ports.map((port) => (
          <DeviceNode
            key={port.port_id}
            id={`port:${port.port_id}`}
            kind="port"
            label={port.by_id ? port.by_id.split('/').pop() || port.dev : port.dev}
            sublabel={`${port.motor_ids.length} ${t('motorsFound')}`}
            moved={port.moved}
          />
        ))}
        {cameras.map((cam) => (
          <DeviceNode
            key={`cam-${cam.index}`}
            id={`camera:${cam.index}`}
            kind="camera"
            label={cam.by_id ? cam.by_id.split('/').pop() || cam.dev : cam.dev}
            sublabel={`${cam.width}×${cam.height}`}
            previewUrl={cam.preview_url}
          />
        ))}
      </div>
    </div>
  )
}
