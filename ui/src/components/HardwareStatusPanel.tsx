import { useState, useEffect } from 'react'
import type { HardwareStatus } from '../controllers/dashboard'

interface Props {
  status: HardwareStatus | null
  recordingActive: boolean
}

function ArmCard({ arm }: { arm: HardwareStatus['arms'][number] }) {
  const roleLabel = arm.role === 'leader' ? '主动臂' : '从动臂'
  const roleBadge =
    arm.role === 'leader'
      ? 'bg-ac/15 text-ac'
      : 'bg-gn/15 text-gn'

  return (
    <div className="rounded bg-sf p-4 border border-bd">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-tx">{arm.alias}</span>
        <span className={`px-2 py-0.5 rounded-sm text-2xs font-semibold tracking-wide ${roleBadge}`}>{roleLabel}</span>
      </div>
      <div className="space-y-1 text-sm text-tx2">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${arm.connected ? 'bg-gn' : 'bg-rd'}`} />
          {arm.connected ? '已连接' : '未连接'}
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${arm.calibrated ? 'bg-gn' : 'bg-yl'}`} />
          {arm.calibrated ? '已校准' : '未校准'}
        </div>
        <div className="text-xs text-tx2">{arm.type}</div>
      </div>
    </div>
  )
}

function CameraCard({
  camera,
  recordingActive,
}: {
  camera: HardwareStatus['cameras'][number]
  recordingActive: boolean
}) {
  const [previewTs, setPreviewTs] = useState(() => Date.now())

  useEffect(() => {
    if (recordingActive || !camera.connected) return
    const timer = setInterval(() => setPreviewTs(Date.now()), 2000)
    return () => clearInterval(timer)
  }, [recordingActive, camera.connected])

  return (
    <div className="rounded bg-sf p-4 border border-bd">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-tx">{camera.alias}</span>
        <span className={`inline-block w-2 h-2 rounded-full ${camera.connected ? 'bg-gn' : 'bg-rd'}`} />
      </div>
      {camera.connected && !recordingActive && (
        <>
          <div className="mb-2 rounded overflow-hidden bg-bd aspect-video">
            <img
              src={`/api/dashboard/camera-preview/${camera.alias}?t=${previewTs}`}
              alt={`${camera.alias} 预览`}
              className="w-full h-full object-cover"
            />
          </div>
          <div className="text-xs text-tx2">
            {camera.width} x {camera.height}
          </div>
        </>
      )}
      {camera.connected && recordingActive && (
        <div className="text-sm text-tx2">采集中，预览暂停</div>
      )}
      {!camera.connected && (
        <div className="text-sm text-tx2">摄像头未连接</div>
      )}
    </div>
  )
}

export default function HardwareStatusPanel({ status, recordingActive }: Props) {
  if (!status) {
    return (
      <div className="rounded bg-sf p-4 border border-bd">
        <div className="text-tx2">正在加载硬件状态...</div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Readiness banner */}
      <div
        className={`rounded-lg p-4 border ${
          status.ready
            ? 'border-gn/40 bg-gn/5'
            : 'border-rd/40 bg-rd/5'
        }`}
      >
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-3 h-3 rounded-full ${
              status.ready ? 'bg-gn' : 'bg-rd'
            }`}
          />
          <span className="font-semibold text-tx">
            {status.ready ? '可以开始数采' : '未就绪'}
          </span>
        </div>
        {!status.ready && status.missing.length > 0 && (
          <ul className="mt-2 ml-5 text-sm text-rd list-disc">
            {status.missing.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}
      </div>

      {/* Arms */}
      {status.arms.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-tx2 mb-2">机械臂</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {status.arms.map((arm) => (
              <ArmCard key={arm.alias} arm={arm} />
            ))}
          </div>
        </div>
      )}

      {/* Cameras */}
      {status.cameras.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-tx2 mb-2">摄像头</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {status.cameras.map((cam) => (
              <CameraCard
                key={cam.alias}
                camera={cam}
                recordingActive={recordingActive}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
