import { useState, useEffect } from 'react'
import type { HardwareStatus } from '../controllers/dashboard'

interface Props {
  status: HardwareStatus | null
  recordingActive: boolean
}

function ArmCard({ arm }: { arm: HardwareStatus['arms'][number] }) {
  const roleLabel = arm.role === 'leader' ? '主动臂' : '从动臂'
  const roleBg = arm.role === 'leader' ? 'bg-blue-600' : 'bg-green-600'

  return (
    <div className="rounded-lg bg-gray-800 p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium">{arm.alias}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${roleBg}`}>{roleLabel}</span>
      </div>
      <div className="space-y-1 text-sm text-gray-400">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${arm.connected ? 'bg-green-500' : 'bg-red-500'}`} />
          {arm.connected ? '已连接' : '未连接'}
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${arm.calibrated ? 'bg-green-500' : 'bg-yellow-500'}`} />
          {arm.calibrated ? '已校准' : '未校准'}
        </div>
        <div className="text-xs text-gray-500">{arm.type}</div>
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
    <div className="rounded-lg bg-gray-800 p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium">{camera.alias}</span>
        <span className={`inline-block w-2 h-2 rounded-full ${camera.connected ? 'bg-green-500' : 'bg-red-500'}`} />
      </div>
      {camera.connected && !recordingActive && (
        <>
          <div className="mb-2 rounded overflow-hidden bg-gray-900 aspect-video">
            <img
              src={`/api/dashboard/camera-preview/${camera.alias}?t=${previewTs}`}
              alt={`${camera.alias} 预览`}
              className="w-full h-full object-cover"
            />
          </div>
          <div className="text-xs text-gray-500">
            {camera.width} x {camera.height}
          </div>
        </>
      )}
      {camera.connected && recordingActive && (
        <div className="text-sm text-gray-500">采集中，预览暂停</div>
      )}
      {!camera.connected && (
        <div className="text-sm text-gray-500">摄像头未连接</div>
      )}
    </div>
  )
}

export default function HardwareStatusPanel({ status, recordingActive }: Props) {
  if (!status) {
    return (
      <div className="rounded-lg bg-gray-800 p-4 border border-gray-700">
        <div className="text-gray-400">正在加载硬件状态...</div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Readiness banner */}
      <div
        className={`rounded-lg p-4 border ${
          status.ready
            ? 'border-green-500/40 bg-green-500/10'
            : 'border-red-500/40 bg-red-500/10'
        }`}
      >
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-3 h-3 rounded-full ${
              status.ready ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="font-semibold">
            {status.ready ? '可以开始数采' : '未就绪'}
          </span>
        </div>
        {!status.ready && status.missing.length > 0 && (
          <ul className="mt-2 ml-5 text-sm text-red-200 list-disc">
            {status.missing.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}
      </div>

      {/* Arms */}
      {status.arms.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-2">机械臂</h3>
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
          <h3 className="text-sm font-medium text-gray-400 mb-2">摄像头</h3>
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
