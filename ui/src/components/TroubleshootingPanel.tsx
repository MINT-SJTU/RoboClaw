import { useState } from 'react'
import { useDashboard } from '../controllers/dashboard'
import type { Fault, TroubleshootEntry } from '../controllers/dashboard'

interface Props {
  faults: Fault[]
  troubleshootMap: Record<string, TroubleshootEntry>
}

function downloadSnapshot(snapshot: any) {
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `fault-report-${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
}

function FaultCard({
  fault,
  entry,
}: {
  fault: Fault
  entry: TroubleshootEntry | undefined
}) {
  const { recheckFault, dismissFault, generateSnapshot } = useDashboard()
  const [rechecking, setRechecking] = useState(false)
  const [snapshotLoading, setSnapshotLoading] = useState(false)

  const title = entry?.title || fault.fault_type
  const description = entry?.description || fault.message
  const steps = entry?.steps || []
  const canRecheck = entry?.can_recheck ?? false

  async function handleRecheck() {
    setRechecking(true)
    try {
      await recheckFault(fault.fault_type, fault.device_alias)
    } finally {
      setRechecking(false)
    }
  }

  async function handleSnapshot() {
    setSnapshotLoading(true)
    try {
      const snapshot = await generateSnapshot()
      downloadSnapshot(snapshot)
    } finally {
      setSnapshotLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-yellow-500 bg-yellow-900/20 p-4 space-y-3">
      <div>
        <div className="font-semibold text-yellow-200">{title}</div>
        {fault.device_alias && (
          <div className="text-xs text-gray-400 mt-0.5">设备: {fault.device_alias}</div>
        )}
        <div className="text-sm text-gray-300 mt-1">{description}</div>
      </div>

      {steps.length > 0 && (
        <ol className="list-decimal ml-5 text-sm text-gray-300 space-y-1">
          {steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      )}

      <div className="flex gap-2 flex-wrap">
        {canRecheck && (
          <button
            onClick={handleRecheck}
            disabled={rechecking}
            className="rounded-lg bg-blue-600 px-4 py-1.5 text-sm text-white hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
          >
            {rechecking ? '检测中...' : '重新检测'}
          </button>
        )}
        <button
          onClick={() => dismissFault(fault.fault_type, fault.device_alias)}
          className="rounded-lg border border-gray-600 px-4 py-1.5 text-sm text-gray-300 hover:bg-gray-700 transition-colors"
        >
          忽略
        </button>
        <button
          onClick={handleSnapshot}
          disabled={snapshotLoading}
          className="rounded-lg border border-gray-600 px-4 py-1.5 text-sm text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed transition-colors"
        >
          {snapshotLoading ? '生成中...' : '联系技术支持'}
        </button>
      </div>
    </div>
  )
}

export default function TroubleshootingPanel({ faults, troubleshootMap }: Props) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-yellow-200">故障排查</h3>
      {faults.map((fault) => (
        <FaultCard
          key={`${fault.fault_type}-${fault.device_alias}`}
          fault={fault}
          entry={troubleshootMap[fault.fault_type]}
        />
      ))}
    </div>
  )
}
