import { useEffect } from 'react'
import { useDashboard } from '../controllers/dashboard'
import HardwareStatusPanel from '../components/HardwareStatusPanel'
import RecordingPanel from '../components/RecordingPanel'
import TroubleshootingPanel from '../components/TroubleshootingPanel'

export default function DashboardView() {
  const {
    hardwareStatus,
    recording,
    completionSummary,
    activeFaults,
    troubleshootMap,
    fetchHardwareStatus,
    fetchTroubleshootMap,
  } = useDashboard()

  useEffect(() => {
    fetchTroubleshootMap()
  }, [fetchTroubleshootMap])

  useEffect(() => {
    fetchHardwareStatus()
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchHardwareStatus()
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [fetchHardwareStatus])

  return (
    <div className="flex flex-col h-full">
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <h2 className="text-xl font-semibold">数据采集</h2>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <HardwareStatusPanel
          status={hardwareStatus}
          recordingActive={recording !== null}
        />
        <RecordingPanel
          hardwareReady={hardwareStatus?.ready ?? false}
          recording={recording}
          completionSummary={completionSummary}
        />
        {activeFaults.length > 0 && troubleshootMap && (
          <TroubleshootingPanel
            faults={activeFaults}
            troubleshootMap={troubleshootMap}
          />
        )}
      </div>
    </div>
  )
}
