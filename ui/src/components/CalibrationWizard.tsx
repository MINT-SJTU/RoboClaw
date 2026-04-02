import { useEffect, useRef, useState } from 'react'
import { useDashboard } from '../controllers/dashboard'

const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

export function CalibrationWizard({ onClose }: { onClose: () => void }) {
  const {
    calibration: cal,
    setCalibrationHoming,
    pollCalibrationPositions,
    finishCalibration,
    cancelCalibration,
  } = useDashboard()
  const [polling, setPolling] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval>>()

  // Start polling positions when in recording state
  useEffect(() => {
    if (cal.state === 'recording' && polling) {
      pollRef.current = setInterval(() => {
        pollCalibrationPositions()
      }, 100) // 10Hz
      return () => clearInterval(pollRef.current)
    }
  }, [cal.state, polling, pollCalibrationPositions])

  const handleSetHoming = async () => {
    await setCalibrationHoming()
    setPolling(true)
  }

  const handleFinish = async () => {
    setPolling(false)
    clearInterval(pollRef.current)
    await finishCalibration()
  }

  const handleCancel = async () => {
    setPolling(false)
    clearInterval(pollRef.current)
    await cancelCalibration()
    onClose()
  }

  if (cal.state === 'done') {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-sf rounded-lg shadow-xl p-6 w-[480px]">
          <h2 className="text-lg font-bold text-tx mb-4">✓ 校准完成</h2>
          <p className="text-sm text-tx2 mb-4">
            臂 <strong>{cal.arm_alias}</strong> 校准数据已保存。
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-ac text-white text-sm hover:bg-ac/90"
          >
            关闭
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-sf rounded-lg shadow-xl p-6 w-[560px] max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold text-tx">
            校准: {cal.arm_alias}
          </h2>
          <button onClick={handleCancel} className="text-tx2 hover:text-tx text-xl">×</button>
        </div>

        {cal.error && (
          <div className="bg-rd/10 text-rd text-sm p-3 rounded mb-4">{cal.error}</div>
        )}

        {/* Step 1: Set middle position */}
        {cal.state === 'connected' && (
          <div>
            <div className="bg-ac/5 border border-ac/20 rounded p-4 mb-4">
              <h3 className="font-semibold text-tx mb-2">步骤 1：设定中位</h3>
              <p className="text-sm text-tx2">
                把机械臂的每个关节都摆到行程的中间位置，然后点击"确认中位"。
              </p>
            </div>
            <button
              onClick={handleSetHoming}
              className="px-4 py-2 rounded bg-ac text-white text-sm hover:bg-ac/90"
            >
              确认中位
            </button>
          </div>
        )}

        {/* Step 2: Record range */}
        {cal.state === 'recording' && (
          <div>
            <div className="bg-gn/5 border border-gn/20 rounded p-4 mb-4">
              <h3 className="font-semibold text-tx mb-2">步骤 2：录制关节范围</h3>
              <p className="text-sm text-tx2">
                逐个晃动每个关节到两端极限位置。下方表格实时显示各关节的最小值、当前值、最大值。
                <br />完成后点击"保存校准"。
              </p>
            </div>

            {/* Live position table */}
            <div className="overflow-x-auto mb-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-bd">
                    <th className="text-left py-1.5 text-tx2 font-medium">关节</th>
                    <th className="text-right py-1.5 text-tx2 font-medium">最小</th>
                    <th className="text-right py-1.5 text-tx2 font-medium">当前</th>
                    <th className="text-right py-1.5 text-tx2 font-medium">最大</th>
                    <th className="text-right py-1.5 text-tx2 font-medium">范围</th>
                  </tr>
                </thead>
                <tbody>
                  {MOTOR_NAMES.filter(m => m !== 'wrist_roll').map(motor => {
                    const pos = cal.positions?.[motor] ?? '-'
                    const min = cal.mins?.[motor] ?? '-'
                    const max = cal.maxes?.[motor] ?? '-'
                    const range = typeof min === 'number' && typeof max === 'number'
                      ? max - min : '-'
                    const moved = typeof range === 'number' && range > 50
                    return (
                      <tr key={motor} className="border-b border-bd/50">
                        <td className="py-1.5 text-tx font-mono">{motor}</td>
                        <td className="text-right py-1.5 font-mono text-tx2">{min}</td>
                        <td className="text-right py-1.5 font-mono text-tx">{pos}</td>
                        <td className="text-right py-1.5 font-mono text-tx2">{max}</td>
                        <td className={`text-right py-1.5 font-mono ${moved ? 'text-gn' : 'text-yl'}`}>
                          {range}
                        </td>
                      </tr>
                    )
                  })}
                  <tr className="border-b border-bd/50">
                    <td className="py-1.5 text-tx font-mono">wrist_roll</td>
                    <td colSpan={4} className="text-right py-1.5 text-tx2 text-xs">
                      连续旋转 — 自动 [0, 4095]
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleFinish}
                className="px-4 py-2 rounded bg-gn text-white text-sm hover:bg-gn/90"
              >
                保存校准
              </button>
              <button
                onClick={handleCancel}
                className="px-4 py-2 rounded bg-sf border border-bd text-tx text-sm hover:bg-bd/20"
              >
                取消
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
