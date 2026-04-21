import { useCallback, useEffect, useState } from 'react'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'

const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
const MOTOR_SHORT = ['Pan', 'Lift', 'Elbow', 'Wrist', 'Roll', 'Grip']

function tempToHsl(temp: number): string {
  // 20°C → cool teal (180), 45°C → warm amber (40), 65°C+ → hot red (0)
  const t = Math.max(0, Math.min(1, (temp - 20) / 50))
  const hue = 180 - t * 180
  const sat = 70 + t * 20
  const light = 45 - t * 10
  return `hsl(${hue}, ${sat}%, ${light}%)`
}

function tempToTrackHsl(temp: number): string {
  const t = Math.max(0, Math.min(1, (temp - 20) / 50))
  const hue = 180 - t * 180
  return `hsl(${hue}, 30%, 92%)`
}

function RingGauge({ temp, label, size = 56 }: { temp: number | null; label: string; size?: number }) {
  const r = (size - 8) / 2
  const circumference = 2 * Math.PI * r
  const fraction = temp != null ? Math.min(temp / 80, 1) : 0
  const dashOffset = circumference * (1 - fraction)
  const center = size / 2

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background track */}
          <circle
            cx={center} cy={center} r={r}
            fill="none"
            stroke={temp != null ? tempToTrackHsl(temp) : '#e8ebf2'}
            strokeWidth={5}
          />
          {/* Value arc */}
          {temp != null && (
            <circle
              cx={center} cy={center} r={r}
              fill="none"
              stroke={tempToHsl(temp)}
              strokeWidth={5}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.6s ease' }}
            />
          )}
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex items-center justify-center">
          {temp != null ? (
            <span
              className="text-xs font-bold tabular-nums"
              style={{ color: tempToHsl(temp), transition: 'color 0.6s ease' }}
            >
              {temp}°
            </span>
          ) : (
            <span className="text-2xs text-tx3">--</span>
          )}
        </div>
      </div>
      <span className="text-2xs text-tx3 font-medium">{label}</span>
    </div>
  )
}

function ArmTemperatureCard({ alias, temps }: {
  alias: string
  temps: Record<string, number> | null
}) {
  const values = MOTOR_NAMES.map(m => temps?.[m] ?? null)
  const validTemps = values.filter((v): v is number => v != null)
  const avg = validTemps.length > 0 ? Math.round(validTemps.reduce((a, b) => a + b, 0) / validTemps.length) : null
  const max = validTemps.length > 0 ? Math.max(...validTemps) : null

  return (
    <div className="rounded-xl border border-bd/30 bg-white p-4 shadow-card">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-tx">{alias}</span>
        {avg != null && max != null && (
          <div className="flex items-center gap-2 text-2xs font-mono">
            <span className="text-tx3">avg</span>
            <span style={{ color: tempToHsl(avg) }} className="font-bold">{avg}°C</span>
            <span className="text-tx3">max</span>
            <span style={{ color: tempToHsl(max) }} className="font-bold">{max}°C</span>
          </div>
        )}
      </div>
      <div className="flex justify-between">
        {MOTOR_NAMES.map((motor, i) => (
          <RingGauge key={motor} temp={values[i]} label={MOTOR_SHORT[i]} />
        ))}
      </div>
    </div>
  )
}

export function TemperatureHeatMap() {
  const { t } = useI18n()
  const servoPollingEnabled = useHardwareStore((state) => state.servoPollingEnabled)
  const [temperatures, setTemperatures] = useState<Record<string, Record<string, number>>>({})
  const [loading, setLoading] = useState(true)

  const poll = useCallback(async () => {
    if (!servoPollingEnabled) return
    const r = await fetch('/api/hardware/servos')
    const data = await r.json()
    if (data.error || !data.arms) return
    const nextTemps: Record<string, Record<string, number>> = {}
    for (const [alias, armData] of Object.entries(data.arms)) {
      const temps = (armData as any).temperatures
      if (typeof temps === 'object' && temps) nextTemps[alias] = temps
    }
    setTemperatures(nextTemps)
    setLoading(false)
  }, [servoPollingEnabled])

  useEffect(() => {
    if (!servoPollingEnabled) return
    setLoading(true)
    poll()
    const timer = setInterval(poll, 2000)
    return () => clearInterval(timer)
  }, [poll, servoPollingEnabled])

  const armNames = Object.keys(temperatures)

  if (!servoPollingEnabled) {
    return (
      <div className="rounded-lg border border-rd/20 bg-rd/5 px-3 py-2 text-sm text-rd">
        {t('servoPollingDisabled')}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="text-sm text-tx3 text-center py-6">{t('servoLoading')}</div>
    )
  }

  if (armNames.length === 0) {
    return (
      <div className="text-sm text-tx3 text-center py-6">{t('servoNoData')}</div>
    )
  }

  return (
    <div className="space-y-3">
      {armNames.map(alias => (
        <ArmTemperatureCard key={alias} alias={alias} temps={temperatures[alias]} />
      ))}
    </div>
  )
}
