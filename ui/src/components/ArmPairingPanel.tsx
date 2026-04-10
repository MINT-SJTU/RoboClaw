import { useState, useEffect } from 'react'
import { useI18n } from '../controllers/i18n'
import type { ArmStatus } from '../controllers/dashboard'

interface ArmPairingPanelProps {
  arms: ArmStatus[]
  onChange: (selectedAliases: string) => void
}

export function ArmPairingPanel({ arms, onChange }: ArmPairingPanelProps) {
  const { t } = useI18n()
  const [selected, setSelected] = useState<Set<string>>(() => new Set(arms.map(a => a.alias)))

  useEffect(() => {
    setSelected(new Set(arms.map(a => a.alias)))
  }, [arms.length])

  useEffect(() => {
    // All selected = empty string (use all arms, same as default)
    const allSelected = selected.size === arms.length
    onChange(allSelected ? '' : Array.from(selected).join(','))
  }, [selected, arms.length])

  const followers = arms.filter(a => a.role === 'follower')
  const leaders = arms.filter(a => a.role === 'leader')

  function toggle(alias: string) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(alias)) next.delete(alias)
      else next.add(alias)
      return next
    })
  }

  function renderArm(arm: ArmStatus) {
    const ok = arm.connected && arm.calibrated
    return (
      <label key={arm.alias} className="flex items-center gap-2 text-xs cursor-pointer py-0.5">
        <input
          type="checkbox"
          checked={selected.has(arm.alias)}
          onChange={() => toggle(arm.alias)}
          className="w-3.5 h-3.5 rounded border-bd accent-ac"
        />
        <span className={`w-2 h-2 rounded-full ${ok ? 'bg-gn' : 'bg-yl'}`} />
        <span className="font-mono text-tx">{arm.alias}</span>
      </label>
    )
  }

  return (
    <div className="space-y-1.5">
      <span className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('armSelection')}</span>
      <div className="flex gap-4">
        <div className="space-y-0.5">
          <span className="text-2xs text-tx3 font-medium">{t('followers')}</span>
          {followers.length > 0
            ? followers.map(renderArm)
            : <span className="text-2xs text-tx3">--</span>}
        </div>
        <div className="space-y-0.5">
          <span className="text-2xs text-tx3 font-medium">{t('leaders')}</span>
          {leaders.length > 0
            ? leaders.map(renderArm)
            : <span className="text-2xs text-tx3">--</span>}
        </div>
      </div>
    </div>
  )
}
