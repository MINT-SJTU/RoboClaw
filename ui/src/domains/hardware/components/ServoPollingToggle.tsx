import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'

export function ServoPollingToggle() {
  const { t } = useI18n()
  const enabled = useHardwareStore((state) => state.servoPollingEnabled)
  const setEnabled = useHardwareStore((state) => state.setServoPollingEnabled)

  return (
    <div className="flex items-center gap-2">
      <span className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('servoPollingToggle')}</span>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label={t('servoPollingToggle')}
        onClick={() => setEnabled(!enabled)}
        className={`relative inline-flex h-[26px] w-[46px] shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
          enabled ? 'bg-gn' : 'bg-tx3/30'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-[22px] w-[22px] rounded-full bg-white shadow-lg ring-0 transition-transform duration-200 ease-in-out ${
            enabled ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  )
}
