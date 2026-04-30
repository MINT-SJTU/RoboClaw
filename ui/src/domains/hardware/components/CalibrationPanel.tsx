import { useEffect } from 'react'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useI18n } from '@/i18n'
import { postJson } from '@/shared/api/client'

const CALIBRATION = '/api/calibration'

function sendCommand(cmd: string) {
  postJson(`${CALIBRATION}/command`, { command: cmd }).catch(() => {})
}

export function CalibrationPanel({ armAlias, onClose }: { armAlias: string; onClose: () => void }) {
  const step = useSessionStore((state) => state.session.calibration_step)
  const positions = useSessionStore((state) => state.session.calibration_positions)
  const sessionState = useSessionStore((state) => state.session.state)
  const { t } = useI18n()

  const isCalibrating = sessionState === 'calibrating'

  useEffect(() => {
    if (sessionState === 'idle' && step === 'done') {
      const timer = setTimeout(onClose, 1500)
      return () => clearTimeout(timer)
    }
  }, [sessionState, step, onClose])

  if (!isCalibrating && step !== 'done') {
    return null
  }

  return (
    <div className="rounded-xl border border-ac/30 bg-ac/5 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-bold text-tx">
          {t('calibrating')}: {armAlias}
        </h4>
        <button onClick={() => { sendCommand('stop'); onClose() }}
          className="text-2xs text-tx3 hover:text-rd">
          {t('cancel')}
        </button>
      </div>

      {!step && isCalibrating && (
        <p className="text-sm text-tx2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
          {t('calStarting')}
        </p>
      )}

      {step === 'choose' && (
        <div className="space-y-2">
          <p className="text-sm text-tx2">{t('calChoosePrompt')}</p>
          <div className="flex gap-2">
            <button onClick={() => sendCommand('confirm')}
              className="px-3 py-1.5 text-sm bg-gn text-white rounded-lg font-medium hover:bg-gn/90">
              {t('calUseExisting')}
            </button>
            <button onClick={() => sendCommand('recalibrate')}
              className="px-3 py-1.5 text-sm bg-ac text-white rounded-lg font-medium hover:bg-ac2">
              {t('calRunNew')}
            </button>
          </div>
        </div>
      )}

      {step === 'starting' && (
        <p className="text-sm text-tx2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-ac animate-pulse" />
          {t('calStarting')}
        </p>
      )}

      {step === 'homing' && (
        <div className="space-y-2">
          <p className="text-sm text-tx2">{t('calHomingPrompt')}</p>
          <button onClick={() => sendCommand('confirm')}
            className="px-3 py-1.5 text-sm bg-ac text-white rounded-lg font-medium hover:bg-ac2">
            {t('calConfirmMiddle')}
          </button>
        </div>
      )}

      {step === 'recording' && (
        <div className="space-y-2">
          <p className="text-sm text-tx2">{t('calRangePrompt')}</p>
          {positions && Object.keys(positions).length > 0 && (
            <div className="rounded-lg bg-bg border border-bd/40 p-2 font-mono text-2xs">
              <div className="grid grid-cols-4 gap-1 text-tx3 mb-1">
                <span>Motor</span><span className="text-right">Min</span>
                <span className="text-right">Pos</span><span className="text-right">Max</span>
              </div>
              {Object.entries(positions).map(([name, v]) => (
                <div key={name} className="grid grid-cols-4 gap-1 text-tx">
                  <span className="truncate">{name}</span>
                  <span className="text-right tabular-nums">{v.min}</span>
                  <span className="text-right tabular-nums font-bold">{v.pos}</span>
                  <span className="text-right tabular-nums">{v.max}</span>
                </div>
              ))}
            </div>
          )}
          <button onClick={() => sendCommand('confirm')}
            className="px-3 py-1.5 text-sm bg-gn text-white rounded-lg font-medium hover:bg-gn/90">
            {t('calFinish')}
          </button>
        </div>
      )}

      {step === 'done' && (
        <p className="text-sm text-gn font-medium flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-gn" />
          {t('calDone')}
        </p>
      )}
    </div>
  )
}
