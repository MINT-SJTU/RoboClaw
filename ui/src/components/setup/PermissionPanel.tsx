import { useState } from 'react'
import { useSetup } from '../../controllers/setup'
import type { PermissionStatus } from '../../controllers/setup'
import { useI18n } from '../../controllers/i18n'

function PermissionRow({ label, ok, count, onFix, fixing }: {
  label: string
  ok: boolean
  count: number
  onFix: () => void
  fixing: boolean
}) {
  const { t } = useI18n()
  const noDevice = count === 0

  return (
    <div className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg bg-white border border-bd/20 shadow-sm">
      <span className={`shrink-0 w-2.5 h-2.5 rounded-full ${
        noDevice ? 'bg-tx3/30' : ok ? 'bg-gn' : 'bg-rd'
      }`} />
      <span className="text-sm text-tx flex-1">{label}</span>
      {noDevice ? (
        <span className="text-2xs text-tx3">{t('permNoDevice')}</span>
      ) : ok ? (
        <span className="text-2xs text-gn font-medium">
          {t('permGranted')} · {t('permDeviceCount').replace('{count}', String(count))}
        </span>
      ) : (
        <div className="flex items-center gap-2">
          <span className="text-2xs text-rd font-medium">{t('permDenied')}</span>
          <button
            onClick={onFix}
            disabled={fixing}
            className="px-2.5 py-1 text-2xs bg-ac text-white rounded-md font-medium hover:bg-ac2 disabled:opacity-40 transition-colors"
          >
            {fixing ? t('permFixing') : t('permFix')}
          </button>
        </div>
      )}
    </div>
  )
}

export default function PermissionPanel({ perms, onFixed }: {
  perms: PermissionStatus
  onFixed: () => void
}) {
  const { t } = useI18n()
  const { fixPermissions, permFixing, permissions } = useSetup()
  const [fixAttempted, setFixAttempted] = useState(false)

  const current = permissions ?? perms
  const allOk = current.serial.ok && current.camera.ok
  const showHint = fixAttempted && !allOk && current.hint

  async function handleFix() {
    setFixAttempted(true)
    await fixPermissions()
    const updated = useSetup.getState().permissions
    if (updated && updated.serial.ok && updated.camera.ok) {
      onFixed()
    }
  }

  if (current.platform !== 'linux') return null

  return (
    <div className="rounded-lg border border-bd/30 bg-sf p-4 space-y-2 shadow-card">
      <PermissionRow
        label={t('permSerial')}
        ok={current.serial.ok}
        count={current.serial.count}
        onFix={handleFix}
        fixing={permFixing}
      />
      <PermissionRow
        label={t('permCamera')}
        ok={current.camera.ok}
        count={current.camera.count}
        onFix={handleFix}
        fixing={permFixing}
      />
      {showHint && (
        <div className="mt-2 p-3 rounded-lg bg-yl/5 border border-yl/20 space-y-1.5">
          <p className="text-2xs text-tx2">{t('permFixFailed')}</p>
          <code className="block px-2.5 py-1.5 bg-sf2 rounded text-xs text-tx font-mono select-all">
            {current.hint}
          </code>
        </div>
      )}
    </div>
  )
}
