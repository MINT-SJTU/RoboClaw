import { useState } from 'react'
import { useSetup } from '../../controllers/setup'
import { useI18n } from '../../controllers/i18n'

/* ── iOS-style toggle switch ────────────────────────────────────────── */

function Toggle({ on, disabled, onClick }: {
  on: boolean
  disabled: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={onClick}
      className={`relative inline-flex h-[26px] w-[46px] shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${on ? 'bg-gn' : 'bg-tx3/30'}`}
    >
      <span className={`pointer-events-none inline-block h-[22px] w-[22px] rounded-full bg-white shadow-lg ring-0 transition-transform duration-200 ease-in-out
        ${on ? 'translate-x-5' : 'translate-x-0'}`}
      />
    </button>
  )
}

/* ── Permission row ─────────────────────────────────────────────────── */

function PermissionRow({ icon, label, ok, count, onToggle, busy }: {
  icon: string
  label: string
  ok: boolean
  count: number
  onToggle: () => void
  busy: boolean
}) {
  const { t } = useI18n()
  const noDevice = count === 0

  return (
    <div className="flex items-center gap-3 py-3 border-b border-bd/15 last:border-b-0">
      <span className="text-base w-6 text-center shrink-0">{icon}</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-tx font-medium">{label}</p>
        <p className="text-2xs text-tx3">
          {noDevice
            ? t('permNoDevice')
            : t('permDeviceCount', { count })}
        </p>
      </div>
      {noDevice ? (
        <Toggle on={false} disabled onClick={() => {}} />
      ) : (
        <Toggle on={ok} disabled={busy || ok} onClick={onToggle} />
      )}
    </div>
  )
}

/* ── Panel ──────────────────────────────────────────────────────────── */

export default function PermissionPanel({ bare, onFixed }: {
  bare?: boolean
  onFixed: () => void
}) {
  const { t } = useI18n()
  const permissions = useSetup((s) => s.permissions)
  const permFixing = useSetup((s) => s.permFixing)
  const fixPermissions = useSetup((s) => s.fixPermissions)
  const [fixAttempted, setFixAttempted] = useState(false)

  if (!permissions || permissions.platform !== 'linux') return null

  const showHint = fixAttempted && !(permissions.serial.ok && permissions.camera.ok) && permissions.hint

  async function handleToggle() {
    setFixAttempted(true)
    await fixPermissions()
    const updated = useSetup.getState().permissions
    if (updated && updated.serial.ok && updated.camera.ok) {
      onFixed()
    }
  }

  const content = (
    <>
      {!bare && <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-3">{t('permTitle')}</h3>}
      <PermissionRow
        icon="🔌"
        label={t('permSerial')}
        ok={permissions.serial.ok}
        count={permissions.serial.count}
        onToggle={handleToggle}
        busy={permFixing}
      />
      <PermissionRow
        icon="📷"
        label={t('permCamera')}
        ok={permissions.camera.ok}
        count={permissions.camera.count}
        onToggle={handleToggle}
        busy={permFixing}
      />
      {showHint && (
        <div className="mt-3 p-3 rounded-lg bg-yl/5 border border-yl/20 space-y-1.5">
          <p className="text-2xs text-tx2">{t('permFixFailed')}</p>
          <code className="block px-2.5 py-1.5 bg-sf2 rounded text-xs text-tx font-mono select-all">
            {permissions.hint}
          </code>
        </div>
      )}
    </>
  )

  if (bare) return <div className="space-y-0">{content}</div>

  return (
    <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac">
      {content}
    </section>
  )
}
