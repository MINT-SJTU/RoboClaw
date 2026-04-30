import { useState } from 'react'
import {
  useSetup,
  ConfiguredArm,
  ConfiguredCamera,
  ConfiguredHand,
} from '@/domains/hardware/setup/store/useSetupStore'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'

interface RowProps {
  alias: string
  typeBadge: string
  dotColor: string
  statusTag?: { label: string; color: string } | null
  onRename: (newAlias: string) => void
  onRemove: () => void
}

function DeviceRow({ alias, typeBadge, dotColor, statusTag, onRename, onRemove }: RowProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(alias)

  const save = () => {
    const trimmed = draft.trim()
    if (trimmed && trimmed !== alias) onRename(trimmed)
    setEditing(false)
  }
  const cancel = () => { setDraft(alias); setEditing(false) }

  return (
    <div className="group flex items-center gap-2 px-3 py-2 rounded-lg bg-white hover:bg-sf border border-transparent hover:border-bd/20 shadow-sm transition-colors">
      <span className={`shrink-0 w-2 h-2 rounded-full ${dotColor}`} />

      {editing ? (
        <div className="flex items-center gap-1 min-w-0">
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') cancel() }}
            className="w-24 px-1.5 py-0.5 text-sm bg-sf2 border border-bd rounded text-tx outline-none focus:border-ac"
          />
          <button onClick={save} className="text-2xs text-ac hover:underline">OK</button>
          <button onClick={cancel} className="text-2xs text-tx2 hover:underline">ESC</button>
        </div>
      ) : (
        <span
          className="text-sm text-tx truncate cursor-pointer hover:text-ac"
          onClick={() => { setDraft(alias); setEditing(true) }}
        >
          {alias}
        </span>
      )}

      <span className="shrink-0 px-1.5 py-0.5 text-2xs rounded bg-sf2 text-tx2 border border-bd/40 font-mono">
        {typeBadge}
      </span>
      {statusTag && (
        <span className={`shrink-0 px-1.5 py-0.5 text-2xs rounded text-white ${statusTag.color}`}>
          {statusTag.label}
        </span>
      )}
      <div className="flex-1" />
      <button
        onClick={onRemove}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-sm text-tx2 hover:text-rd"
      >
        &times;
      </button>
    </div>
  )
}

function dotColorForArm(connected: boolean, calibrated: boolean): string {
  if (!connected) return 'bg-rd'
  if (!calibrated) return 'bg-yl'
  return 'bg-gn'
}

export default function DeviceList({ onManualCalibrate }: { onManualCalibrate?: (alias: string) => void } = {}) {
  const { devices, removeArm, renameArm, removeCamera, renameCamera, removeHand, renameHand } =
    useSetup()
  const hwStatus = useHardwareStore((state) => state.hardwareStatus)
  const { t } = useI18n()
  const { arms, cameras, hands } = devices

  if (arms.length === 0 && cameras.length === 0 && hands.length === 0) {
    return <p className="text-sm text-tx2 text-center py-8">{t('noConfiguredDevices')}</p>
  }

  const section = (label: string, children: React.ReactNode) => (
    <section>
      <h4 className="text-2xs uppercase tracking-wide text-tx2 mb-1">{label}</h4>
      <div className="flex flex-col gap-1">{children}</div>
    </section>
  )

  return (
    <div className="flex flex-col gap-4">
      {arms.length > 0 && section(t('arm'), arms.map((a: ConfiguredArm) => {
        const live = hwStatus?.arms.find((h) => h.alias === a.alias)
        const connected = live ? live.connected : true  // assume connected if no live data yet
        const dotColor = dotColorForArm(connected, a.calibrated)
        const statusTag = !connected
          ? { label: t('hwDisconnected'), color: 'bg-rd' }
          : a.calibrated
            ? { label: t('hwCalibrated'), color: 'bg-gn' }
            : { label: t('hwUncalibrated'), color: 'bg-yl' }
        return (
          <div key={a.alias} className="flex items-center gap-1">
            <div className="flex-1">
              <DeviceRow
                alias={a.alias}
                typeBadge={a.type}
                dotColor={dotColor}
                statusTag={statusTag}
                onRename={(n) => renameArm(a.alias, n)}
                onRemove={() => removeArm(a.alias)}
              />
            </div>
            {connected && onManualCalibrate && (
              <button
                onClick={() => onManualCalibrate(a.alias)}
                className={`shrink-0 px-2 py-1 text-2xs rounded ${
                  a.calibrated
                    ? 'text-tx3 border border-bd/40 hover:text-ac hover:border-ac/60 hover:bg-ac/5'
                    : 'text-ac border border-ac/60 hover:bg-ac/10'
                }`}
              >
                {t('manualCalibrate')}
              </button>
            )}
          </div>
        )
      }))}

      {cameras.length > 0 && section(t('camera'), cameras.map((c: ConfiguredCamera) => {
        const live = hwStatus?.cameras.find((h) => h.alias === c.alias)
        const connected = live ? live.connected : true
        return (
          <DeviceRow
            key={c.alias}
            alias={c.alias}
            typeBadge={c.port}
            dotColor={connected ? 'bg-gn' : 'bg-rd'}
            statusTag={!connected ? { label: t('hwDisconnected'), color: 'bg-rd' } : null}
            onRename={(n) => renameCamera(c.alias, n)}
            onRemove={() => removeCamera(c.alias)}
          />
        )
      }))}

      {hands.length > 0 && section(t('hand'), hands.map((h: ConfiguredHand) => (
        <DeviceRow
          key={h.alias}
          alias={h.alias}
          typeBadge={h.type}
          dotColor="bg-gn"
          onRename={(n) => renameHand(h.alias, n)}
          onRemove={() => removeHand(h.alias)}
        />
      )))}
    </div>
  )
}
