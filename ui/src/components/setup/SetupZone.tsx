import { useDroppable } from '@dnd-kit/core'
import { useI18n } from '../../controllers/i18n'
import type { ConfiguredArm, ConfiguredCamera } from '../../controllers/setup'

interface Props {
  arms: ConfiguredArm[]
  cameras: ConfiguredCamera[]
  onRemoveArm: (alias: string) => void
  onRemoveCamera: (alias: string) => void
}

export default function SetupZone({ arms, cameras, onRemoveArm, onRemoveCamera }: Props) {
  const { isOver, setNodeRef } = useDroppable({ id: 'setup-zone' })
  const { t } = useI18n()

  return (
    <div
      ref={setNodeRef}
      className={`
        flex-1 min-h-[300px] rounded-xl border-2 border-dashed p-4 transition-colors duration-300
        ${isOver ? 'border-ac bg-ac/5' : 'border-bd bg-sf/50'}
      `}
    >
      <h3 className="text-xs text-tx2 uppercase tracking-wider font-medium mb-3">
        {t('configuredSetup')}
      </h3>

      {arms.length === 0 && cameras.length === 0 && (
        <div className="flex items-center justify-center h-32 text-tx2 text-sm">
          {t('dragToSetup')}
        </div>
      )}

      {arms.length > 0 && (
        <div className="mb-3">
          <div className="text-2xs text-tx2 uppercase mb-1.5">{t('arms')}</div>
          <div className="space-y-1.5">
            {arms.map((arm) => (
              <div
                key={arm.alias}
                className="flex items-center gap-2 bg-bg border border-bd rounded-lg px-3 py-2 group"
              >
                <span className={`w-2 h-2 rounded-full ${arm.type.includes('leader') ? 'bg-ac' : 'bg-gn'}`} />
                <span className="text-sm font-medium text-tx flex-1">{arm.alias}</span>
                <span className="text-2xs text-tx2">
                  {arm.type.includes('leader') ? t('leader') : t('follower')}
                </span>
                <button
                  onClick={() => onRemoveArm(arm.alias)}
                  className="text-2xs text-rd opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {cameras.length > 0 && (
        <div>
          <div className="text-2xs text-tx2 uppercase mb-1.5">{t('cameras')}</div>
          <div className="space-y-1.5">
            {cameras.map((cam) => (
              <div
                key={cam.alias}
                className="flex items-center gap-2 bg-bg border border-bd rounded-lg px-3 py-2 group"
              >
                <span className="w-2 h-2 rounded-full bg-ac" />
                <span className="text-sm font-medium text-tx flex-1">{cam.alias}</span>
                <button
                  onClick={() => onRemoveCamera(cam.alias)}
                  className="text-2xs text-rd opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
