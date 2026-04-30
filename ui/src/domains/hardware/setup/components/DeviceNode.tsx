import { useDraggable } from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'

interface Props {
  id: string
  kind: 'port' | 'camera'
  label: string
  sublabel: string
  moved?: boolean
  previewUrl?: string | null
  children?: React.ReactNode
}

export default function DeviceNode({ id, kind, label, sublabel, moved, previewUrl, children }: Props) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({ id })

  const style = {
    transform: CSS.Translate.toString(transform),
    opacity: isDragging ? 0.5 : 1,
  }

  const icon = kind === 'port' ? '⊞' : '◎'
  const glowCls = moved ? 'ring-2 ring-gn/30 shadow-card border-gn/30' : ''

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
      className={`
        flex items-center gap-3 px-3 py-2.5 rounded-lg border bg-white shadow-card
        cursor-grab active:cursor-grabbing transition-all select-none
        ${glowCls}
        ${moved ? 'border-gn/30 bg-gn/[0.03]' : 'border-bd/30 hover:border-ac/40'}
      `}
    >
      {previewUrl ? (
        <img
          src={previewUrl}
          alt={label}
          className="h-10 w-12 shrink-0 rounded-md border border-bd/30 object-cover"
          draggable={false}
        />
      ) : (
        <span className="flex h-10 w-12 shrink-0 items-center justify-center rounded-md border border-bd/30 bg-sf2 text-sm text-tx2">
          {icon}
        </span>
      )}
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-tx">{label}</div>
        {sublabel && (
          <div className="truncate text-2xs text-tx2">{sublabel}</div>
        )}
      </div>
      {moved && (
        <span className="shrink-0 w-2 h-2 rounded-full bg-gn animate-pulse" />
      )}
      {children}
    </div>
  )
}
