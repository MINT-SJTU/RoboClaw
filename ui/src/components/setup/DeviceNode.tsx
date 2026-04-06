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
          className="w-12 h-9 object-cover rounded shrink-0"
          draggable={false}
        />
      ) : (
        <span className="w-8 h-8 flex items-center justify-center rounded-md bg-sf2 border border-bd/30 text-tx2 text-sm shrink-0">
          {icon}
        </span>
      )}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-tx truncate">{label}</div>
        <div className="text-2xs text-tx2 truncate">{sublabel}</div>
      </div>
      {moved && (
        <span className="shrink-0 w-2 h-2 rounded-full bg-gn animate-pulse" />
      )}
      {children}
    </div>
  )
}
