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

  const shape = kind === 'port' ? 'rounded-full' : 'rounded-lg'
  const glowCls = moved ? 'animate-glow-green shadow-[0_0_20px_rgba(34,197,94,0.6)]' : ''

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
      className={`
        relative p-3 border-2 bg-sf cursor-grab active:cursor-grabbing
        transition-all duration-500 animate-node-appear select-none
        ${shape} ${glowCls}
        ${moved ? 'border-gn bg-gn/5' : 'border-bd hover:border-ac'}
      `}
    >
      {previewUrl && (
        <img
          src={previewUrl}
          alt={label}
          className="w-full aspect-video object-cover rounded mb-2"
          draggable={false}
        />
      )}
      <div className="text-sm font-medium text-tx truncate">{label}</div>
      <div className="text-2xs text-tx2 truncate">{sublabel}</div>
      {children}
    </div>
  )
}
