import { useI18n } from '@/i18n'

type StageStatus = 'idle' | 'running' | 'completed' | 'error'

interface PipelineStageProps {
  title: string
  description: string
  status: StageStatus
  stageNumber: number
  disabled?: boolean
  children?: React.ReactNode
}

const STATUS_COLORS: Record<StageStatus, string> = {
  idle: 'var(--c-tx2)',
  running: 'var(--c-ac)',
  completed: '#22c55e',
  error: '#ef4444',
}

export default function PipelineStage({
  title,
  description,
  status,
  stageNumber,
  disabled,
  children,
}: PipelineStageProps) {
  const { t } = useI18n()

  const statusLabel = t(status as 'idle' | 'running' | 'completed' | 'error')

  return (
    <div className={`pipeline-stage ${disabled ? 'pipeline-stage--disabled' : ''}`}>
      <div className="pipeline-stage__header">
        <span className="pipeline-stage__number">{stageNumber}</span>
        <div className="pipeline-stage__title-group">
          <h3 className="pipeline-stage__title">{title}</h3>
          <p className="pipeline-stage__desc">{description}</p>
        </div>
        <span
          className="pipeline-stage__badge"
          style={{ color: STATUS_COLORS[status] }}
        >
          {status === 'running' && <span className="pipeline-stage__spinner" />}
          {statusLabel}
        </span>
      </div>
      {children && <div className="pipeline-stage__body">{children}</div>}
    </div>
  )
}
