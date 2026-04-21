import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

interface SummaryMetric {
  label: string
  value: string
}

interface SettingsSummaryCardProps {
  to: string
  title: string
  description: string
  actionLabel: string
  status: string
  metrics: SummaryMetric[]
  accent: 'ac' | 'yl' | 'gn' | 'rd'
  icon: ReactNode
}

const ACCENT_STYLES = {
  ac: 'border-ac/25 shadow-glow-ac',
  yl: 'border-yl/25 shadow-[0_18px_45px_-28px_rgba(217,163,47,0.45)]',
  gn: 'border-gn/25 shadow-[0_18px_45px_-28px_rgba(63,160,95,0.45)]',
  rd: 'border-rd/25 shadow-[0_18px_45px_-28px_rgba(200,58,58,0.45)]',
}

const STATUS_STYLES = {
  ac: 'bg-ac/10 text-ac',
  yl: 'bg-yl/10 text-yl',
  gn: 'bg-gn/10 text-gn',
  rd: 'bg-rd/10 text-rd',
}

export default function SettingsSummaryCard({
  to,
  title,
  description,
  actionLabel,
  status,
  metrics,
  accent,
  icon,
}: SettingsSummaryCardProps) {
  return (
    <Link
      to={to}
      className={`group block rounded-2xl border bg-white p-5 transition-all hover:-translate-y-0.5 hover:border-ac/35 ${ACCENT_STYLES[accent]}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl ${STATUS_STYLES[accent]}`}>
            {icon}
          </div>
          <div className="min-w-0">
            <h3 className="text-lg font-semibold text-tx">{title}</h3>
            <p className="mt-1 text-sm text-tx3">{description}</p>
          </div>
        </div>
        <span className={`shrink-0 rounded-full px-3 py-1 text-2xs font-semibold ${STATUS_STYLES[accent]}`}>
          {status}
        </span>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {metrics.map((metric) => (
          <div key={metric.label} className="min-w-0 rounded-xl border border-bd/30 bg-sf px-3 py-3">
            <div className="text-2xs uppercase tracking-[0.16em] text-tx3">{metric.label}</div>
            <div className="mt-2 break-words text-base font-semibold text-tx">{metric.value}</div>
          </div>
        ))}
      </div>

      <div className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-ac">
        <span>{actionLabel}</span>
        <span className="transition-transform group-hover:translate-x-1">→</span>
      </div>
    </Link>
  )
}
