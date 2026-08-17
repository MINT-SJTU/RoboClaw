import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react'

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

type ButtonVariant = 'primary' | 'secondary' | 'success' | 'warning' | 'danger'

const buttonStyles: Record<ButtonVariant, string> = {
  primary: 'bg-ac text-white shadow-[0_12px_28px_rgba(47,111,228,0.22)] hover:-translate-y-0.5 hover:shadow-[0_18px_34px_rgba(47,111,228,0.28)]',
  secondary: 'border border-[color:rgba(47,111,228,0.14)] bg-white/92 text-tx hover:-translate-y-0.5 hover:bg-white',
  success: 'bg-ac text-white shadow-[0_12px_28px_rgba(47,111,228,0.22)] hover:-translate-y-0.5 hover:shadow-[0_18px_34px_rgba(47,111,228,0.28)]',
  warning: 'border border-[color:rgba(47,111,228,0.16)] bg-[rgba(47,111,228,0.08)] text-ac hover:-translate-y-0.5 hover:bg-[rgba(47,111,228,0.14)]',
  danger: 'border border-[color:rgba(17,17,17,0.14)] bg-[rgba(17,17,17,0.04)] text-tx hover:-translate-y-0.5 hover:bg-[rgba(17,17,17,0.08)]',
}

export function ActionButton({
  children,
  className,
  variant = 'primary',
  type,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant
}) {
  return (
    <button
      type={type ?? 'button'}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold transition duration-200 ease-out disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-40',
        buttonStyles[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function GlassPanel({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('glass-panel rounded-[28px] p-5 md:p-6', className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function SectionIntro({
  eyebrow,
  title,
  description,
  aside,
  className,
}: {
  eyebrow?: string
  title: string
  description?: string
  aside?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col gap-4 md:flex-row md:items-end md:justify-between', className)}>
      <div className="space-y-2">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <div className="space-y-2">
          <h2 className="display-title text-3xl text-tx md:text-4xl">{title}</h2>
          {description && <p className="max-w-2xl text-sm leading-7 text-tx2 md:text-base">{description}</p>}
        </div>
      </div>
      {aside && <div className="flex flex-wrap items-center gap-3">{aside}</div>}
    </div>
  )
}

export function MetricCard({
  label,
  value,
  caption,
  accent = 'teal',
}: {
  label: string
  value: string | number
  caption?: string
  accent?: 'teal' | 'amber' | 'coral' | 'sage'
}) {
  return (
    <div className={cn('metric-card', `metric-card--${accent}`)}>
      <div className="text-[11px] uppercase tracking-[0.22em] text-tx2">{label}</div>
      <div className="mt-3 text-2xl font-semibold text-tx md:text-[2rem]">{value}</div>
      {caption && <div className="mt-2 text-sm text-tx2">{caption}</div>}
    </div>
  )
}

export function StatusPill({
  active,
  children,
}: {
  active?: boolean
  children: ReactNode
}) {
  return (
    <span className={cn('status-pill', active && 'status-pill--live')}>
      <span className="status-pill__dot" />
      {children}
    </span>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      <div className="empty-state__orb" />
      <div className="relative z-[1] max-w-md space-y-3 text-center">
        <h3 className="display-title text-2xl text-tx">{title}</h3>
        <p className="text-sm leading-7 text-tx2">{description}</p>
        {action && <div className="pt-2">{action}</div>}
      </div>
    </div>
  )
}
