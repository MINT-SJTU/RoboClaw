import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useI18n } from '@/i18n'

interface SettingsPageFrameProps {
  title: string
  description: string
  actions?: ReactNode
  children: ReactNode
}

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

export default function SettingsPageFrame({
  title,
  description,
  actions,
  children,
}: SettingsPageFrameProps) {
  const { t } = useI18n()

  const tabs = [
    { to: '/settings', label: t('settingsOverviewTab'), end: true },
    { to: '/settings/hardware', label: t('settingsHardware') },
    { to: '/settings/provider', label: t('settingsProvider') },
    { to: '/settings/hub', label: t('hfConfig') },
  ]

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 bg-sf">
        <div className="w-full px-6 py-5 2xl:px-10">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="text-2xs font-semibold uppercase tracking-[0.22em] text-tx3">
                {t('settings')}
              </div>
              <h2 className="mt-2 text-2xl font-bold tracking-tight text-tx">{title}</h2>
              <p className="mt-2 max-w-2xl text-sm text-tx3">{description}</p>
            </div>
            {actions && <div className="shrink-0">{actions}</div>}
          </div>

          <nav className="mt-5 flex flex-wrap gap-2">
            {tabs.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                end={tab.end}
                className={({ isActive }) => cn(
                  'rounded-full border px-3.5 py-2 text-sm font-medium transition-all',
                  isActive
                    ? 'border-ac bg-ac text-white shadow-glow-ac'
                    : 'border-bd/40 bg-white text-tx2 hover:border-ac/30 hover:text-ac',
                )}
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      <div className="flex-1 w-full px-6 py-6 2xl:px-10">
        {children}
      </div>
    </div>
  )
}
