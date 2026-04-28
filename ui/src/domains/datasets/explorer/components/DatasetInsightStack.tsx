import { useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from '@/shared/lib/cn'
import {
  type ExplorerDashboard,
  type ExplorerEpisodePage,
  type ExplorerSummary,
  type FeatureStat,
} from '@/domains/datasets/explorer/store/useExplorerStore'

type StackCardId = 'statistics' | 'filtering' | 'frames' | 'action-insights'

interface StackCardDefinition {
  id: StackCardId
  title: string
  eyebrow: string
  summary: string
  metric: string
  detail: string
  body: ReactNode
}

interface DatasetInsightStackProps {
  summary: ExplorerSummary['summary'] | null | undefined
  dashboard: ExplorerDashboard | null | undefined
  episodePage: ExplorerEpisodePage | null | undefined
  dashboardLoading: boolean
  dashboardError: string
  featureStatsNode: ReactNode
  modalitiesNode: ReactNode
  typeDistributionNode: ReactNode
}

function formatNumber(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '--'
  return value.toLocaleString()
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '--'
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`
}

function getFeatureScalar(stat: FeatureStat, key: string): number | null {
  const values = stat.stats_preview[key]?.values
  if (!values || values.length === 0) return null
  const numbers = values.filter((value): value is number => typeof value === 'number')
  if (numbers.length === 0) return null
  return numbers.reduce((total, value) => total + Math.abs(value), 0) / numbers.length
}

function getActionFeatures(dashboard: ExplorerDashboard | null | undefined): FeatureStat[] {
  return (dashboard?.feature_stats ?? []).filter((feature) => {
    const name = feature.name.toLowerCase()
    return name === 'action' || name.includes('.action') || name.includes('action.')
  })
}

function EpisodeLengthPreview({
  episodePage,
  fps,
}: {
  episodePage: ExplorerEpisodePage | null | undefined
  fps: number | null | undefined
}) {
  const episodes = episodePage?.episodes ?? []
  const usableFps = fps && fps > 0 ? fps : null
  const lengths = episodes.map((episode) => ({
    id: episode.episode_index,
    frames: episode.length,
    seconds: usableFps ? episode.length / usableFps : null,
  }))
  if (lengths.length === 0) {
    return <div className="explorer-empty">Load episodes to inspect length distribution.</div>
  }

  const min = Math.min(...lengths.map((episode) => episode.frames))
  const max = Math.max(...lengths.map((episode) => episode.frames))
  const spread = max - min || 1
  const shortest = [...lengths].sort((a, b) => a.frames - b.frames).slice(0, 6)
  const longest = [...lengths].sort((a, b) => b.frames - a.frames).slice(0, 6)

  return (
    <div className="dataset-stack-filter">
      <div className="dataset-stack-filter__range">
        <span>{formatSeconds(usableFps ? min / usableFps : null)}</span>
        <div className="dataset-stack-filter__track">
          <div className="dataset-stack-filter__fill" />
        </div>
        <span>{formatSeconds(usableFps ? max / usableFps : null)}</span>
      </div>
      <div className="dataset-stack-filter__groups">
        <EpisodeLengthList title="Shortest" episodes={shortest} min={min} spread={spread} />
        <EpisodeLengthList title="Longest" episodes={longest} min={min} spread={spread} />
      </div>
    </div>
  )
}

function EpisodeLengthList({
  title,
  episodes,
  min,
  spread,
}: {
  title: string
  episodes: Array<{ id: number; frames: number; seconds: number | null }>
  min: number
  spread: number
}) {
  return (
    <div className="dataset-stack-mini-list">
      <div className="dataset-stack-mini-list__title">{title}</div>
      {episodes.map((episode) => (
        <div key={episode.id} className="dataset-stack-mini-list__row">
          <span>ep {episode.id}</span>
          <div className="dataset-stack-mini-list__bar">
            <div
              style={{ width: `${Math.max(8, ((episode.frames - min) / spread) * 100)}%` }}
            />
          </div>
          <strong>{formatNumber(episode.frames)}</strong>
        </div>
      ))}
    </div>
  )
}

function ActionInsightPreview({
  dashboard,
  summary,
}: {
  dashboard: ExplorerDashboard | null | undefined
  summary: ExplorerSummary['summary'] | null | undefined
}) {
  const actionFeatures = getActionFeatures(dashboard)
  const actionRows = actionFeatures
    .map((feature) => ({
      name: feature.name,
      components: feature.component_names.length || feature.shape.length || 1,
      std: getFeatureScalar(feature, 'std'),
      mean: getFeatureScalar(feature, 'mean'),
    }))
    .slice(0, 8)
  const maxStd = Math.max(...actionRows.map((row) => row.std ?? 0), 1)

  if (!dashboard) {
    return <div className="explorer-empty">Load feature statistics to inspect action signals.</div>
  }

  return (
    <div className="dataset-stack-action">
      <div className="dataset-stack-callout">
        <strong>{summary?.chunks_size ? summary.chunks_size : '--'}</strong>
        <span>Configured chunk size from dataset metadata</span>
      </div>
      {actionRows.length > 0 ? (
        <div className="dataset-stack-action__grid">
          {actionRows.map((row) => (
            <div key={row.name} className="dataset-stack-action__row">
              <div>
                <strong>{row.name}</strong>
                <span>{row.components} components</span>
              </div>
              <div className="dataset-stack-action__bar">
                <div style={{ width: `${Math.max(6, ((row.std ?? 0) / maxStd) * 100)}%` }} />
              </div>
              <span>{row.std == null ? '--' : row.std.toFixed(4)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="dataset-stack-note">
          No action feature statistics were found in the current dashboard payload.
        </div>
      )}
    </div>
  )
}

function FrameSchemaPreview({ dashboard }: { dashboard: ExplorerDashboard | null | undefined }) {
  const featureStats = dashboard?.feature_stats ?? []
  const preview = featureStats.slice(0, 10)

  if (!dashboard) {
    return <div className="explorer-empty">Load a dataset to inspect frame schema.</div>
  }

  return (
    <div className="dataset-stack-schema">
      <div className="dataset-stack-schema__meta">
        <span>{formatNumber(dashboard.dataset_stats.row_count)} rows</span>
        <span>{formatNumber(dashboard.feature_names.length)} features</span>
        <span>{formatNumber(dashboard.dataset_stats.vector_features)} vector features</span>
      </div>
      <div className="dataset-stack-schema__grid">
        {preview.map((feature) => (
          <div key={feature.name} className="dataset-stack-schema__item">
            <strong>{feature.name}</strong>
            <span>{feature.dtype}</span>
            <span>{feature.shape.length > 0 ? `[${feature.shape.join(', ')}]` : 'scalar'}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function LoadingOrError({
  loading,
  error,
  fallback,
}: {
  loading: boolean
  error: string
  fallback: ReactNode
}) {
  if (loading) {
    return <div className="explorer-empty">Loading...</div>
  }
  if (error) {
    return <div className="explorer-empty quality-sidebar__error">{error}</div>
  }
  return fallback
}

export function DatasetInsightStack({
  summary,
  dashboard,
  episodePage,
  dashboardLoading,
  dashboardError,
  featureStatsNode,
  modalitiesNode,
  typeDistributionNode,
}: DatasetInsightStackProps) {
  const [collapsed, setCollapsed] = useState<Set<StackCardId>>(
    () => new Set<StackCardId>(['frames']),
  )
  const actionFeatures = useMemo(() => getActionFeatures(dashboard), [dashboard])
  const averageFrames =
    episodePage && episodePage.episodes.length > 0
      ? Math.round(
          episodePage.episodes.reduce((total, episode) => total + episode.length, 0)
          / episodePage.episodes.length,
        )
      : null

  const cards: StackCardDefinition[] = [
    {
      id: 'statistics',
      title: 'Statistics',
      eyebrow: 'Dataset',
      summary: 'File inventory, modality coverage, and feature-level statistics.',
      metric: formatNumber(summary?.total_frames ?? dashboard?.dataset_stats.row_count),
      detail: `${formatNumber(dashboard?.feature_names.length)} features`,
      body: (
        <LoadingOrError
          loading={dashboardLoading}
          error={dashboardError}
          fallback={
            <>
              {modalitiesNode}
              <div className="dataset-stack-stat-grid">
                <div className="dataset-stack-stat-grid__main">{featureStatsNode}</div>
                <div className="dataset-stack-stat-grid__side">{typeDistributionNode}</div>
              </div>
            </>
          }
        />
      ),
    },
    {
      id: 'filtering',
      title: 'Filtering',
      eyebrow: 'Quality',
      summary: 'Episode-length spread and candidates that deserve a quick review.',
      metric: averageFrames == null ? '--' : formatNumber(averageFrames),
      detail: 'avg frames',
      body: <EpisodeLengthPreview episodePage={episodePage} fps={summary?.fps} />,
    },
    {
      id: 'frames',
      title: 'Frames',
      eyebrow: 'Schema',
      summary: 'Frame rows, vector features, dtypes, and compact schema preview.',
      metric: formatNumber(dashboard?.dataset_stats.row_count),
      detail: `${formatNumber(dashboard?.dataset_stats.vector_features)} vectors`,
      body: <FrameSchemaPreview dashboard={dashboard} />,
    },
    {
      id: 'action-insights',
      title: 'Action Insights',
      eyebrow: 'Control',
      summary: 'Action feature coverage and a stats-backed smoothness proxy.',
      metric: formatNumber(actionFeatures.length),
      detail: 'action features',
      body: <ActionInsightPreview dashboard={dashboard} summary={summary} />,
    },
  ]

  function toggleCard(id: StackCardId): void {
    setCollapsed((current) => {
      const next = new Set(current)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  function getCardStyle(index: number): CSSProperties {
    return {
      '--stack-index': index,
      zIndex: 10 - index,
    } as CSSProperties
  }

  return (
    <div className="dataset-insight-stack">
      {cards.map((card, index) => {
        const isCollapsed = collapsed.has(card.id)
        return (
          <section
            key={card.id}
            className={cn('dataset-stack-card', isCollapsed && 'is-collapsed')}
            style={getCardStyle(index)}
          >
            <button
              type="button"
              className="dataset-stack-card__header"
              onClick={() => toggleCard(card.id)}
              aria-expanded={!isCollapsed}
            >
              <span className="dataset-stack-card__badge">{card.eyebrow}</span>
              <span className="dataset-stack-card__title-block">
                <strong>{card.title}</strong>
                <span>{card.summary}</span>
              </span>
              <span className="dataset-stack-card__metric">
                <strong>{card.metric}</strong>
                <span>{card.detail}</span>
              </span>
              <span className="dataset-stack-card__chevron" aria-hidden="true" />
            </button>
            {!isCollapsed && <div className="dataset-stack-card__body">{card.body}</div>}
          </section>
        )
      })}
    </div>
  )
}
