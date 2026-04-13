import { useEffect, useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useI18n } from '../controllers/i18n'
import { useExplorer, type FeatureStat, type ModalityItem, type EpisodeDetail } from '../controllers/explorer'
import { useWorkflow } from '../controllers/curation'
import { ActionButton, GlassPanel, MetricCard } from '../components/ux'

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

// ---------------------------------------------------------------------------
// Modality chips
// ---------------------------------------------------------------------------

function ModalityChips({ items }: { items: ModalityItem[] }) {
  return (
    <div className="explorer-modalities">
      {items.map((item) => (
        <span
          key={item.id}
          className={cn('explorer-modality-chip', item.present && 'is-active')}
          title={item.detail}
        >
          {item.label}
        </span>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Feature stats table
// ---------------------------------------------------------------------------

function formatStatValues(values: unknown[] | undefined): string {
  if (!values || values.length === 0) return '-'
  return values
    .map((v) => (typeof v === 'number' ? v.toFixed(3) : String(v)))
    .join(', ')
}

function FeatureStatsTable({ stats }: { stats: FeatureStat[] }) {
  const { t } = useI18n()

  if (stats.length === 0) {
    return <div className="explorer-empty">{t('noStats')}</div>
  }

  return (
    <div className="quality-table-wrap explorer-feature-stats-wrap">
      <table className="quality-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Dtype</th>
            <th>{t('shape')}</th>
            <th>{t('components')}</th>
            <th>Min</th>
            <th>Max</th>
            <th>Mean</th>
            <th>Std</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((feat) => (
            <tr key={feat.name}>
              <td className="explorer-feature-name">{feat.name}</td>
              <td>{feat.dtype}</td>
              <td>{feat.shape.length > 0 ? `[${feat.shape.join(', ')}]` : '-'}</td>
              <td>
                {feat.component_names.length > 0
                  ? feat.component_names.length > 3
                    ? `${feat.component_names.slice(0, 3).join(', ')}...`
                    : feat.component_names.join(', ')
                  : '-'}
              </td>
              <td>{formatStatValues(feat.stats_preview.min?.values)}</td>
              <td>{formatStatValues(feat.stats_preview.max?.values)}</td>
              <td>{formatStatValues(feat.stats_preview.mean?.values)}</td>
              <td>{formatStatValues(feat.stats_preview.std?.values)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Feature type distribution chart
// ---------------------------------------------------------------------------

function TypeDistribution({ items }: { items: Array<{ name: string; value: number }> }) {
  const maxValue = Math.max(...items.map((i) => i.value), 1)

  return (
    <div className="explorer-type-dist">
      {items.map((item) => (
        <div key={item.name} className="quality-chart-card__row">
          <div className="quality-chart-card__label">{item.name}</div>
          <div className="quality-chart-card__track">
            <div
              className="quality-chart-card__fill"
              style={{ width: `${(item.value / maxValue) * 100}%` }}
            />
          </div>
          <div className="quality-chart-card__value">{item.value}</div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Episode browser
// ---------------------------------------------------------------------------

function EpisodeBrowser() {
  const { t } = useI18n()
  const {
    dashboard,
    selectedEpisodeIndex,
    selectEpisode,
    episodeDetail,
    episodeLoading,
    episodeError,
    clearEpisode,
  } = useExplorer()
  const episodes = dashboard?.episodes ?? []
  const selectedDataset = dashboard?.dataset ?? ''

  const [hoveredEpisodeIndex, setHoveredEpisodeIndex] = useState<number | null>(null)
  const [hoveredPreview, setHoveredPreview] = useState<EpisodeDetail | null>(null)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const closeTimerRef = useRef<number | null>(null)

  if (episodes.length === 0) {
    return <div className="explorer-empty">{t('noDatasets')}</div>
  }

  const handleMouseEnter = async (episodeIndex: number) => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
    setHoveredEpisodeIndex(episodeIndex)
    try {
      const response = await fetch(
        `/api/explorer/episode?dataset=${encodeURIComponent(selectedDataset)}&episode_index=${episodeIndex}`
      )
      if (!response.ok) return
      const detail: EpisodeDetail = await response.json()
      setHoveredPreview(detail)
    } catch (error) {
      console.error('Failed to load preview:', error)
    }
  }

  const handleMouseLeave = () => {
    closeTimerRef.current = window.setTimeout(() => {
      setHoveredEpisodeIndex(null)
      setHoveredPreview(null)
      setVideoCurrentTime(0)
    }, 200)
  }

  return (
    <div className="explorer-episodes">
      <div className="explorer-episodes__list">
        {episodes.map((ep) => (
          <button
            key={ep.episode_index}
            type="button"
            className={cn(
              'explorer-episode-item',
              selectedEpisodeIndex === ep.episode_index && 'is-selected',
            )}
            onClick={() => {
              if (selectedEpisodeIndex === ep.episode_index) {
                clearEpisode()
              } else {
                void selectEpisode(selectedDataset || '', ep.episode_index)
              }
            }}
            onMouseEnter={() => handleMouseEnter(ep.episode_index)}
            onMouseLeave={handleMouseLeave}
          >
            <span className="explorer-episode-item__idx">#{ep.episode_index}</span>
            <span className="explorer-episode-item__len">{ep.length} frames</span>
          </button>
        ))}
      </div>

      {hoveredEpisodeIndex != null && hoveredPreview && createPortal(
        <div
          className="explorer-hover-preview"
          onMouseEnter={() => {
            if (closeTimerRef.current) {
              window.clearTimeout(closeTimerRef.current)
              closeTimerRef.current = null
            }
          }}
          onMouseLeave={handleMouseLeave}
        >
          <button
            className="explorer-hover-preview__close"
            onClick={() => {
              setHoveredEpisodeIndex(null)
              setHoveredPreview(null)
              setVideoCurrentTime(0)
            }}
          >
            ×
          </button>

          <div className="explorer-hover-preview__header">
            <h3>Episode #{hoveredPreview.episode_index}</h3>
            <div className="explorer-hover-preview__meta">
              <span>{hoveredPreview.summary.row_count} frames</span>
              <span>{hoveredPreview.summary.duration_s}s</span>
              <span>{hoveredPreview.summary.fps} fps</span>
            </div>
          </div>

          {hoveredPreview.videos.length > 0 && (
            <div className="explorer-hover-preview__video">
              <video
                ref={videoRef}
                src={hoveredPreview.videos[0].url}
                controls
                autoPlay
                muted
                loop
                onTimeUpdate={(e) => {
                  setVideoCurrentTime(e.currentTarget.currentTime)
                }}
              />
            </div>
          )}

          {hoveredPreview.joint_trajectory && hoveredPreview.joint_trajectory.joint_trajectories.length > 0 && (
            <div className="explorer-hover-preview__charts">
              <h4>Joint Trajectories</h4>
              <div className="explorer-hover-preview__legend">
                <span className="explorer-hover-preview__legend-state">State</span>
                <span className="explorer-hover-preview__legend-action">Action</span>
              </div>
              {hoveredPreview.joint_trajectory.joint_trajectories.map((joint) => {
                const timestamps = hoveredPreview.joint_trajectory.time_values
                const actionValues = joint.action_values.filter((v): v is number => v != null)
                const stateValues = joint.state_values.filter((v): v is number => v != null)

                const allValues = [...actionValues, ...stateValues]
                const min = Math.min(...allValues)
                const max = Math.max(...allValues)
                const range = max - min || 1
                const padding = range * 0.1
                const yMin = min - padding
                const yMax = max + padding
                const yRange = yMax - yMin

                const toY = (value: number) => {
                  return 10 + ((yMax - value) / yRange) * 40
                }

                const actionPoints = actionValues
                  .map((val, i) => `${(i / (actionValues.length - 1)) * 100},${toY(val)}`)
                  .join(' ')
                const statePoints = stateValues
                  .map((val, i) => `${(i / (stateValues.length - 1)) * 100},${toY(val)}`)
                  .join(' ')

                const timeMin = timestamps[0]
                const timeMax = timestamps[timestamps.length - 1]
                const timeRange = timeMax - timeMin || 1
                const currentTimePercent = ((videoCurrentTime - timeMin) / timeRange) * 100

                return (
                  <div key={joint.joint_name} className="explorer-hover-preview__chart">
                    <div className="explorer-hover-preview__chart-title">{joint.joint_name}</div>
                    <div className="explorer-hover-preview__chart-container">
                      <div className="explorer-hover-preview__chart-yaxis">
                        <span>{yMax.toFixed(2)}</span>
                        <span>{((yMax + yMin) / 2).toFixed(2)}</span>
                        <span>{yMin.toFixed(2)}</span>
                      </div>
                      <div className="explorer-hover-preview__chart-svg-wrap">
                        <svg viewBox="0 0 100 60" preserveAspectRatio="none">
                          <polyline
                            points={statePoints}
                            fill="none"
                            stroke="#2f6fe4"
                            strokeWidth="0.5"
                            vectorEffect="non-scaling-stroke"
                          />
                          <polyline
                            points={actionPoints}
                            fill="none"
                            stroke="#f59e0b"
                            strokeWidth="0.5"
                            vectorEffect="non-scaling-stroke"
                          />
                          {videoCurrentTime > 0 && (
                            <line
                              x1={currentTimePercent}
                              y1="10"
                              x2={currentTimePercent}
                              y2="50"
                              stroke="#ef4444"
                              strokeWidth="0.3"
                              strokeDasharray="2,2"
                              vectorEffect="non-scaling-stroke"
                            />
                          )}
                        </svg>
                        <div className="explorer-hover-preview__chart-xaxis">
                          <span>{timeMin.toFixed(1)}s</span>
                          <span>{((timeMin + timeMax) / 2).toFixed(1)}s</span>
                          <span>{timeMax.toFixed(1)}s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>,
        document.body
      )}

      {episodeLoading && (
        <div className="explorer-episode-detail">
          <p>{t('running')}...</p>
        </div>
      )}

      {episodeError && !episodeLoading && (
        <div className="explorer-episode-detail">
          <p className="quality-sidebar__error">{episodeError}</p>
        </div>
      )}

      {episodeDetail && !episodeLoading && !episodeError && (
        <div className="explorer-episode-detail">
          <div className="explorer-episode-detail__summary">
            <span>{episodeDetail.summary.row_count} rows</span>
            <span>{episodeDetail.summary.duration_s}s</span>
            <span>{episodeDetail.summary.fps} fps</span>
            <span>{episodeDetail.summary.video_count} videos</span>
          </div>

          {episodeDetail.videos.length > 0 && (
            <div className="explorer-episode-detail__section">
              <h4>Videos</h4>
              <ul className="explorer-video-list">
                {episodeDetail.videos.map((v) => (
                  <li key={v.path}>{v.stream} — {v.path}</li>
                ))}
              </ul>
            </div>
          )}

          {episodeDetail.sample_rows.length > 0 && (
            <div className="explorer-episode-detail__section">
              <h4>{t('sampleRows')}</h4>
              <div className="quality-table-wrap">
                <table className="quality-table explorer-sample-table">
                  <thead>
                    <tr>
                      {Object.keys(episodeDetail.sample_rows[0]).map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {episodeDetail.sample_rows.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((val, ci) => (
                          <td key={ci}>
                            {Array.isArray(val)
                              ? `[${val.join(', ')}]`
                              : val == null
                                ? '-'
                                : typeof val === 'number'
                                  ? val.toFixed(4)
                                  : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main view
// ---------------------------------------------------------------------------

export default function DatasetExplorerView() {
  const { t } = useI18n()
  const {
    selectedDataset,
    selectDataset,
  } = useWorkflow()
  const {
    dashboard,
    dashboardLoading,
    dashboardError,
    loadDashboard,
  } = useExplorer()
  const [datasetIdInput, setDatasetIdInput] = useState('')
  const currentDataset = selectedDataset || dashboard?.dataset || ''

  useEffect(() => {
    if (!currentDataset || dashboard?.dataset === currentDataset) {
      return
    }
    void loadDashboard(currentDataset).catch(() => {})
  }, [currentDataset, dashboard?.dataset, loadDashboard])

  async function handleLoad(): Promise<void> {
    const datasetId = datasetIdInput.trim()
    if (!datasetId) {
      return
    }
    await loadDashboard(datasetId)
    try {
      await selectDataset(datasetId)
    } catch {
      // Explorer dashboard is already loaded; workflow sync can retry on destination pages.
    }
  }

  const summary = dashboard?.summary

  return (
    <div className="quality-view">
      <div className="quality-view__hero">
        <div>
          <h2 className="quality-view__title">{t('explorerTitle')}</h2>
          <p className="quality-view__desc">{t('explorerDesc')}</p>
        </div>
      </div>

      <div className="dataset-workbench">
        <div className="dataset-workbench__controls">
          <label className="dataset-workbench__control dataset-workbench__control--wide">
            <span>{t('hfDatasetId')}</span>
            <input
              className="dataset-workbench__input"
              type="text"
              value={datasetIdInput}
              onChange={(event) => setDatasetIdInput(event.target.value)}
              placeholder={t('hfDatasetPlaceholder')}
            />
          </label>

          <ActionButton
            type="button"
            variant="secondary"
            onClick={() => void handleLoad()}
            disabled={!datasetIdInput.trim()}
            className="dataset-workbench__import-btn"
          >
            {t('selectDataset')}
          </ActionButton>
        </div>
      </div>

      {/* Info bar */}
      {currentDataset && summary ? (
        <div className="workflow-view__info-bar">
          <span>{dashboard.dataset}</span>
          <span>{summary.total_episodes} {t('episodes')}</span>
          <span>{summary.fps} fps</span>
          <span>{summary.robot_type}</span>
          {summary.codebase_version && <span>v{summary.codebase_version}</span>}
        </div>
      ) : dashboardError ? (
        <GlassPanel className="quality-view__empty">
          <span className="quality-sidebar__error">{dashboardError}</span>
        </GlassPanel>
      ) : !dashboardLoading ? (
        <GlassPanel className="quality-view__empty">
          {t('hfDatasetPlaceholder')}
        </GlassPanel>
      ) : null}

      {dashboardLoading && (
        <GlassPanel className="quality-view__empty">{t('running')}...</GlassPanel>
      )}

      {currentDataset && dashboard && !dashboardLoading && (
        <div className="quality-layout">
          <div className="quality-layout__main">
            {/* KPIs */}
            <div className="quality-kpis">
              <MetricCard label={t('totalEpisodes')} value={summary!.total_episodes} />
              <MetricCard label="Frames" value={summary!.total_frames} accent="sage" />
              <MetricCard label="FPS" value={summary!.fps} accent="amber" />
              <MetricCard label={t('parquetFiles')} value={dashboard.files.parquet_files} accent="teal" />
              <MetricCard label={t('videoFiles')} value={dashboard.files.video_files} accent="coral" />
            </div>

            {/* Modality chips */}
            <GlassPanel className="explorer-section">
              <h3>{t('modalities')}</h3>
              <ModalityChips items={dashboard.modality_summary} />
            </GlassPanel>

            {/* Feature stats table */}
            <GlassPanel className="explorer-section">
              <h3>{t('featureStats')}</h3>
              <p className="explorer-section__sub">
                {dashboard.feature_names.length} features
                {dashboard.dataset_stats.features_with_stats > 0 &&
                  ` / ${dashboard.dataset_stats.features_with_stats} with stats`}
              </p>
              <FeatureStatsTable stats={dashboard.feature_stats} />
            </GlassPanel>

            {/* Episode browser */}
            <GlassPanel className="explorer-section">
              <h3>{t('episodeBrowser')}</h3>
              <EpisodeBrowser />
            </GlassPanel>
          </div>

          {/* Sidebar */}
          <GlassPanel className="quality-layout__sidebar">
            <div className="quality-sidebar__section">
              <h3>{t('fileInventory')}</h3>
              <div className="explorer-sidebar-stats">
                <div><span className="explorer-sidebar-stats__label">{t('totalFiles')}</span> <span>{dashboard.files.total_files}</span></div>
                <div><span className="explorer-sidebar-stats__label">{t('parquetFiles')}</span> <span>{dashboard.files.parquet_files}</span></div>
                <div><span className="explorer-sidebar-stats__label">{t('videoFiles')}</span> <span>{dashboard.files.video_files}</span></div>
                <div><span className="explorer-sidebar-stats__label">{t('metaFiles')}</span> <span>{dashboard.files.meta_files}</span></div>
                <div><span className="explorer-sidebar-stats__label">{t('otherFiles')}</span> <span>{dashboard.files.other_files}</span></div>
              </div>
            </div>

            <div className="quality-sidebar__section">
              <h3>{t('featureType')}</h3>
              <TypeDistribution items={dashboard.feature_type_distribution} />
            </div>

            {dashboard.dataset_stats.row_count != null && (
              <div className="quality-sidebar__section">
                <div className="explorer-sidebar-stats">
                  <div><span className="explorer-sidebar-stats__label">Total rows</span> <span>{dashboard.dataset_stats.row_count.toLocaleString()}</span></div>
                  <div><span className="explorer-sidebar-stats__label">{t('vectorFeatures')}</span> <span>{dashboard.dataset_stats.vector_features}</span></div>
                </div>
              </div>
            )}
          </GlassPanel>
        </div>
      )}
    </div>
  )
}
