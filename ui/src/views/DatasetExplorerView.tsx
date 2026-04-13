import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useI18n } from '../controllers/i18n'
import {
  useExplorer,
  type EpisodeDetail,
  type FeatureStat,
  type ModalityItem,
} from '../controllers/explorer'
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

function formatAngle(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '--'
  return value.toFixed(3)
}

function getTrajectoryTimeBounds(detail: EpisodeDetail): [number, number] {
  const timeValues = detail.joint_trajectory.time_values
  if (timeValues.length >= 2) {
    return [timeValues[0], timeValues[timeValues.length - 1]]
  }
  const duration = detail.summary.duration_s || 0
  return [0, duration]
}

function getNearestTrajectoryIndex(detail: EpisodeDetail, videoCurrentTime: number): number {
  const timeValues = detail.joint_trajectory.time_values
  if (timeValues.length > 0) {
    let nearestIndex = 0
    let nearestDistance = Number.POSITIVE_INFINITY
    timeValues.forEach((value, index) => {
      const distance = Math.abs(value - videoCurrentTime)
      if (distance < nearestDistance) {
        nearestDistance = distance
        nearestIndex = index
      }
    })
    return nearestIndex
  }

  const firstJoint = detail.joint_trajectory.joint_trajectories[0]
  const totalPoints = Math.max(
    firstJoint?.state_values.length ?? 0,
    firstJoint?.action_values.length ?? 0,
  )
  if (totalPoints <= 1) return 0
  const duration = detail.summary.duration_s || 1
  const progress = Math.min(Math.max(videoCurrentTime / duration, 0), 1)
  return Math.round(progress * (totalPoints - 1))
}

function EpisodeHoverPreview({
  detail,
  loading,
  error,
  playVideo,
  videoCurrentTime,
  onVideoTimeUpdate,
  onClose,
  onMouseEnter,
  onMouseLeave,
}: {
  detail: EpisodeDetail | null
  loading: boolean
  error: string
  playVideo: boolean
  videoCurrentTime: number
  onVideoTimeUpdate: (seconds: number) => void
  onClose: () => void
  onMouseEnter: () => void
  onMouseLeave: () => void
}) {
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([])
  const syncLockRef = useRef(false)
  const jointTrajectories = detail?.joint_trajectory.joint_trajectories ?? []
  const [timeMin, timeMax] = detail ? getTrajectoryTimeBounds(detail) : [0, 0]
  const timeRange = timeMax - timeMin || 1
  const currentIndex = detail ? getNearestTrajectoryIndex(detail, videoCurrentTime) : 0
  const currentTimePercent = detail
    ? Math.min(Math.max(((videoCurrentTime - timeMin) / timeRange) * 100, 0), 100)
    : 0

  useEffect(() => {
    videoRefs.current = []
  }, [detail?.episode_index])

  useEffect(() => {
    const syncFromSource = (
      sourceIndex: number,
      options: { forceSeek?: boolean } = {},
    ) => {
      const source = videoRefs.current[sourceIndex]
      if (!source || syncLockRef.current) return

      syncLockRef.current = true
      const sourceTime = source.currentTime
      const sourcePaused = source.paused
      const sourceRate = source.playbackRate

      videoRefs.current.forEach((target, targetIndex) => {
        if (!target || targetIndex === sourceIndex) return

        if (target.playbackRate !== sourceRate) {
          target.playbackRate = sourceRate
        }

        const shouldSeek =
          options.forceSeek || Math.abs(target.currentTime - sourceTime) > 0.08
        if (shouldSeek) {
          try {
            target.currentTime = sourceTime
          } catch (_error) {
            // Ignore currentTime assignment failures until metadata is ready.
          }
        }

        if (sourcePaused || !playVideo) {
          if (!target.paused) {
            target.pause()
          }
        } else if (target.paused) {
          const playPromise = target.play()
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {})
          }
        }
      })

      queueMicrotask(() => {
        syncLockRef.current = false
      })
    }

    const listeners: Array<() => void> = []
    videoRefs.current.forEach((video, index) => {
      if (!video) return

      const handlePlay = () => {
        if (syncLockRef.current) return
        onVideoTimeUpdate(video.currentTime)
        syncFromSource(index, { forceSeek: true })
      }
      const handlePause = () => {
        if (syncLockRef.current) return
        onVideoTimeUpdate(video.currentTime)
        syncFromSource(index)
      }
      const handleSeeking = () => {
        if (syncLockRef.current) return
        onVideoTimeUpdate(video.currentTime)
        syncFromSource(index, { forceSeek: true })
      }
      const handleSeeked = () => {
        if (syncLockRef.current) return
        onVideoTimeUpdate(video.currentTime)
        syncFromSource(index, { forceSeek: true })
      }
      const handleRateChange = () => {
        if (syncLockRef.current) return
        syncFromSource(index)
      }
      const handleTimeUpdate = () => {
        if (syncLockRef.current) return
        onVideoTimeUpdate(video.currentTime)
        syncFromSource(index)
      }
      const handleLoadedMetadata = () => {
        if (syncLockRef.current) return
        syncFromSource(index, { forceSeek: true })
      }

      video.addEventListener('play', handlePlay)
      video.addEventListener('pause', handlePause)
      video.addEventListener('seeking', handleSeeking)
      video.addEventListener('seeked', handleSeeked)
      video.addEventListener('ratechange', handleRateChange)
      video.addEventListener('timeupdate', handleTimeUpdate)
      video.addEventListener('loadedmetadata', handleLoadedMetadata)

      listeners.push(() => {
        video.removeEventListener('play', handlePlay)
        video.removeEventListener('pause', handlePause)
        video.removeEventListener('seeking', handleSeeking)
        video.removeEventListener('seeked', handleSeeked)
        video.removeEventListener('ratechange', handleRateChange)
        video.removeEventListener('timeupdate', handleTimeUpdate)
        video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      })
    })

    return () => {
      listeners.forEach((cleanup) => cleanup())
    }
  }, [detail?.episode_index, onVideoTimeUpdate, playVideo])

  useEffect(() => {
    const videos = videoRefs.current.filter((video): video is HTMLVideoElement => Boolean(video))
    if (!videos.length) return

    if (!playVideo) {
      videos.forEach((video) => video.pause())
      return
    }

    let attempts = 0
    const tryPlay = () => {
      const currentVideos = videoRefs.current.filter(
        (video): video is HTMLVideoElement => Boolean(video),
      )
      if (!currentVideos.length) {
        return
      }

      currentVideos.forEach((video) => {
        if (!video.paused) {
          return
        }
        const playPromise = video.play()
        if (playPromise && typeof playPromise.catch === 'function') {
          playPromise.catch(() => {})
        }
      })
    }

    tryPlay()
    const retryTimer = window.setInterval(() => {
      attempts += 1
      tryPlay()
      const currentVideos = videoRefs.current.filter(
        (video): video is HTMLVideoElement => Boolean(video),
      )
      const allPlaying = currentVideos.length > 0 && currentVideos.every((video) => !video.paused)
      if (allPlaying || attempts >= 12) {
        window.clearInterval(retryTimer)
      }
    }, 120)

    return () => {
      window.clearInterval(retryTimer)
    }
  }, [playVideo, detail?.videos])

  useEffect(() => {
    const interval = window.setInterval(() => {
      const [leader, ...followers] = videoRefs.current.filter(
        (video): video is HTMLVideoElement => Boolean(video),
      )
      if (!leader || followers.length === 0 || syncLockRef.current) {
        return
      }

      const leaderTime = leader.currentTime
      const leaderPaused = leader.paused || !playVideo
      const leaderRate = leader.playbackRate

      followers.forEach((video) => {
        if (video.playbackRate !== leaderRate) {
          video.playbackRate = leaderRate
        }

        if (Math.abs(video.currentTime - leaderTime) > 0.08) {
          try {
            video.currentTime = leaderTime
          } catch (_error) {
            // Ignore currentTime sync failures until metadata is available.
          }
        }

        if (leaderPaused) {
          if (!video.paused) {
            video.pause()
          }
        } else if (video.paused) {
          const playPromise = video.play()
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {})
          }
        }
      })
    }, 120)

    return () => {
      window.clearInterval(interval)
    }
  }, [detail?.episode_index, playVideo])

  return createPortal(
    <div className="explorer-hover-preview" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <div className="explorer-hover-preview__dialog">
        <button
          type="button"
          className="explorer-hover-preview__close"
          onClick={onClose}
          aria-label="Close preview"
        >
          ×
        </button>

        {!detail && loading && (
          <div className="explorer-hover-preview__empty">Loading preview...</div>
        )}

        {!detail && error && (
          <div className="explorer-hover-preview__empty explorer-hover-preview__empty--error">
            {error}
          </div>
        )}

        {detail && (
          <>
            <div className="explorer-hover-preview__header">
              <h3>Episode #{detail.episode_index}</h3>
              <div className="explorer-hover-preview__meta">
                <span>{detail.summary.row_count} frames</span>
                <span>{detail.summary.duration_s}s</span>
                <span>{detail.summary.fps} fps</span>
                <span>{detail.summary.video_count} videos</span>
              </div>
            </div>

            <div className="explorer-hover-preview__body">
              <div className="explorer-hover-preview__video-grid">
                {detail.videos.length > 0 ? (
                  detail.videos.map((video, index) => (
                    <div key={video.path} className="explorer-hover-preview__video-card">
                      <div className="explorer-hover-preview__status">
                        Stream: {video.stream}
                      </div>
                      <video
                        ref={(node) => {
                          videoRefs.current[index] = node
                        }}
                        src={video.url}
                        autoPlay={playVideo}
                        controls
                        muted
                        loop
                        playsInline
                        preload="metadata"
                        onTimeUpdate={(event) => {
                          if (index === 0) {
                            onVideoTimeUpdate(event.currentTarget.currentTime)
                          }
                        }}
                      />
                    </div>
                  ))
                ) : (
                  <div className="explorer-hover-preview__empty">
                    No video stream available for this episode.
                  </div>
                )}
              </div>

              {jointTrajectories.length > 0 && (
                <div className="explorer-hover-preview__charts">
                  <h4>Joint Angle Info</h4>
                  <div className="explorer-hover-preview__legend">
                    <span className="explorer-hover-preview__legend-state">State</span>
                    <span className="explorer-hover-preview__legend-action">Action</span>
                  </div>

                  <div className="explorer-hover-preview__charts-grid">
                    {jointTrajectories.map((joint) => {
                      const actionValues = joint.action_values.map((value) => value ?? 0)
                      const stateValues = joint.state_values.map((value) => value ?? 0)
                      const allValues = [...actionValues, ...stateValues]
                      const minValue = Math.min(...allValues)
                      const maxValue = Math.max(...allValues)
                      const padding = (maxValue - minValue || 1) * 0.1
                      const yMin = minValue - padding
                      const yMax = maxValue + padding
                      const yRange = yMax - yMin || 1

                      const toY = (value: number) => 10 + ((yMax - value) / yRange) * 40
                      const buildPolyline = (values: number[]) =>
                        values
                          .map((value, index) => {
                            const x = values.length > 1 ? (index / (values.length - 1)) * 100 : 50
                            return `${x},${toY(value)}`
                          })
                          .join(' ')

                      const currentState = stateValues[Math.min(currentIndex, stateValues.length - 1)]
                      const currentAction = actionValues[Math.min(currentIndex, actionValues.length - 1)]

                      return (
                        <div key={joint.joint_name} className="explorer-hover-preview__chart">
                          <div className="explorer-hover-preview__chart-title-row">
                            <div className="explorer-hover-preview__chart-title">{joint.joint_name}</div>
                            <div className="explorer-hover-preview__chart-current">
                              S {formatAngle(currentState)} / A {formatAngle(currentAction)}
                            </div>
                          </div>

                          <div className="explorer-hover-preview__chart-container">
                            <div className="explorer-hover-preview__chart-yaxis">
                              <span>{yMax.toFixed(2)}</span>
                              <span>{((yMax + yMin) / 2).toFixed(2)}</span>
                              <span>{yMin.toFixed(2)}</span>
                            </div>

                            <div className="explorer-hover-preview__chart-svg-wrap">
                              <svg viewBox="0 0 100 60" preserveAspectRatio="none">
                                <polyline
                                  points={buildPolyline(stateValues)}
                                  fill="none"
                                  stroke="#2f6fe4"
                                  strokeWidth="0.55"
                                  vectorEffect="non-scaling-stroke"
                                />
                                <polyline
                                  points={buildPolyline(actionValues)}
                                  fill="none"
                                  stroke="#f59e0b"
                                  strokeWidth="0.55"
                                  vectorEffect="non-scaling-stroke"
                                />
                                <line
                                  x1={currentTimePercent}
                                  y1="10"
                                  x2={currentTimePercent}
                                  y2="50"
                                  stroke="#ef4444"
                                  strokeWidth="0.35"
                                  strokeDasharray="2,2"
                                  vectorEffect="non-scaling-stroke"
                                />
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
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function EpisodeBrowser() {
  const { t } = useI18n()
  const {
    episodePage,
    episodePageLoading,
    episodePageError,
    loadEpisodePage,
    selectedEpisodeIndex,
    selectEpisode,
    episodeDetail,
    episodeLoading,
    episodeError,
    clearEpisode,
  } = useExplorer()
  const episodes = episodePage?.episodes ?? []
  const selectedDataset = episodePage?.dataset ?? ''
  const hoverTimerRef = useRef<number | null>(null)
  const closeTimerRef = useRef<number | null>(null)
  const requestTokenRef = useRef(0)
  const previewCacheRef = useRef<Map<number, EpisodeDetail>>(new Map())
  const [hoveredEpisodeIndex, setHoveredEpisodeIndex] = useState<number | null>(null)
  const [hoveredPreview, setHoveredPreview] = useState<EpisodeDetail | null>(null)
  const [hoveredPreviewLoading, setHoveredPreviewLoading] = useState(false)
  const [hoveredPreviewError, setHoveredPreviewError] = useState('')
  const [previewPlayReady, setPreviewPlayReady] = useState(false)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)

  useEffect(() => {
    previewCacheRef.current.clear()
    setHoveredEpisodeIndex(null)
    setHoveredPreview(null)
    setHoveredPreviewLoading(false)
    setHoveredPreviewError('')
    setPreviewPlayReady(false)
    setVideoCurrentTime(0)
  }, [selectedDataset])

  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        window.clearTimeout(hoverTimerRef.current)
      }
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current)
      }
    }
  }, [])

  if (episodePageLoading && !episodePage) {
    return <div className="explorer-empty">{t('running')}...</div>
  }

  if (episodePageError && !episodePageLoading) {
    return <div className="explorer-empty quality-sidebar__error">{episodePageError}</div>
  }

  if (episodes.length === 0) {
    return <div className="explorer-empty">{t('noDatasets')}</div>
  }

  const pageStart = (episodePage!.page - 1) * episodePage!.page_size + 1
  const pageStop = pageStart + episodes.length - 1

  const previewVisible = hoveredEpisodeIndex !== null

  const cancelClosePreview = () => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
  }

  const scheduleClosePreview = () => {
    cancelClosePreview()
    if (hoverTimerRef.current) {
      window.clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
    closeTimerRef.current = window.setTimeout(() => {
      setHoveredEpisodeIndex(null)
      setHoveredPreview(null)
      setHoveredPreviewLoading(false)
      setHoveredPreviewError('')
      setPreviewPlayReady(false)
      setVideoCurrentTime(0)
    }, 180)
  }

  const scheduleHoverPreview = (episodeIndex: number) => {
    cancelClosePreview()
    if (hoverTimerRef.current) {
      window.clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
    setHoveredPreviewError('')
    setPreviewPlayReady(false)
    setVideoCurrentTime(0)

    hoverTimerRef.current = window.setTimeout(async () => {
      setHoveredEpisodeIndex(episodeIndex)
      setPreviewPlayReady(true)

      const cached = previewCacheRef.current.get(episodeIndex)
      if (cached) {
        setHoveredPreview(cached)
        setHoveredPreviewLoading(false)
        return
      }

      setHoveredPreview(null)
      setHoveredPreviewLoading(true)

      const requestToken = ++requestTokenRef.current
      try {
        const response = await fetch(
          `/api/explorer/episode?dataset=${encodeURIComponent(selectedDataset)}&episode_index=${episodeIndex}`,
        )
        if (!response.ok) {
          throw new Error(`Failed to load episode preview (${response.status})`)
        }
        const detail: EpisodeDetail = await response.json()
        previewCacheRef.current.set(episodeIndex, detail)
        if (requestToken === requestTokenRef.current) {
          setHoveredPreview(detail)
        }
      } catch (error) {
        if (requestToken === requestTokenRef.current) {
          setHoveredPreviewError(error instanceof Error ? error.message : 'Failed to load preview')
        }
      } finally {
        if (requestToken === requestTokenRef.current) {
          setHoveredPreviewLoading(false)
        }
      }
    }, 500)
  }

  return (
    <div className="explorer-episodes">
      <div className="explorer-episodes__toolbar">
        <div className="explorer-episodes__summary">
          <span>{episodePage!.total_episodes} {t('episodes')}</span>
          <span>{pageStart}-{pageStop}</span>
          <span>{episodePage!.page}/{episodePage!.total_pages}</span>
        </div>
        <div className="explorer-episodes__pagination">
          <button
            type="button"
            className="explorer-episodes__pager"
            disabled={episodePage!.page <= 1 || episodePageLoading}
            onClick={() => void loadEpisodePage(selectedDataset, episodePage!.page - 1, episodePage!.page_size)}
          >
            Prev
          </button>
          <button
            type="button"
            className="explorer-episodes__pager"
            disabled={episodePage!.page >= episodePage!.total_pages || episodePageLoading}
            onClick={() => void loadEpisodePage(selectedDataset, episodePage!.page + 1, episodePage!.page_size)}
          >
            Next
          </button>
        </div>
      </div>

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
            onMouseEnter={() => scheduleHoverPreview(ep.episode_index)}
            onMouseMove={() => scheduleHoverPreview(ep.episode_index)}
            onMouseLeave={scheduleClosePreview}
          >
            <span className="explorer-episode-item__idx">#{ep.episode_index}</span>
            <span className="explorer-episode-item__len">{ep.length} frames</span>
          </button>
        ))}
      </div>

      {previewVisible && (
        <EpisodeHoverPreview
          detail={hoveredPreview}
          loading={hoveredPreviewLoading}
          error={hoveredPreviewError}
          videoCurrentTime={videoCurrentTime}
          onVideoTimeUpdate={setVideoCurrentTime}
          onClose={() => {
            setHoveredEpisodeIndex(null)
            setHoveredPreview(null)
            setHoveredPreviewLoading(false)
            setHoveredPreviewError('')
            setPreviewPlayReady(false)
            setVideoCurrentTime(0)
          }}
          playVideo={previewPlayReady}
          onMouseEnter={cancelClosePreview}
          onMouseLeave={scheduleClosePreview}
        />
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
    summary,
    summaryLoading,
    summaryError,
    dashboard,
    dashboardLoading,
    dashboardError,
    episodePage,
    loadSummary,
    loadDashboard,
    loadEpisodePage,
  } = useExplorer()
  const [datasetIdInput, setDatasetIdInput] = useState('')
  const currentDataset = selectedDataset || summary?.dataset || dashboard?.dataset || episodePage?.dataset || ''

  useEffect(() => {
    if (!currentDataset) {
      return
    }
    if (summary?.dataset !== currentDataset) {
      void loadSummary(currentDataset).catch(() => {})
    }
    if (dashboard?.dataset !== currentDataset) {
      void loadDashboard(currentDataset).catch(() => {})
    }
    if (episodePage?.dataset !== currentDataset) {
      void loadEpisodePage(currentDataset, 1, 50)
    }
  }, [currentDataset, summary?.dataset, dashboard?.dataset, episodePage?.dataset, loadSummary, loadDashboard, loadEpisodePage])

  async function handleLoad(): Promise<void> {
    const datasetId = datasetIdInput.trim()
    if (!datasetId) {
      return
    }
    await Promise.allSettled([
      loadSummary(datasetId),
      loadDashboard(datasetId),
      loadEpisodePage(datasetId, 1, 50),
    ])
    try {
      await selectDataset(datasetId)
    } catch {
      // Explorer dashboard is already loaded; workflow sync can retry on destination pages.
    }
  }

  const datasetSummary = summary?.summary

  return (
    <div className="page-enter quality-view">
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
      {currentDataset && datasetSummary ? (
        <div className="workflow-view__info-bar">
          <span>{summary!.dataset}</span>
          <span>{datasetSummary.total_episodes} {t('episodes')}</span>
          <span>{datasetSummary.fps} fps</span>
          <span>{datasetSummary.robot_type}</span>
          {datasetSummary.codebase_version && <span>{datasetSummary.codebase_version}</span>}
        </div>
      ) : summaryError ? (
        <GlassPanel className="quality-view__empty">
          <span className="quality-sidebar__error">{summaryError}</span>
        </GlassPanel>
      ) : !summaryLoading ? (
        <GlassPanel className="quality-view__empty">
          {t('hfDatasetPlaceholder')}
        </GlassPanel>
      ) : null}

      {summaryLoading && (
        <GlassPanel className="quality-view__empty">{t('running')}...</GlassPanel>
      )}

      {currentDataset && (datasetSummary || dashboard || episodePage) && (
        <div className="quality-layout">
          <div className="quality-layout__main">
            {/* KPIs */}
            <div className="quality-kpis">
              <MetricCard label={t('totalEpisodes')} value={datasetSummary?.total_episodes ?? '--'} />
              <MetricCard label="Frames" value={datasetSummary?.total_frames ?? '--'} accent="sage" />
              <MetricCard label="FPS" value={datasetSummary?.fps ?? '--'} accent="amber" />
              <MetricCard label={t('parquetFiles')} value={dashboard?.files.parquet_files ?? '--'} accent="teal" />
              <MetricCard label={t('videoFiles')} value={dashboard?.files.video_files ?? '--'} accent="coral" />
            </div>

            {/* Modality chips */}
            <GlassPanel className="explorer-section">
              <h3>{t('modalities')}</h3>
              {dashboardLoading && !dashboard ? (
                <div className="explorer-empty">{t('running')}...</div>
              ) : dashboard ? (
                <ModalityChips items={dashboard.modality_summary} />
              ) : (
                <div className="explorer-empty">{dashboardError || t('noStats')}</div>
              )}
            </GlassPanel>

            {/* Feature stats table */}
            <GlassPanel className="explorer-section">
              <h3>{t('featureStats')}</h3>
              {dashboard ? (
                <>
                  <p className="explorer-section__sub">
                    {dashboard.feature_names.length} features
                    {dashboard.dataset_stats.features_with_stats > 0 &&
                      ` / ${dashboard.dataset_stats.features_with_stats} with stats`}
                  </p>
                  <FeatureStatsTable stats={dashboard.feature_stats} />
                </>
              ) : (
                <div className="explorer-empty">{dashboardLoading ? t('running') : (dashboardError || t('noStats'))}</div>
              )}
            </GlassPanel>
          </div>

          {/* Sidebar */}
          <GlassPanel className="quality-layout__sidebar">
            <div className="quality-sidebar__section">
              <h3>{t('episodeBrowser')}</h3>
              <EpisodeBrowser />
            </div>

            <div className="quality-sidebar__section">
              <h3>{t('fileInventory')}</h3>
              {dashboard ? (
                <div className="explorer-sidebar-stats">
                  <div><span className="explorer-sidebar-stats__label">{t('totalFiles')}</span> <span>{dashboard.files.total_files}</span></div>
                  <div><span className="explorer-sidebar-stats__label">{t('parquetFiles')}</span> <span>{dashboard.files.parquet_files}</span></div>
                  <div><span className="explorer-sidebar-stats__label">{t('videoFiles')}</span> <span>{dashboard.files.video_files}</span></div>
                  <div><span className="explorer-sidebar-stats__label">{t('metaFiles')}</span> <span>{dashboard.files.meta_files}</span></div>
                  <div><span className="explorer-sidebar-stats__label">{t('otherFiles')}</span> <span>{dashboard.files.other_files}</span></div>
                </div>
              ) : (
                <div className="explorer-empty">{dashboardLoading ? t('running') : (dashboardError || t('noStats'))}</div>
              )}
            </div>

            <div className="quality-sidebar__section">
              <h3>{t('featureType')}</h3>
              {dashboard ? (
                <TypeDistribution items={dashboard.feature_type_distribution} />
              ) : (
                <div className="explorer-empty">{dashboardLoading ? t('running') : (dashboardError || t('noStats'))}</div>
              )}
            </div>

            {dashboard?.dataset_stats.row_count != null && (
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
