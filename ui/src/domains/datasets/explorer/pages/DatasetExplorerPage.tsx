import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { useI18n } from '@/i18n'
import {
  listExplorerDatasets,
  searchDatasetSuggestions,
  type ExplorerDatasetRef,
  type ExplorerSource,
  type DatasetSuggestion,
  useExplorer,
  type EpisodeDetail,
  type FeatureStat,
  type ModalityItem,
} from '@/domains/datasets/explorer/store/useExplorerStore'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'
import { cn } from '@/shared/lib/cn'
import { ActionButton, GlassPanel, MetricCard } from '@/shared/ui'
import {
  clampAbsolutePlaybackTime,
  formatClipWindowLabel,
  getClipStart,
  getRelativePlaybackTime,
  shouldLoopVideo,
  type EpisodeVideo,
} from './datasetExplorerPlayback'

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

function syncVideoIntoClipWindow(
  element: HTMLVideoElement,
  video: EpisodeVideo | null | undefined,
  options: { loopToStart?: boolean; forceSeek?: boolean } = {},
): number {
  const nextTime = clampAbsolutePlaybackTime(
    video,
    element.currentTime,
    element.duration,
    { loopToStart: options.loopToStart },
  )
  if (options.forceSeek || Math.abs(element.currentTime - nextTime) > 0.08) {
    try {
      element.currentTime = nextTime
    } catch (_error) {
      // Ignore currentTime assignment failures until metadata is ready.
    }
  }
  return nextTime
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

function hasTrajectoryData(detail: EpisodeDetail | null | undefined): boolean {
  if (!detail) return false
  return (
    detail.joint_trajectory.joint_trajectories.length > 0 ||
    detail.joint_trajectory.total_points > 0
  )
}

function EpisodePlaybackSurface({
  detail,
  playVideo,
  videoCurrentTime,
  onVideoTimeUpdate,
  emptyLabel,
}: {
  detail: EpisodeDetail
  playVideo: boolean
  videoCurrentTime: number
  onVideoTimeUpdate: (seconds: number) => void
  emptyLabel: string
}) {
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([])
  const syncLockRef = useRef(false)
  const lastTimelineTimeRef = useRef<number>(-1)
  const jointTrajectories = detail.joint_trajectory.joint_trajectories
  const [timeMin, timeMax] = getTrajectoryTimeBounds(detail)
  const timeRange = timeMax - timeMin || 1
  const currentIndex = getNearestTrajectoryIndex(detail, videoCurrentTime)
  const currentTimePercent = Math.min(
    Math.max(((videoCurrentTime - timeMin) / timeRange) * 100, 0),
    100,
  )

  useEffect(() => {
    videoRefs.current = []
    lastTimelineTimeRef.current = -1
  }, [detail.episode_index])

  useEffect(() => {
    const timelineLeaderIndex = 0
    const updateTimelineFrom = (
      index: number,
      absoluteTime: number,
      options: { force?: boolean } = {},
    ) => {
      if (!options.force && index !== timelineLeaderIndex) {
        return
      }
      const relativeTime = getRelativePlaybackTime(getVideoMeta(index), absoluteTime)
      if (
        !options.force &&
        lastTimelineTimeRef.current >= 0 &&
        Math.abs(relativeTime - lastTimelineTimeRef.current) < 0.033
      ) {
        return
      }
      lastTimelineTimeRef.current = relativeTime
      onVideoTimeUpdate(relativeTime)
    }

    const getVideoMeta = (index: number): EpisodeVideo | null => detail.videos[index] ?? null
    const syncFromSource = (
      sourceIndex: number,
      options: { forceSeek?: boolean } = {},
    ) => {
      const source = videoRefs.current[sourceIndex]
      if (!source || syncLockRef.current) return

      syncLockRef.current = true
      const sourceMeta = getVideoMeta(sourceIndex)
      const sourceAbsoluteTime = syncVideoIntoClipWindow(source, sourceMeta, {
        loopToStart: !source.paused && playVideo,
        forceSeek: options.forceSeek,
      })
      const sourceTime = getRelativePlaybackTime(sourceMeta, sourceAbsoluteTime)
      const sourcePaused = source.paused
      const sourceRate = source.playbackRate

      videoRefs.current.forEach((target, targetIndex) => {
        if (!target || targetIndex === sourceIndex) return
        const targetMeta = getVideoMeta(targetIndex)

        if (target.playbackRate !== sourceRate) {
          target.playbackRate = sourceRate
        }

        const targetAbsoluteTime = clampAbsolutePlaybackTime(
          targetMeta,
          getClipStart(targetMeta) + sourceTime,
          target.duration,
          { loopToStart: !sourcePaused && playVideo },
        )
        const shouldSeek =
          options.forceSeek || Math.abs(target.currentTime - targetAbsoluteTime) > 0.08
        if (shouldSeek) {
          try {
            target.currentTime = targetAbsoluteTime
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
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta, { loopToStart: true })
        updateTimelineFrom(index, absoluteTime, { force: true })
        syncFromSource(index, { forceSeek: true })
      }
      const handlePause = () => {
        if (syncLockRef.current) return
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta)
        updateTimelineFrom(index, absoluteTime, { force: true })
        syncFromSource(index)
      }
      const handleSeeking = () => {
        if (syncLockRef.current) return
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta, { forceSeek: true })
        updateTimelineFrom(index, absoluteTime, { force: true })
        syncFromSource(index, { forceSeek: true })
      }
      const handleSeeked = () => {
        if (syncLockRef.current) return
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta, { forceSeek: true })
        updateTimelineFrom(index, absoluteTime, { force: true })
        syncFromSource(index, { forceSeek: true })
      }
      const handleRateChange = () => {
        if (syncLockRef.current) return
        syncFromSource(index)
      }
      const handleTimeUpdate = () => {
        if (syncLockRef.current) return
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta, {
          loopToStart: playVideo,
          forceSeek: false,
        })
        updateTimelineFrom(index, absoluteTime)
        if (index !== timelineLeaderIndex) {
          return
        }
        syncFromSource(index)
      }
      const handleLoadedMetadata = () => {
        if (syncLockRef.current) return
        const meta = getVideoMeta(index)
        const absoluteTime = syncVideoIntoClipWindow(video, meta, { forceSeek: true })
        updateTimelineFrom(index, absoluteTime, { force: true })
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
  }, [detail, onVideoTimeUpdate, playVideo])

  useEffect(() => {
    const videos = videoRefs.current.filter((video): video is HTMLVideoElement => Boolean(video))
    if (!videos.length) return

    if (!playVideo) {
      videos.forEach((video) => video.pause())
      return
    }

    let attempts = 0
    const tryPlay = () => {
      const currentVideos = videoRefs.current
        .map((video, index) => ({ video, index }))
        .filter((entry): entry is { video: HTMLVideoElement; index: number } => Boolean(entry.video))
      if (!currentVideos.length) {
        return
      }

      currentVideos.forEach(({ video, index }) => {
        syncVideoIntoClipWindow(video, detail.videos[index] ?? null, { forceSeek: true })
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
      const currentVideos = videoRefs.current
        .map((video, index) => ({ video, index }))
        .filter((entry): entry is { video: HTMLVideoElement; index: number } => Boolean(entry.video))
      const allPlaying = currentVideos.length > 0 && currentVideos.every(({ video }) => !video.paused)
      if (allPlaying || attempts >= 12) {
        window.clearInterval(retryTimer)
      }
    }, 120)

    return () => {
      window.clearInterval(retryTimer)
    }
  }, [playVideo, detail])

  useEffect(() => {
    const interval = window.setInterval(() => {
      const entries = videoRefs.current
        .map((video, index) => ({ video, index }))
        .filter((entry): entry is { video: HTMLVideoElement; index: number } => Boolean(entry.video))
      const [leaderEntry, ...followers] = entries
      if (!leaderEntry || followers.length === 0 || syncLockRef.current) {
        return
      }

      const leaderMeta = detail.videos[leaderEntry.index] ?? null
      const leaderAbsoluteTime = syncVideoIntoClipWindow(leaderEntry.video, leaderMeta, {
        loopToStart: playVideo,
      })
      const leaderTime = getRelativePlaybackTime(leaderMeta, leaderAbsoluteTime)
      const leaderPaused = leaderEntry.video.paused || !playVideo
      const leaderRate = leaderEntry.video.playbackRate

      followers.forEach(({ video, index }) => {
        const videoMeta = detail.videos[index] ?? null
        if (video.playbackRate !== leaderRate) {
          video.playbackRate = leaderRate
        }

        const targetAbsoluteTime = clampAbsolutePlaybackTime(
          videoMeta,
          getClipStart(videoMeta) + leaderTime,
          video.duration,
          { loopToStart: !leaderPaused },
        )
        if (Math.abs(video.currentTime - targetAbsoluteTime) > 0.08) {
          try {
            video.currentTime = targetAbsoluteTime
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
  }, [detail, playVideo])

  return (
    <div className="explorer-hover-preview__body explorer-episode-playback">
      <div className="explorer-hover-preview__video-grid">
        {detail.videos.length > 0 ? (
          detail.videos.map((video, index) => {
            const clipLabel = formatClipWindowLabel(video)
            return (
              <div key={video.path} className="explorer-hover-preview__video-card">
                <div className="explorer-hover-preview__status">
                  <strong>{video.stream}</strong>
                  {clipLabel ? <span> · {clipLabel}</span> : null}
                </div>
                <video
                  ref={(node) => {
                    videoRefs.current[index] = node
                  }}
                  src={video.url}
                  autoPlay={playVideo}
                  controls
                  muted
                  loop={shouldLoopVideo(video)}
                  playsInline
                  preload="metadata"
                />
              </div>
            )
          })
        ) : (
          <div className="explorer-hover-preview__empty">{emptyLabel}</div>
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
  )
}

function EpisodeHoverPreview({
  detail,
  loading,
  trajectoryLoading,
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
  trajectoryLoading: boolean
  error: string
  playVideo: boolean
  videoCurrentTime: number
  onVideoTimeUpdate: (seconds: number) => void
  onClose: () => void
  onMouseEnter: () => void
  onMouseLeave: () => void
}) {
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

            <EpisodePlaybackSurface
              detail={detail}
              playVideo={playVideo}
              videoCurrentTime={videoCurrentTime}
              onVideoTimeUpdate={onVideoTimeUpdate}
              emptyLabel="No video stream available for this episode."
            />

            {!hasTrajectoryData(detail) && trajectoryLoading && (
              <div className="explorer-hover-preview__empty">
                Loading trajectory comparison...
              </div>
            )}
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function buildExplorerQuery(ref: ExplorerDatasetRef): string {
  const params = new URLSearchParams()
  params.set('source', ref.source)
  if (ref.dataset) {
    params.set('dataset', ref.dataset)
  }
  if (ref.path) {
    params.set('path', ref.path)
  }
  return params.toString()
}

function EpisodeBrowser({ datasetRef }: { datasetRef: ExplorerDatasetRef }) {
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
  const selectedDataset = datasetRef.dataset ?? episodePage?.dataset ?? ''
  const hoverTimerRef = useRef<number | null>(null)
  const closeTimerRef = useRef<number | null>(null)
  const playReadyTimerRef = useRef<number | null>(null)
  const requestTokenRef = useRef(0)
  const previewCacheRef = useRef<Map<number, EpisodeDetail>>(new Map())
  const previewDetailStateRef = useRef<Map<number, 'loading' | 'loaded'>>(new Map())
  const hoverRequestAbortRef = useRef<AbortController | null>(null)
  const [hoveredEpisodeIndex, setHoveredEpisodeIndex] = useState<number | null>(null)
  const [hoveredPreview, setHoveredPreview] = useState<EpisodeDetail | null>(null)
  const [hoveredPreviewLoading, setHoveredPreviewLoading] = useState(false)
  const [hoveredPreviewTrajectoryLoading, setHoveredPreviewTrajectoryLoading] = useState(false)
  const [hoveredPreviewError, setHoveredPreviewError] = useState('')
  const [previewPlayReady, setPreviewPlayReady] = useState(false)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)
  const [detailVideoCurrentTime, setDetailVideoCurrentTime] = useState(0)

  useEffect(() => {
    previewCacheRef.current.clear()
    previewDetailStateRef.current.clear()
    hoverRequestAbortRef.current?.abort()
    hoverRequestAbortRef.current = null
    if (playReadyTimerRef.current) {
      window.clearTimeout(playReadyTimerRef.current)
      playReadyTimerRef.current = null
    }
    setHoveredEpisodeIndex(null)
    setHoveredPreview(null)
    setHoveredPreviewLoading(false)
    setHoveredPreviewTrajectoryLoading(false)
    setHoveredPreviewError('')
    setPreviewPlayReady(false)
    setVideoCurrentTime(0)
    setDetailVideoCurrentTime(0)
  }, [selectedDataset, datasetRef.path, datasetRef.source])

  useEffect(() => {
    setDetailVideoCurrentTime(0)
  }, [episodeDetail?.episode_index])

  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        window.clearTimeout(hoverTimerRef.current)
      }
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current)
      }
      if (playReadyTimerRef.current) {
        window.clearTimeout(playReadyTimerRef.current)
      }
      hoverRequestAbortRef.current?.abort()
    }
  }, [])

  useEffect(() => {
    if (
      hoveredEpisodeIndex === null ||
      !hoveredPreview ||
      hasTrajectoryData(hoveredPreview) ||
      previewDetailStateRef.current.get(hoveredEpisodeIndex) === 'loaded'
    ) {
      return
    }

    void hydrateHoverPreviewDetail(datasetRef, hoveredEpisodeIndex, requestTokenRef.current)
  }, [hoveredEpisodeIndex, hoveredPreview, datasetRef, selectedDataset])

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
    if (playReadyTimerRef.current) {
      window.clearTimeout(playReadyTimerRef.current)
      playReadyTimerRef.current = null
    }
    closeTimerRef.current = window.setTimeout(() => {
      hoverRequestAbortRef.current?.abort()
      hoverRequestAbortRef.current = null
      previewDetailStateRef.current.forEach((state, key) => {
        if (state === 'loading') {
          previewDetailStateRef.current.delete(key)
        }
      })
      setHoveredEpisodeIndex(null)
      setHoveredPreview(null)
      setHoveredPreviewLoading(false)
      setHoveredPreviewTrajectoryLoading(false)
      setHoveredPreviewError('')
      setPreviewPlayReady(false)
      setVideoCurrentTime(0)
    }, 180)
  }

  const armPreviewPlayback = (requestToken: number, delayMs = 180) => {
    if (playReadyTimerRef.current) {
      window.clearTimeout(playReadyTimerRef.current)
    }
    playReadyTimerRef.current = window.setTimeout(() => {
      if (requestToken === requestTokenRef.current) {
        setPreviewPlayReady(true)
      }
    }, delayMs)
  }

  const hydrateHoverPreviewDetail = async (
    ref: ExplorerDatasetRef,
    episodeIndex: number,
    requestToken: number,
  ) => {
    const currentState = previewDetailStateRef.current.get(episodeIndex)
    if (currentState === 'loading' || currentState === 'loaded') {
      return
    }

    const controller = new AbortController()
    hoverRequestAbortRef.current = controller
    previewDetailStateRef.current.set(episodeIndex, 'loading')
    if (requestToken === requestTokenRef.current) {
      setHoveredPreviewTrajectoryLoading(true)
    }

    try {
      const response = await fetch(
        `/api/explorer/episode?${buildExplorerQuery(ref)}&episode_index=${episodeIndex}`,
        { signal: controller.signal },
      )
      if (!response.ok) {
        throw new Error(`Failed to load trajectory comparison (${response.status})`)
      }
      const detail: EpisodeDetail = await response.json()
      previewCacheRef.current.set(episodeIndex, detail)
      previewDetailStateRef.current.set(episodeIndex, 'loaded')
      if (requestToken === requestTokenRef.current) {
        setHoveredPreview(detail)
        armPreviewPlayback(requestToken)
      }
    } catch (error) {
      previewDetailStateRef.current.delete(episodeIndex)
      if (
        error instanceof DOMException &&
        error.name === 'AbortError'
      ) {
        return
      }
      if (requestToken === requestTokenRef.current) {
        setHoveredPreviewError(
          error instanceof Error ? error.message : 'Failed to load trajectory comparison',
        )
        armPreviewPlayback(requestToken)
      }
    } finally {
      if (requestToken === requestTokenRef.current) {
        setHoveredPreviewTrajectoryLoading(false)
      }
    }
  }

  const scheduleHoverPreview = (episodeIndex: number) => {
    if (
      hoveredEpisodeIndex === episodeIndex &&
      (hoveredPreview !== null || hoveredPreviewLoading || hoveredPreviewTrajectoryLoading)
    ) {
      cancelClosePreview()
      return
    }

    cancelClosePreview()
    if (hoverTimerRef.current) {
      window.clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
    if (playReadyTimerRef.current) {
      window.clearTimeout(playReadyTimerRef.current)
      playReadyTimerRef.current = null
    }
    hoverRequestAbortRef.current?.abort()
    previewDetailStateRef.current.forEach((state, key) => {
      if (state === 'loading') {
        previewDetailStateRef.current.delete(key)
      }
    })
    setHoveredPreviewError('')
    setPreviewPlayReady(false)
    setVideoCurrentTime(0)
    setHoveredPreviewTrajectoryLoading(false)

    hoverTimerRef.current = window.setTimeout(async () => {
      const controller = new AbortController()
      hoverRequestAbortRef.current = controller
      const requestToken = ++requestTokenRef.current
      setHoveredEpisodeIndex(episodeIndex)
      setPreviewPlayReady(true)

      const cached = previewCacheRef.current.get(episodeIndex)
      if (cached) {
        setHoveredPreview(cached)
        setHoveredPreviewLoading(false)
        if (hasTrajectoryData(cached) || previewDetailStateRef.current.get(episodeIndex) === 'loaded') {
          armPreviewPlayback(requestToken)
        }
        return
      }

      setHoveredPreview(null)
      setHoveredPreviewLoading(true)

      try {
        const response = await fetch(
          `/api/explorer/episode?${buildExplorerQuery(datasetRef)}&episode_index=${episodeIndex}&preview=1`,
          { signal: controller.signal },
        )
        if (!response.ok) {
          throw new Error(`Failed to load episode preview (${response.status})`)
        }
        const detail: EpisodeDetail = await response.json()
        previewCacheRef.current.set(episodeIndex, detail)
        if (requestToken === requestTokenRef.current) {
          setHoveredPreview(detail)
        }
        if (hasTrajectoryData(detail)) {
          previewDetailStateRef.current.set(episodeIndex, 'loaded')
          armPreviewPlayback(requestToken)
        }
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === 'AbortError'
        ) {
          return
        }
        if (requestToken === requestTokenRef.current) {
          setHoveredPreviewError(error instanceof Error ? error.message : 'Failed to load preview')
          armPreviewPlayback(requestToken)
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
            onClick={() => void loadEpisodePage(datasetRef, episodePage!.page - 1, episodePage!.page_size)}
          >
            Prev
          </button>
          <button
            type="button"
            className="explorer-episodes__pager"
            disabled={episodePage!.page >= episodePage!.total_pages || episodePageLoading}
            onClick={() => void loadEpisodePage(datasetRef, episodePage!.page + 1, episodePage!.page_size)}
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
                void selectEpisode(datasetRef, ep.episode_index)
              }
            }}
            onMouseEnter={() => scheduleHoverPreview(ep.episode_index)}
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
          trajectoryLoading={hoveredPreviewTrajectoryLoading}
          error={hoveredPreviewError}
          videoCurrentTime={videoCurrentTime}
          onVideoTimeUpdate={setVideoCurrentTime}
          onClose={() => {
            setHoveredEpisodeIndex(null)
            setHoveredPreview(null)
            setHoveredPreviewLoading(false)
            setHoveredPreviewTrajectoryLoading(false)
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
              <h4>Playback</h4>
              <EpisodePlaybackSurface
                detail={episodeDetail}
                playVideo
                videoCurrentTime={detailVideoCurrentTime}
                onVideoTimeUpdate={setDetailVideoCurrentTime}
                emptyLabel="No video stream available for this episode."
              />
            </div>
          )}

          {episodeDetail.videos.length > 0 && (
            <div className="explorer-episode-detail__section">
              <h4>Video Sources</h4>
              <ul className="explorer-video-list">
                {episodeDetail.videos.map((v) => (
                  <li key={v.path}>
                    <strong>{v.stream}</strong> — {v.path}
                    {formatClipWindowLabel(v) ? ` (${formatClipWindowLabel(v)})` : ''}
                  </li>
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
  const navigate = useNavigate()
  const { prepareRemoteDatasetForWorkflow, createLocalDirectorySession } = useWorkflow()
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
  const [source, setSource] = useState<ExplorerSource>('remote')
  const [datasetIdInput, setDatasetIdInput] = useState('')
  const [remoteDatasetSelected, setRemoteDatasetSelected] = useState('')
  const [localDatasetInput, setLocalDatasetInput] = useState('')
  const [localDatasetPathInput, setLocalDatasetPathInput] = useState('')
  const [localDatasetPathSelected, setLocalDatasetPathSelected] = useState('')
  const [localPathDatasetLabel, setLocalPathDatasetLabel] = useState('')
  const [localDatasets, setLocalDatasets] = useState<DatasetSuggestion[]>([])
  const [datasetSuggestions, setDatasetSuggestions] = useState<DatasetSuggestion[]>([])
  const [suggestionsOpen, setSuggestionsOpen] = useState(false)
  const [suggestionsLoading, setSuggestionsLoading] = useState(false)
  const [highlightedSuggestionIndex, setHighlightedSuggestionIndex] = useState(-1)
  const [prepareStatus, setPrepareStatus] = useState('')
  const [prepareError, setPrepareError] = useState('')
  const [preparingForQuality, setPreparingForQuality] = useState(false)
  const datasetInputRef = useRef<HTMLInputElement | null>(null)
  const localDirectoryInputRef = useRef<HTMLInputElement | null>(null)
  const blurTimerRef = useRef<number | null>(null)
  const suggestionRequestRef = useRef(0)
  const requestedDatasetKeyRef = useRef('')
  const preparingForQualityRef = useRef(false)
  const currentDataset = summary?.dataset || dashboard?.dataset || episodePage?.dataset || ''

  const datasetRef: ExplorerDatasetRef =
    source === 'remote'
      ? { source, dataset: remoteDatasetSelected.trim() || currentDataset }
      : source === 'local'
        ? { source, dataset: localDatasetInput.trim() || currentDataset }
        : {
            source,
            dataset: localPathDatasetLabel.trim() || undefined,
            path: localDatasetPathSelected.trim() || undefined,
          }

  function buildDatasetRequestKey(ref: ExplorerDatasetRef): string {
    return `${ref.source}|${ref.dataset?.trim() ?? ''}|${ref.path?.trim() ?? ''}`
  }

  async function loadDataset(ref: ExplorerDatasetRef): Promise<void> {
    requestedDatasetKeyRef.current = buildDatasetRequestKey(ref)
    await Promise.allSettled([
      loadSummary(ref),
      loadDashboard(ref),
      loadEpisodePage(ref, 1, 50),
    ])
  }

  useEffect(() => {
    if (source !== 'local') {
      return
    }
    void listExplorerDatasets('local')
      .then((items) => setLocalDatasets(items))
      .catch(() => setLocalDatasets([]))
  }, [source])

  useEffect(() => {
    const activeDataset = datasetRef.dataset?.trim() ?? ''
    const activePath = datasetRef.path?.trim() ?? ''
    const requestKey = buildDatasetRequestKey(datasetRef)
    if (!activeDataset && !activePath) {
      return
    }
    if (requestedDatasetKeyRef.current === requestKey) {
      return
    }
    void loadDataset(datasetRef)
  }, [
    source,
    remoteDatasetSelected,
    localDatasetInput,
    localDatasetPathSelected,
    localPathDatasetLabel,
    currentDataset,
    loadSummary,
    loadDashboard,
    loadEpisodePage,
  ])

  useEffect(() => {
    return () => {
      if (blurTimerRef.current != null) {
        window.clearTimeout(blurTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    function handlePipelineEvent(event: Event): void {
      const detail = (event as CustomEvent<Record<string, unknown>>).detail
      if (!detail || detail.type !== 'pipeline.dataset_prepared') {
        return
      }
      const sourceDataset =
        typeof detail.source_dataset === 'string' && detail.source_dataset.trim()
          ? detail.source_dataset.trim()
          : typeof detail.dataset_id === 'string'
            ? detail.dataset_id.trim()
            : ''
      if (!sourceDataset) {
        return
      }

      const preparedName = typeof detail.dataset_name === 'string' ? detail.dataset_name : ''
      const nextRef: ExplorerDatasetRef = { source: 'remote', dataset: sourceDataset }
      requestedDatasetKeyRef.current = buildDatasetRequestKey(nextRef)
      setSource('remote')
      setDatasetIdInput(sourceDataset)
      setRemoteDatasetSelected(sourceDataset)
      setPrepareError('')
      setPrepareStatus(
        preparedName
          ? `${t('preparedForQuality')}: ${preparedName}`
          : `${t('preparedForQuality')}: ${sourceDataset}`,
      )
      void Promise.allSettled([
        loadSummary(nextRef),
        loadDashboard(nextRef),
        loadEpisodePage(nextRef, 1, 50),
      ])
    }

    window.addEventListener('roboclaw:pipeline-event', handlePipelineEvent)
    return () => window.removeEventListener('roboclaw:pipeline-event', handlePipelineEvent)
  }, [loadDashboard, loadEpisodePage, loadSummary, t])

  useEffect(() => {
    if (source !== 'remote') {
      setDatasetSuggestions([])
      setSuggestionsLoading(false)
      setSuggestionsOpen(false)
      setHighlightedSuggestionIndex(-1)
      return
    }
    const needle = datasetIdInput.trim()
    if (needle.length < 2 || needle === currentDataset.trim()) {
      suggestionRequestRef.current += 1
      setDatasetSuggestions([])
      setSuggestionsLoading(false)
      setSuggestionsOpen(false)
      setHighlightedSuggestionIndex(-1)
      return
    }

    const requestId = suggestionRequestRef.current + 1
    suggestionRequestRef.current = requestId
    setSuggestionsLoading(true)
    const timer = window.setTimeout(() => {
      void searchDatasetSuggestions(needle, source, 8)
        .then((items) => {
          if (suggestionRequestRef.current !== requestId) {
            return
          }
          setDatasetSuggestions(items)
          setHighlightedSuggestionIndex(items.length > 0 ? 0 : -1)
          if (document.activeElement === datasetInputRef.current) {
            setSuggestionsOpen(true)
          }
        })
        .catch(() => {
          if (suggestionRequestRef.current !== requestId) {
            return
          }
          setDatasetSuggestions([])
          setHighlightedSuggestionIndex(-1)
        })
        .finally(() => {
          if (suggestionRequestRef.current === requestId) {
            setSuggestionsLoading(false)
          }
        })
    }, 180)

    return () => {
      window.clearTimeout(timer)
    }
  }, [datasetIdInput, currentDataset, source])

  async function handleLoad(
    override?: Partial<ExplorerDatasetRef> & { datasetOverride?: string },
  ): Promise<void> {
    const nextSource = override?.source ?? source
    const nextDataset =
      override?.datasetOverride
      ?? override?.dataset
      ?? (nextSource === 'remote'
        ? datasetIdInput
        : nextSource === 'local'
          ? localDatasetInput
          : currentDataset)
    const nextPath = override?.path ?? (nextSource === 'path' ? localDatasetPathInput : undefined)
    const nextRef: ExplorerDatasetRef = {
      source: nextSource,
      dataset: nextDataset?.trim() || undefined,
      path: nextPath?.trim() || undefined,
    }
    if (!nextRef.dataset && !nextRef.path) {
      return
    }
    if (nextSource === 'remote' && nextRef.dataset) {
      setDatasetIdInput(nextRef.dataset)
      setRemoteDatasetSelected(nextRef.dataset)
    }
    if (nextSource === 'local' && nextRef.dataset) {
      setLocalDatasetInput(nextRef.dataset)
    }
    if (nextSource === 'path' && nextRef.path) {
      setLocalDatasetPathInput(nextRef.path)
      setLocalDatasetPathSelected(nextRef.path)
      setLocalPathDatasetLabel(nextRef.dataset ?? '')
    }
    setSuggestionsOpen(false)
    setDatasetSuggestions([])
    setHighlightedSuggestionIndex(-1)
    await loadDataset(nextRef)
  }

  function openSuggestions(): void {
    if (blurTimerRef.current != null) {
      window.clearTimeout(blurTimerRef.current)
      blurTimerRef.current = null
    }
    if (datasetIdInput.trim().length >= 2) {
      setSuggestionsOpen(true)
    }
  }

  function closeSuggestionsSoon(): void {
    if (blurTimerRef.current != null) {
      window.clearTimeout(blurTimerRef.current)
    }
    blurTimerRef.current = window.setTimeout(() => {
      setSuggestionsOpen(false)
    }, 120)
  }

  async function handleSuggestionSelect(datasetId: string): Promise<void> {
    await handleLoad({ source: 'remote', datasetOverride: datasetId })
  }

  async function handleInputKeyDown(
    event: KeyboardEvent<HTMLInputElement>,
  ): Promise<void> {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      if (!suggestionsOpen) {
        openSuggestions()
      }
      if (datasetSuggestions.length > 0) {
        setHighlightedSuggestionIndex((current) => (current + 1) % datasetSuggestions.length)
      }
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      if (!suggestionsOpen) {
        openSuggestions()
      }
      if (datasetSuggestions.length > 0) {
        setHighlightedSuggestionIndex((current) =>
          current <= 0 ? datasetSuggestions.length - 1 : current - 1,
        )
      }
      return
    }
    if (event.key === 'Escape') {
      setSuggestionsOpen(false)
      setHighlightedSuggestionIndex(-1)
      return
    }
    if (event.key === 'Enter') {
      event.preventDefault()
      const highlighted = datasetSuggestions[highlightedSuggestionIndex]
      if (suggestionsOpen && highlighted) {
        await handleSuggestionSelect(highlighted.id)
        return
      }
      await handleLoad()
    }
  }

  async function handlePrepareRemote(): Promise<void> {
    if (preparingForQualityRef.current) return
    const datasetId = datasetIdInput.trim()
    if (!datasetId) return
    preparingForQualityRef.current = true
    setPreparingForQuality(true)
    setPrepareStatus(t('preparingForQuality'))
    setPrepareError('')
    try {
      setRemoteDatasetSelected(datasetId)
      const payload = await prepareRemoteDatasetForWorkflow(datasetId, false)
      setPrepareStatus(`${t('preparedForQuality')}: ${payload.dataset_name}`)
      navigate('/curation/quality')
    } catch (error) {
      setPrepareError(error instanceof Error ? error.message : t('qualityRunFailed'))
    } finally {
      preparingForQualityRef.current = false
      setPreparingForQuality(false)
    }
  }

  async function handleChooseLocalDirectory(
    event: React.ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const files = Array.from(event.target.files || [])
    if (files.length === 0) {
      return
    }
    const relativePaths = files.map((file) => {
      const maybeRelative = (file as File & { webkitRelativePath?: string }).webkitRelativePath
      return maybeRelative && maybeRelative.trim() ? maybeRelative : file.name
    })
    const displayName = relativePaths[0]?.split('/')[0] || files[0].name
    setPrepareStatus(t('localDirectoryUploading'))
    setPrepareError('')
    try {
      const payload = await createLocalDirectorySession(files, relativePaths, displayName)
      setLocalDatasetPathInput(payload.local_path)
      setLocalDatasetPathSelected(payload.local_path)
      setLocalPathDatasetLabel(payload.dataset_name)
      setPrepareStatus(payload.display_name)
      await handleLoad({
        source: 'path',
        path: payload.local_path,
        datasetOverride: payload.dataset_name,
      })
    } catch (error) {
      setPrepareError(error instanceof Error ? error.message : t('qualityRunFailed'))
    } finally {
      event.target.value = ''
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
          <label className="dataset-workbench__control">
            <span>{t('dataSource')}</span>
            <select
              className="dataset-workbench__select"
              value={source}
              onChange={(event) => {
                const nextSource = event.target.value as ExplorerSource
                setSource(nextSource)
                setDatasetSuggestions([])
                setSuggestionsOpen(false)
                setHighlightedSuggestionIndex(-1)
                requestedDatasetKeyRef.current = ''
              }}
            >
              <option value="remote">{t('remoteDataset')}</option>
              <option value="local">{t('localDataset')}</option>
              <option value="path">{t('localDirectory')}</option>
            </select>
          </label>

          {source === 'remote' && (
          <label className="dataset-workbench__control dataset-workbench__control--wide">
            <span>{t('hfDatasetId')}</span>
            <div className="dataset-workbench__combobox">
              <input
                ref={datasetInputRef}
                className="dataset-workbench__input"
                type="text"
                value={datasetIdInput}
                onChange={(event) => {
                  const nextValue = event.target.value
                  setDatasetIdInput(nextValue)
                  if (nextValue.trim() !== remoteDatasetSelected.trim()) {
                    setRemoteDatasetSelected('')
                  }
                }}
                onFocus={openSuggestions}
                onBlur={closeSuggestionsSoon}
                onKeyDown={(event) => {
                  void handleInputKeyDown(event)
                }}
                placeholder={t('hfDatasetPlaceholder')}
                role="combobox"
                aria-autocomplete="list"
                aria-expanded={suggestionsOpen}
                aria-controls="explorer-dataset-suggestions"
                aria-activedescendant={
                  highlightedSuggestionIndex >= 0
                    ? `explorer-dataset-suggestion-${highlightedSuggestionIndex}`
                    : undefined
                }
              />
              {suggestionsOpen && (suggestionsLoading || datasetSuggestions.length > 0 || datasetIdInput.trim().length >= 2) && (
                <div className="dataset-workbench__suggestions" id="explorer-dataset-suggestions" role="listbox">
                  {suggestionsLoading ? (
                    <div className="dataset-workbench__suggestion-status">
                      {t('datasetSuggestionsLoading')}
                    </div>
                  ) : datasetSuggestions.length > 0 ? (
                    datasetSuggestions.map((suggestion, index) => (
                      <button
                        key={suggestion.id}
                        id={`explorer-dataset-suggestion-${index}`}
                        type="button"
                        role="option"
                        aria-selected={index === highlightedSuggestionIndex}
                        className={cn(
                          'dataset-workbench__suggestion',
                          index === highlightedSuggestionIndex && 'is-active',
                        )}
                        onMouseDown={(event) => event.preventDefault()}
                        onMouseEnter={() => setHighlightedSuggestionIndex(index)}
                        onClick={() => {
                          void handleSuggestionSelect(suggestion.id)
                        }}
                      >
                        {suggestion.id}
                      </button>
                    ))
                  ) : (
                    <div className="dataset-workbench__suggestion-status">
                      {t('noDatasetSuggestions')}
                    </div>
                  )}
                </div>
              )}
            </div>
          </label>
          )}

          {source === 'local' && (
            <label className="dataset-workbench__control dataset-workbench__control--wide">
              <span>{t('localDataset')}</span>
              <select
                className="dataset-workbench__select"
                value={localDatasetInput}
                onChange={(event) => setLocalDatasetInput(event.target.value)}
              >
                <option value="">{t('selectDataset')}</option>
                {localDatasets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label || item.id}
                  </option>
                ))}
              </select>
            </label>
          )}

          {source === 'path' && (
            <label className="dataset-workbench__control dataset-workbench__control--wide">
              <span>{t('localDirectory')}</span>
              <div className="dataset-workbench__controls">
                <ActionButton
                  type="button"
                  variant="secondary"
                  onClick={() => localDirectoryInputRef.current?.click()}
                  className="dataset-workbench__import-btn"
                >
                  {t('chooseLocalDirectory')}
                </ActionButton>
                <input
                  className="dataset-workbench__input"
                  type="text"
                  value={localDatasetPathInput}
                  onChange={(event) => setLocalDatasetPathInput(event.target.value)}
                  placeholder={t('localPathPlaceholder')}
                />
                <input
                  ref={localDirectoryInputRef}
                  type="file"
                  multiple
                  hidden
                  // @ts-expect-error vendor directory picker attribute
                  webkitdirectory=""
                  onChange={(event) => {
                    void handleChooseLocalDirectory(event)
                  }}
                />
              </div>
            </label>
          )}

          <ActionButton
            type="button"
            variant="secondary"
            onClick={() => void handleLoad()}
            disabled={
              preparingForQuality
              || (source === 'remote'
                ? !datasetIdInput.trim()
                : source === 'local'
                  ? !localDatasetInput.trim()
                  : !localDatasetPathInput.trim())
            }
            className="dataset-workbench__import-btn"
          >
            {t('browseDataset')}
          </ActionButton>

          {source === 'remote' && (
            <ActionButton
              type="button"
              variant="secondary"
              onClick={() => void handlePrepareRemote()}
              disabled={!datasetIdInput.trim() || preparingForQuality}
              className="dataset-workbench__import-btn"
            >
              {preparingForQuality ? t('preparingForQuality') : t('prepareForQuality')}
            </ActionButton>
          )}
        </div>
        {(prepareStatus || prepareError) && (
          <div className={`dataset-workbench__status ${prepareError ? 'is-error' : ''}`}>
            {prepareError || prepareStatus}
          </div>
        )}
      </div>

      {/* Info bar */}
      {currentDataset && datasetSummary ? (
        <div className="workflow-view__info-bar">
          <span>{prepareStatus || summary!.dataset}</span>
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
          {source === 'remote' ? t('hfDatasetPlaceholder') : t('chooseLocalDirectory')}
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
              <EpisodeBrowser datasetRef={datasetRef} />
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
