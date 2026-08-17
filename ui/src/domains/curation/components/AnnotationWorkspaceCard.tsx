import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MutableRefObject,
  type PointerEvent,
} from 'react'
import { useI18n } from '@/i18n'
import type { AnnotationItem } from '@/domains/curation/store/useCurationStore'

const BASE_PIXELS_PER_SECOND = 120
const MIN_PIXELS_PER_SECOND = 36
const MAX_PIXELS_PER_SECOND = 360
const TIMELINE_PADDING = 56
const FOLLOW_PADDING = 180
const TIMELINE_RULER_HEIGHT = 48
const TIMELINE_LANE_ROW_HEIGHT = 44

interface LaneDefinition {
  id: string
  laneLabel: string
  cardLabel: string
  color: string
  tags: string[]
}

interface AnnotationWorkspaceCardProps {
  videoRef: MutableRefObject<HTMLVideoElement | null>
  videoSource: string
  videoTitle: string
  fps: number
  streamLabel: string
  chunkLabel: string
  currentFrame: number | null
  isPaused: boolean
  videoCurrentTime: number
  timelineDuration: number
  annotations: AnnotationItem[]
  selectedAnnotationId: string | null
  onSelectAnnotation: (annotationId: string) => void
  onCreateAnnotation: (seedTime?: number) => void
  onUpdateAnnotation: (annotationId: string, patch: Partial<AnnotationItem>) => void
  onDeleteAnnotation: (annotationId: string) => void
  onJumpToTime: (timeValue: number) => void
}

function buildLaneDefinitions(locale: 'zh' | 'en'): LaneDefinition[] {
  if (locale === 'zh') {
    return [
      {
        id: 'movement',
        laneLabel: '标注',
        cardLabel: '核心标注',
        color: '#ff8a5b',
        tags: ['时间段', '自然语言'],
      },
    ]
  }

  return [
    {
      id: 'movement',
      laneLabel: 'Annotation',
      cardLabel: 'Core Annotation',
      color: '#ff8a5b',
      tags: ['Time Span', 'Language'],
    },
  ]
}

function clampValue(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function trimText(value: string | undefined, maxLength = 132): string {
  const normalized = (value || '').trim()
  if (!normalized) return ''
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength).trimEnd()}...`
}

function formatClock(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds)) return '00:00'
  const roundedSeconds = Math.max(totalSeconds, 0)
  const wholeSeconds = Math.floor(roundedSeconds)
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const seconds = wholeSeconds % 60

  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  }

  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function formatTimecode(totalSeconds: number, fps = 30): string {
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 30
  const boundedSeconds = Math.max(totalSeconds || 0, 0)
  const wholeSeconds = Math.floor(boundedSeconds)
  const frame = Math.floor((boundedSeconds - wholeSeconds) * safeFps)
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const seconds = wholeSeconds % 60

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}:${String(frame).padStart(2, '0')}`
}

function roundToHundredths(value: number): number {
  return Math.round(value * 100) / 100
}

function formatRangeField(value: number): string {
  if (!Number.isFinite(value)) return '0.00'
  return roundToHundredths(Math.max(value, 0)).toFixed(2)
}

function getTickSteps(pixelsPerSecond: number): { major: number; minor: number } {
  if (pixelsPerSecond >= 280) return { major: 0.5, minor: 0.1 }
  if (pixelsPerSecond >= 160) return { major: 1, minor: 0.2 }
  if (pixelsPerSecond >= 96) return { major: 1, minor: 0.5 }
  if (pixelsPerSecond >= 60) return { major: 2, minor: 0.5 }
  return { major: 5, minor: 1 }
}

function buildTicks(
  duration: number,
  majorStep: number,
  minorStep: number,
): Array<{ time: number; isMajor: boolean }> {
  const safeDuration = Math.max(duration, 0)
  const ticks = []
  const totalSteps = Math.floor(safeDuration / minorStep)

  for (let step = 0; step <= totalSteps; step += 1) {
    const time = Number((step * minorStep).toFixed(6))
    const isMajor =
      Math.abs(time / majorStep - Math.round(time / majorStep)) < 0.0001
    ticks.push({ time, isMajor })
  }

  if (!ticks.length || ticks[ticks.length - 1].time < safeDuration) {
    ticks.push({ time: safeDuration, isMajor: true })
  }

  return ticks
}

function parseNumericInput(
  rawValue: string,
  fallbackValue: number,
  minValue: number,
  maxValue: number,
): number {
  const parsedValue = Number(rawValue)
  if (!Number.isFinite(parsedValue)) {
    return roundToHundredths(clampValue(fallbackValue, minValue, maxValue))
  }
  return roundToHundredths(clampValue(parsedValue, minValue, maxValue))
}

function getLaneDefinition(
  laneDefinitions: LaneDefinition[],
  category: string,
): LaneDefinition {
  return laneDefinitions.find((definition) => definition.id === category) || laneDefinitions[0]
}

function TimeInput({
  value,
  min,
  max,
  onCommit,
}: {
  value: number
  min: number
  max: number
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const committed = formatRangeField(value)

  function commit(raw: string): void {
    setDraft(null)
    const parsed = parseNumericInput(raw, value, min, max)
    if (parsed !== roundToHundredths(clampValue(value, min, max))) {
      onCommit(parsed)
    }
  }

  return (
    <input
      type="text"
      inputMode="decimal"
      value={draft ?? committed}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={(event) => commit(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.currentTarget.blur()
        } else if (event.key === 'Escape') {
          setDraft(null)
          event.currentTarget.blur()
        }
      }}
    />
  )
}

export default function AnnotationWorkspaceCard({
  videoRef,
  videoSource,
  videoTitle,
  fps,
  streamLabel,
  chunkLabel,
  currentFrame,
  isPaused,
  videoCurrentTime,
  timelineDuration,
  annotations,
  selectedAnnotationId,
  onSelectAnnotation,
  onCreateAnnotation,
  onUpdateAnnotation,
  onDeleteAnnotation,
  onJumpToTime,
}: AnnotationWorkspaceCardProps) {
  const { locale } = useI18n()
  const copy = locale === 'zh'
    ? {
        noPlayableVideo: '当前流没有可播放视频。',
        liveReview: '实时复核',
        unknownStream: '未知流',
        videoFeed: '视频流',
        stream: '流',
        file: '文件',
        chunk: '分块',
        frame: '帧',
        activeAnnotations: '当前标注',
        activeAnnotationsDesc: '与共享播放光标联动的紧凑事件卡片。',
        createNew: '新建标注',
        noAnnotations: '当前还没有标注。点击“新建标注”可在当前位置创建标注窗口。',
        label: '标签',
        lane: '泳道',
        start: '开始',
        end: '结束',
        description: '描述',
        jumpTo: '跳转到此处',
        useCursor: '使用光标时间',
        delete: '删除',
        mainTimeline: '主时间轴',
        video: '视频',
        unknown: '-',
        mark: '标记',
        createNote: '创建标注',
        track: '跟随',
        followCursor: '跟随光标',
        fit: '适配',
        resetZoom: '重置缩放',
      }
    : {
        noPlayableVideo: 'No playable video for the current stream.',
        liveReview: 'Live Review',
        unknownStream: 'Unknown Stream',
        videoFeed: 'Video Feed',
        stream: 'Stream',
        file: 'File',
        chunk: 'Chunk',
        frame: 'Frame',
        activeAnnotations: 'Active Annotations',
        activeAnnotationsDesc: 'Compact event cards linked to the shared playback cursor.',
        createNew: 'Create New',
        noAnnotations: 'No active annotations yet. Use Create New to mark the current playback window.',
        label: 'Label',
        lane: 'Lane',
        start: 'Start',
        end: 'End',
        description: 'Description',
        jumpTo: 'Jump To',
        useCursor: 'Use Cursor',
        delete: 'Delete',
        mainTimeline: 'Main Timeline',
        video: 'Video',
        unknown: '-',
        mark: 'Mark',
        createNote: 'Create note',
        track: 'Track',
        followCursor: 'Follow cursor',
        fit: 'Fit',
        resetZoom: 'Reset zoom',
      }

  const laneDefinitions = useMemo(() => buildLaneDefinitions(locale), [locale])
  const [pixelsPerSecond, setPixelsPerSecond] = useState(BASE_PIXELS_PER_SECOND)
  const [followPlayback, setFollowPlayback] = useState(true)
  const [draggingAnnotationId, setDraggingAnnotationId] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const dragStateRef = useRef<{
    annotationId: string
    startClientX: number
    originalStartTime: number
    originalEndTime: number
    originalCategory: string
    pixelsPerSecond: number
    safeDuration: number
  } | null>(null)

  const safeDuration = Math.max(timelineDuration, 1)
  const currentTime = Number.isFinite(videoCurrentTime) ? videoCurrentTime : 0
  const sortedAnnotations = useMemo(
    () => [...annotations].sort((left, right) => left.startTime - right.startTime),
    [annotations],
  )
  const tickSteps = useMemo(() => getTickSteps(pixelsPerSecond), [pixelsPerSecond])
  const ticks = useMemo(
    () => buildTicks(safeDuration, tickSteps.major, tickSteps.minor),
    [safeDuration, tickSteps.major, tickSteps.minor],
  )
  const timelineWidth = Math.max(
    safeDuration * pixelsPerSecond + TIMELINE_PADDING * 2,
    1120,
  )
  const lanes = useMemo(
    () =>
      laneDefinitions.map((definition) => ({
        ...definition,
        annotations: sortedAnnotations.filter(
          (annotation) => annotation.category === definition.id,
        ),
      })),
    [laneDefinitions, sortedAnnotations],
  )
  const timelineCanvasHeight =
    TIMELINE_RULER_HEIGHT + TIMELINE_LANE_ROW_HEIGHT * lanes.length

  useEffect(() => {
    if (!followPlayback || !scrollRef.current) return

    const scrollElement = scrollRef.current
    const cursorLeft =
      TIMELINE_PADDING + clampValue(currentTime, 0, safeDuration) * pixelsPerSecond
    const minVisible = scrollElement.scrollLeft + FOLLOW_PADDING
    const maxVisible =
      scrollElement.scrollLeft + scrollElement.clientWidth - FOLLOW_PADDING

    if (cursorLeft < minVisible || cursorLeft > maxVisible) {
      scrollElement.scrollLeft = clampValue(
        cursorLeft - scrollElement.clientWidth / 2,
        0,
        Math.max(timelineWidth - scrollElement.clientWidth, 0),
      )
    }
  }, [currentTime, followPlayback, pixelsPerSecond, safeDuration, timelineWidth])

  useEffect(() => {
    function handlePointerMove(event: globalThis.PointerEvent): void {
      const dragState = dragStateRef.current
      if (!dragState) return

      const duration = dragState.originalEndTime - dragState.originalStartTime
      const deltaSeconds =
        (event.clientX - dragState.startClientX) / dragState.pixelsPerSecond
      const nextStartTime = clampValue(
        dragState.originalStartTime + deltaSeconds,
        0,
        Math.max(dragState.safeDuration - duration, 0),
      )
      const nextEndTime = clampValue(
        nextStartTime + duration,
        nextStartTime,
        dragState.safeDuration,
      )
      let nextCategory = dragState.originalCategory

      if (canvasRef.current) {
        const canvasRect = canvasRef.current.getBoundingClientRect()
        const laneTop = canvasRect.top + TIMELINE_RULER_HEIGHT
        const laneIndex = Math.floor(
          (event.clientY - laneTop) / TIMELINE_LANE_ROW_HEIGHT,
        )
        if (laneIndex >= 0 && laneIndex < laneDefinitions.length) {
          nextCategory = laneDefinitions[laneIndex].id
        }
      }

      onUpdateAnnotation(dragState.annotationId, {
        startTime: nextStartTime,
        endTime: nextEndTime,
        category: nextCategory,
      })
    }

    function handlePointerUp(): void {
      if (!dragStateRef.current) return
      dragStateRef.current = null
      setDraggingAnnotationId(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    window.addEventListener('pointercancel', handlePointerUp)

    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
      window.removeEventListener('pointercancel', handlePointerUp)
    }
  }, [laneDefinitions, onUpdateAnnotation])

  function handleTimelineBlockPointerDown(
    event: PointerEvent<HTMLButtonElement>,
    annotation: AnnotationItem,
  ): void {
    const endTime = annotation.endTime ?? annotation.startTime
    dragStateRef.current = {
      annotationId: annotation.id,
      startClientX: event.clientX,
      originalStartTime: annotation.startTime,
      originalEndTime: endTime,
      originalCategory: annotation.category,
      pixelsPerSecond,
      safeDuration,
    }
    setDraggingAnnotationId(annotation.id)
    onSelectAnnotation(annotation.id)
  }

  return (
    <section className="review-console">
      <div className="review-console-shell">
        <div className="review-top">
          <aside className="review-tool-rail" aria-label={copy.activeAnnotations}>
            <button
              type="button"
              className="review-tool-button"
              onClick={() => onCreateAnnotation(currentTime)}
            >
              <strong>{copy.mark}</strong>
              <span>{copy.createNote}</span>
            </button>
            <button
              type="button"
              className={followPlayback ? 'review-tool-button is-active' : 'review-tool-button'}
              aria-pressed={followPlayback}
              onClick={() => setFollowPlayback((currentValue) => !currentValue)}
            >
              <strong>{copy.track}</strong>
              <span>{copy.followCursor}</span>
            </button>
            <button
              type="button"
              className="review-tool-button"
              onClick={() => setPixelsPerSecond(BASE_PIXELS_PER_SECOND)}
            >
              <strong>{copy.fit}</strong>
              <span>{copy.resetZoom}</span>
            </button>
          </aside>

          <div className="review-stage">
            <div className="review-stage-viewport">
              {videoSource ? (
                <video
                  ref={videoRef}
                  className="review-stage-player"
                  preload="metadata"
                  playsInline
                  controls
                  src={videoSource}
                />
              ) : (
                <div className="review-stage-empty">{copy.noPlayableVideo}</div>
              )}

              <div className="review-stage-overlay review-stage-overlay-top">
                <div className="review-timecode-badge">
                  {formatTimecode(videoCurrentTime, fps)}
                </div>
                <div className="review-stage-badges">
                  <span>{copy.liveReview}</span>
                  <span>{streamLabel || copy.unknownStream}</span>
                  <span>{Number.isFinite(fps) ? `${fps} FPS` : copy.videoFeed}</span>
                  <span>{isPaused ? 'Pause' : 'Play'}</span>
                </div>
              </div>
            </div>

            <div className="review-stage-footer">
              <div className="review-stage-meta">
                <span>{copy.stream}</span>
                <strong>{streamLabel || copy.unknown}</strong>
              </div>
              <div className="review-stage-meta">
                <span>{copy.file}</span>
                <strong>{videoTitle || copy.unknown}</strong>
              </div>
              <div className="review-stage-meta">
                <span>{copy.chunk}</span>
                <strong>{chunkLabel || copy.unknown}</strong>
              </div>
              <div className="review-stage-meta">
                <span>{copy.frame}</span>
                <strong>{Number.isFinite(currentFrame) ? currentFrame : copy.unknown}</strong>
              </div>
            </div>
          </div>

          <aside className="review-sidebar">
            <header className="review-sidebar-head">
              <div>
                <h3>{copy.activeAnnotations}</h3>
                <p>{copy.activeAnnotationsDesc}</p>
              </div>
              <button
                type="button"
                className="annotation-secondary-button"
                onClick={() => onCreateAnnotation(currentTime)}
              >
                {copy.createNew}
              </button>
            </header>

            <div className="review-sidebar-list">
              {sortedAnnotations.length ? (
                sortedAnnotations.map((annotation) => {
                  const laneDefinition = getLaneDefinition(
                    laneDefinitions,
                    annotation.category,
                  )
                  const isSelected = annotation.id === selectedAnnotationId
                  const annotationEnd = annotation.endTime ?? annotation.startTime
                  const tags = Array.isArray(annotation.tags) ? annotation.tags : []

                  return (
                    <article
                      key={annotation.id}
                      className={isSelected ? 'annotation-rail-card is-selected' : 'annotation-rail-card'}
                      style={{ '--annotation-accent': laneDefinition.color } as CSSProperties}
                    >
                      <button
                        type="button"
                        className="annotation-rail-main"
                        onClick={() => onSelectAnnotation(annotation.id)}
                      >
                        <div className="annotation-rail-head">
                          <span className="annotation-rail-type">{laneDefinition.cardLabel}</span>
                          <span className="annotation-rail-time">
                            {formatTimecode(annotation.startTime, fps)}
                          </span>
                        </div>
                        <strong className="annotation-rail-title">{annotation.label}</strong>
                        <p className="annotation-rail-copy">{trimText(annotation.text)}</p>
                        {tags.length ? (
                          <div className="annotation-rail-tags">
                            {tags.map((tag) =>
                              isSelected ? (
                                <button
                                  key={`${annotation.id}-${tag}`}
                                  type="button"
                                  className="annotation-rail-tag is-removable"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    onUpdateAnnotation(annotation.id, {
                                      tags: tags.filter((item) => item !== tag),
                                    })
                                  }}
                                >
                                  <span>{tag}</span>
                                  <span
                                    className="annotation-rail-tag-remove"
                                    aria-hidden="true"
                                  >
                                    ×
                                  </span>
                                </button>
                              ) : (
                                <span
                                  key={`${annotation.id}-${tag}`}
                                  className="annotation-rail-tag"
                                >
                                  {tag}
                                </span>
                              ),
                            )}
                          </div>
                        ) : null}
                      </button>

                      {isSelected ? (
                        <div className="annotation-rail-editor">
                          <div className="annotation-rail-edit-grid">
                            <label className="annotation-edit-field">
                              <span>{copy.label}</span>
                              <input
                                type="text"
                                value={annotation.label}
                                onChange={(event) =>
                                  onUpdateAnnotation(annotation.id, {
                                    label: event.target.value,
                                  })
                                }
                              />
                            </label>

                            {laneDefinitions.length > 1 ? (
                              <label className="annotation-edit-field">
                                <span>{copy.lane}</span>
                                <select
                                  value={annotation.category}
                                  onChange={(event) =>
                                    onUpdateAnnotation(annotation.id, {
                                      category: event.target.value,
                                    })
                                  }
                                >
                                  {laneDefinitions.map((definition) => (
                                    <option key={definition.id} value={definition.id}>
                                      {definition.laneLabel}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            ) : null}

                            <label className="annotation-edit-field">
                              <span>{copy.start}</span>
                              <TimeInput
                                value={annotation.startTime}
                                min={0}
                                max={safeDuration}
                                onCommit={(nextValue) =>
                                  onUpdateAnnotation(annotation.id, {
                                    startTime: nextValue,
                                  })
                                }
                              />
                            </label>

                            <label className="annotation-edit-field">
                              <span>{copy.end}</span>
                              <TimeInput
                                value={annotationEnd}
                                min={annotation.startTime}
                                max={safeDuration}
                                onCommit={(nextValue) =>
                                  onUpdateAnnotation(annotation.id, {
                                    endTime: nextValue,
                                  })
                                }
                              />
                            </label>
                          </div>

                          <label className="annotation-edit-textarea">
                            <span>{copy.description}</span>
                            <textarea
                              rows={4}
                              value={annotation.text}
                              onChange={(event) =>
                                onUpdateAnnotation(annotation.id, {
                                  text: event.target.value,
                                })
                              }
                            />
                          </label>

                          <div className="annotation-rail-actions">
                            <button
                              type="button"
                              className="annotation-secondary-button"
                              onClick={() => onJumpToTime(annotation.startTime)}
                            >
                              {copy.jumpTo}
                            </button>
                            <button
                              type="button"
                              className="annotation-secondary-button"
                              onClick={() =>
                                onUpdateAnnotation(annotation.id, {
                                  startTime: currentTime,
                                  endTime: clampValue(
                                    currentTime + 1.25,
                                    currentTime,
                                    safeDuration,
                                  ),
                                })
                              }
                            >
                              {copy.useCursor}
                            </button>
                            {annotation.source === 'user' ? (
                              <button
                                type="button"
                                className="annotation-secondary-button"
                                onClick={() => onDeleteAnnotation(annotation.id)}
                              >
                                {copy.delete}
                              </button>
                            ) : null}
                          </div>
                        </div>
                      ) : null}
                    </article>
                  )
                })
              ) : (
                <div className="review-sidebar-empty">{copy.noAnnotations}</div>
              )}
            </div>
          </aside>
        </div>

        <div className="timeline-dock">
          <div className="timeline-toolbar">
            <div className="timeline-toolbar-left">
              <span className="timeline-toolbar-title">{copy.mainTimeline}</span>
            </div>

            <div className="timeline-toolbar-center">
              <span>
                {copy.video} {formatTimecode(videoCurrentTime, fps)}
              </span>
            </div>

            <div className="timeline-toolbar-right">
              <button
                type="button"
                className="annotation-secondary-button"
                onClick={() =>
                  setPixelsPerSecond((currentValue) =>
                    clampValue(
                      Math.round(currentValue / 1.28),
                      MIN_PIXELS_PER_SECOND,
                      MAX_PIXELS_PER_SECOND,
                    ),
                  )
                }
              >
                -
              </button>
              <input
                className="timeline-zoom-slider"
                type="range"
                min={MIN_PIXELS_PER_SECOND}
                max={MAX_PIXELS_PER_SECOND}
                value={pixelsPerSecond}
                onChange={(event) => setPixelsPerSecond(Number(event.target.value))}
              />
              <button
                type="button"
                className="annotation-secondary-button"
                onClick={() =>
                  setPixelsPerSecond((currentValue) =>
                    clampValue(
                      Math.round(currentValue * 1.28),
                      MIN_PIXELS_PER_SECOND,
                      MAX_PIXELS_PER_SECOND,
                    ),
                  )
                }
              >
                +
              </button>
            </div>
          </div>

          <div
            className="timeline-frame"
            style={{ minHeight: `${timelineCanvasHeight}px` }}
          >
            <div
              className="timeline-labels-column"
              style={{
                gridTemplateRows: `${TIMELINE_RULER_HEIGHT}px repeat(${lanes.length}, ${TIMELINE_LANE_ROW_HEIGHT}px)`,
              }}
            >
              <div className="timeline-label-spacer" />
              {lanes.map((lane) => (
                <div
                  key={lane.id}
                  className="timeline-lane-label"
                  style={{ minHeight: `${TIMELINE_LANE_ROW_HEIGHT}px` }}
                >
                  {lane.laneLabel}
                </div>
              ))}
            </div>

            <div ref={scrollRef} className="timeline-scroll">
              <div
                ref={canvasRef}
                className="timeline-canvas"
                style={{
                  width: `${timelineWidth}px`,
                  minHeight: `${timelineCanvasHeight}px`,
                }}
              >
                <div
                  className="timeline-ruler-strip"
                  style={{ height: `${TIMELINE_RULER_HEIGHT}px` }}
                >
                  {ticks.map((tick) => {
                    const left = TIMELINE_PADDING + tick.time * pixelsPerSecond
                    return (
                      <div
                        key={`ruler-${tick.time}`}
                        className={tick.isMajor ? 'timeline-ruler-tick is-major' : 'timeline-ruler-tick'}
                        style={{ left: `${left}px` }}
                      >
                        {tick.isMajor ? <span>{formatClock(tick.time)}</span> : null}
                      </div>
                    )
                  })}
                </div>

                <div
                  className="timeline-playhead"
                  style={{
                    left: `${TIMELINE_PADDING + clampValue(currentTime, 0, safeDuration) * pixelsPerSecond}px`,
                  }}
                >
                  <div className="timeline-playhead-head" />
                </div>

                <div className="timeline-lanes">
                  {lanes.map((lane) => (
                    <div
                      key={lane.id}
                      className="timeline-lane-row"
                      style={{ height: `${TIMELINE_LANE_ROW_HEIGHT}px` }}
                    >
                      {ticks
                        .filter((tick) => tick.isMajor)
                        .map((tick) => {
                          const left = TIMELINE_PADDING + tick.time * pixelsPerSecond
                          return (
                            <div
                              key={`${lane.id}-${tick.time}`}
                              className="timeline-gridline"
                              style={{ left: `${left}px` }}
                            />
                          )
                        })}

                      {lane.annotations.map((annotation) => {
                        const laneDefinition = getLaneDefinition(
                          laneDefinitions,
                          annotation.category,
                        )
                        const startTime = clampValue(annotation.startTime, 0, safeDuration)
                        const endTime = clampValue(
                          annotation.endTime ?? Math.min(annotation.startTime + 0.25, safeDuration),
                          startTime,
                          safeDuration,
                        )
                        const left = TIMELINE_PADDING + startTime * pixelsPerSecond
                        const width = Math.max((endTime - startTime) * pixelsPerSecond, 18)
                        const isSelected = annotation.id === selectedAnnotationId

                        return (
                          <button
                            key={annotation.id}
                            type="button"
                            className={
                              draggingAnnotationId === annotation.id
                                ? 'timeline-block is-selected is-dragging'
                                : isSelected
                                  ? 'timeline-block is-selected'
                                  : 'timeline-block'
                            }
                            style={
                              {
                                left: `${left}px`,
                                width: `${width}px`,
                                '--timeline-block-color': laneDefinition.color,
                              } as CSSProperties
                            }
                            onPointerDown={(event) =>
                              handleTimelineBlockPointerDown(event, annotation)
                            }
                            onClick={() => {
                              onSelectAnnotation(annotation.id)
                              onJumpToTime(annotation.startTime)
                            }}
                          >
                            <span>{annotation.label}</span>
                          </button>
                        )
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
