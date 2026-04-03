import { useEffect, useMemo, useRef, useState } from 'react'
import { useI18n } from '../../controllers/i18n'
import {
  useWorkflow,
  type AnnotationItem,
  type AnnotationWorkspacePayload,
  type WorkflowTaskContext,
} from '../../controllers/workflow'
import AnnotationWorkspaceCard from './AnnotationWorkspaceCard'

const ANNOTATION_SEED_COLORS = [
  '#44d7ff',
  '#ff8a5b',
  '#b7ff5c',
  '#ffd84d',
  '#ff6ba8',
  '#8c9bff',
]
const CLIP_TIME_EPSILON = 0.05

interface ComparisonEntry {
  key: string
  label: string
  actionValues: Array<number | null>
  stateValues: Array<number | null>
  xValues: number[]
}

interface SavedComparisonContext {
  jointName: string
  timeS: number | null
  frameIndex: number | null
  actionValue: number | null
  stateValue: number | null
  source: string
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function clampAnnotationTime(value: number, maxValue: number): number {
  return Math.min(Math.max(value, 0), Math.max(maxValue, 0))
}

function getClipStart(videoItem: AnnotationWorkspacePayload['videos'][number] | null): number {
  return typeof videoItem?.from_timestamp === 'number' ? videoItem.from_timestamp : 0
}

function getClipEnd(
  videoItem: AnnotationWorkspacePayload['videos'][number] | null,
): number | null {
  return typeof videoItem?.to_timestamp === 'number' ? videoItem.to_timestamp : null
}

function clampToClipWindow(
  videoItem: AnnotationWorkspacePayload['videos'][number] | null,
  absoluteTime: number,
  duration = Number.POSITIVE_INFINITY,
): number {
  const clipStart = getClipStart(videoItem)
  const clipEnd = getClipEnd(videoItem)
  let nextTime = Number.isFinite(absoluteTime) ? absoluteTime : clipStart

  nextTime = Math.max(nextTime, clipStart)
  if (isFiniteNumber(clipEnd)) {
    nextTime = Math.min(nextTime, clipEnd)
  }
  if (Number.isFinite(duration)) {
    nextTime = Math.min(nextTime, duration)
  }

  return nextTime
}

function getRelativePlaybackTime(
  videoItem: AnnotationWorkspacePayload['videos'][number] | null,
  absoluteTime: number,
): number {
  return Math.max(absoluteTime - getClipStart(videoItem), 0)
}

function findClosestPlaybackIndex(timeValues: number[], currentTime: number): number {
  if (!timeValues.length) return 0

  let closestIndex = 0
  let smallestDiff = Number.POSITIVE_INFINITY

  timeValues.forEach((timeValue, index) => {
    const diff = Math.abs(timeValue - currentTime)
    if (diff < smallestDiff) {
      smallestDiff = diff
      closestIndex = index
    }
  })

  return closestIndex
}

function buildDefaultAnnotationText(summary: AnnotationWorkspacePayload['summary'] | null): string {
  if (!summary) return 'Add an annotation for the current task.'
  if (summary.task_label) return summary.task_label
  if (summary.task_value) return summary.task_value
  return `Episode: ${summary.record_key}`
}

function deriveAnnotationLabel(text: string, fallback: string): string {
  const firstLine = String(text || '')
    .split('\n')
    .map((line) => line.trim())
    .find(Boolean)

  if (!firstLine) return fallback
  return firstLine.slice(0, 48)
}

function normalizeAnnotation(
  annotation: Partial<AnnotationItem> | null | undefined,
  fallbackKey = 'episode',
): AnnotationItem | null {
  if (!annotation || typeof annotation !== 'object') return null

  return {
    id:
      annotation.id ??
      `${fallbackKey}-annotation-${Math.random().toString(36).slice(2, 8)}`,
    label:
      annotation.label ||
      deriveAnnotationLabel(annotation.text || '', 'Annotation'),
    category: annotation.category || 'movement',
    color: annotation.color || ANNOTATION_SEED_COLORS[0],
    startTime: Number(annotation.startTime ?? 0),
    endTime:
      annotation.endTime === null || annotation.endTime === undefined
        ? null
        : Number(annotation.endTime),
    text: String(annotation.text || ''),
    tags: Array.isArray(annotation.tags) ? annotation.tags : [],
    source: annotation.source || 'user',
  }
}

function formatSeconds(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : '0.00'
}

function formatValue(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(3) : '-'
}

function buildComparisonSelectionKey(entry: ComparisonEntry): string {
  return `${entry.label}|${entry.key}`
}

function matchComparisonSelectionKey(
  entries: ComparisonEntry[],
  jointName: string,
): string {
  if (!jointName) return ''
  const normalizedJoint = String(jointName).trim().toLowerCase()
  const matchedEntry = entries.find(
    (entry) => entry.label.toLowerCase() === normalizedJoint,
  )
  return matchedEntry ? buildComparisonSelectionKey(matchedEntry) : ''
}

function normalizeSavedComparisonContext(
  taskContext: WorkflowTaskContext | null | undefined,
): SavedComparisonContext | null {
  if (!taskContext || typeof taskContext !== 'object') return null

  const timeValue = Number(taskContext.time_s)
  return {
    jointName: String(taskContext.joint_name || '').trim(),
    timeS: Number.isFinite(timeValue) ? Math.max(timeValue, 0) : null,
    frameIndex: Number.isFinite(Number(taskContext.frame_index))
      ? Number(taskContext.frame_index)
      : null,
    actionValue: Number.isFinite(Number(taskContext.action_value))
      ? Number(taskContext.action_value)
      : null,
    stateValue: Number.isFinite(Number(taskContext.state_value))
      ? Number(taskContext.state_value)
      : null,
    source: String(taskContext.source || '').trim(),
  }
}

function buildJointComparisonEntries(
  jointTrajectory: AnnotationWorkspacePayload['joint_trajectory'] | null,
): ComparisonEntry[] {
  const timeValues = jointTrajectory?.time_values || []
  const baseTime = isFiniteNumber(timeValues[0]) ? timeValues[0] : 0
  const relativeTimes = timeValues.map((timeValue) =>
    isFiniteNumber(timeValue) ? Math.max(timeValue - baseTime, 0) : 0,
  )

  return (jointTrajectory?.joint_trajectories || [])
    .map((item, index) => ({
      key: `${item.joint_name || item.state_name || item.action_name || 'joint'}-${index}`,
      label: item.joint_name || item.state_name || item.action_name || 'Joint',
      actionValues: item.action_values || [],
      stateValues: item.state_values || [],
      xValues: relativeTimes,
    }))
    .filter((item) => {
      const hasAction = item.actionValues.some(
        (value) => value !== null && value !== undefined,
      )
      const hasState = item.stateValues.some(
        (value) => value !== null && value !== undefined,
      )
      return item.xValues.length && (hasAction || hasState)
    })
}

function buildLinePath(
  xValues: number[],
  series: Array<number | null>,
  minY: number,
  maxY: number,
  width: number,
  height: number,
  padding: number,
): string {
  const maxX = xValues[xValues.length - 1] || 1
  const usableWidth = width - padding * 2
  const usableHeight = height - padding * 2
  const rangeY = maxY - minY || 1
  let path = ''

  xValues.forEach((xValue, index) => {
    const yValue = series[index]
    if (!Number.isFinite(yValue)) return
    const x = padding + (xValue / maxX) * usableWidth
    const y = padding + usableHeight - ((Number(yValue) - minY) / rangeY) * usableHeight
    path += `${path ? ' L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`
  })

  return path
}

function JointComparisonChart({
  entry,
  currentTime,
  emptyLabel,
}: {
  entry: ComparisonEntry
  currentTime: number
  emptyLabel: string
}) {
  const width = 260
  const height = 132
  const padding = 12
  const numericValues = [...entry.actionValues, ...entry.stateValues].filter(
    (value): value is number => Number.isFinite(value),
  )

  if (!numericValues.length) {
    return <div className="episode-preview-empty">{emptyLabel}</div>
  }

  let minY = Math.min(...numericValues)
  let maxY = Math.max(...numericValues)
  if (Math.abs(maxY - minY) < 1e-6) {
    minY -= 1
    maxY += 1
  }

  const maxX = entry.xValues[entry.xValues.length - 1] || 1
  const cursorX =
    padding +
    (Math.min(Math.max(currentTime, 0), maxX) / maxX) * (width - padding * 2)

  const actionPath = buildLinePath(
    entry.xValues,
    entry.actionValues,
    minY,
    maxY,
    width,
    height,
    padding,
  )
  const statePath = buildLinePath(
    entry.xValues,
    entry.stateValues,
    minY,
    maxY,
    width,
    height,
    padding,
  )

  return (
    <svg
      className="joint-comparison-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${entry.label} trajectory`}
    >
      <rect x="0" y="0" width={width} height={height} rx="14" fill="rgba(255, 255, 255, 0.82)" />
      {[0, 0.5, 1].map((ratio) => {
        const y = padding + ratio * (height - padding * 2)
        return (
          <line
            key={ratio}
            x1={padding}
            x2={width - padding}
            y1={y}
            y2={y}
            stroke="rgba(47, 111, 228, 0.12)"
            strokeWidth="1"
          />
        )
      })}
      <line
        x1={cursorX}
        x2={cursorX}
        y1={padding}
        y2={height - padding}
        stroke="rgba(17, 17, 17, 0.35)"
        strokeDasharray="4 4"
        strokeWidth="1.4"
      />
      {actionPath ? (
        <path
          d={actionPath}
          fill="none"
          stroke="#2f6fe4"
          strokeWidth="2.25"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : null}
      {statePath ? (
        <path
          d={statePath}
          fill="none"
          stroke="#ff8a5b"
          strokeWidth="2.25"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : null}
    </svg>
  )
}

function JointComparisonGrid({
  jointTrajectory,
  currentTime,
  copy,
  activeKey,
  onSelectEntry,
}: {
  jointTrajectory: AnnotationWorkspacePayload['joint_trajectory'] | null
  currentTime: number
  copy: {
    noJointData: string
    actionSeries: string
    stateSeries: string
  }
  activeKey: string
  onSelectEntry: (key: string) => void
}) {
  const entries = useMemo(
    () => buildJointComparisonEntries(jointTrajectory),
    [jointTrajectory],
  )

  if (!entries.length) {
    return <div className="episode-preview-empty">{copy.noJointData}</div>
  }

  return (
    <div className="joint-comparison-grid">
      {entries.map((entry) => {
        const selectionKey = buildComparisonSelectionKey(entry)
        const isSelected = selectionKey === activeKey
        return (
          <article
            key={entry.key}
            className={isSelected ? 'joint-comparison-card is-selected' : 'joint-comparison-card'}
            role="button"
            tabIndex={0}
            onClick={() => onSelectEntry(selectionKey)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelectEntry(selectionKey)
              }
            }}
          >
            <div className="joint-comparison-head">
              <strong>{entry.label}</strong>
              <div className="joint-comparison-legend">
                <span className="is-action">{copy.actionSeries}</span>
                <span className="is-state">{copy.stateSeries}</span>
              </div>
            </div>
            <JointComparisonChart
              entry={entry}
              currentTime={currentTime}
              emptyLabel={copy.noJointData}
            />
          </article>
        )
      })}
    </div>
  )
}

export default function AnnotationPanel() {
  const { locale } = useI18n()
  const {
    prototypeResults,
    workflowState,
    propagationResults,
    fetchAnnotationWorkspace,
    saveAnnotations,
    runPropagation,
    loadPropagationResults,
  } = useWorkflow()

  const copy = locale === 'zh'
    ? {
        selectAnchor: '选择一个 anchor episode 开始标注。',
        runPrototypeFirst: '先完成原型发现，系统会为每个聚类给出 anchor episode。',
        anchors: 'Anchor Episodes',
        anchorsDesc: '每个聚类选一个代表性 episode 作为人工标注入口。',
        cluster: '聚类',
        members: '成员数',
        quality: '质量',
        annotated: '已标注',
        propagationDone: '已传播',
        propagationPending: '未传播',
        loadingWorkspace: '正在加载标注工作台...',
        saveAnnotationVersion: '保存标注',
        saving: '保存中...',
        runPropagation: '运行传播',
        saveAndPropagate: '先保存再传播',
        propagating: '传播中...',
        streamLabel: '视频流',
        syncedAxes: 'Action / State 关节对比',
        syncedAxesHint: '保留每个关节的 Action / State 对比，并跟随当前视频时间同步游标。',
        currentCursor: '当前游标',
        focusJoint: '当前对比关节',
        focusActionValue: 'Action 值',
        focusStateValue: 'State 值',
        focusFrame: '帧索引',
        restoreSource: '恢复来源',
        noJointData: '当前 episode 没有可展示的 Action / State 关节对比。',
        actionSeries: 'Action',
        stateSeries: 'State',
        unknownJoint: '未知关节',
        workspaceStatus: '工作台状态',
        savedVersion: '保存版本',
        savedAt: '保存时间',
        notSavedYet: '尚未保存',
        annotationCount: '标注数量',
        noVideoData: '当前 episode 没有可用于标注的视频。',
        saveBeforeSwitch: '切换 anchor 前会自动保存当前修改。',
        targetCount: '传播目标',
        switchVideo: '切换视频流',
      }
    : {
        selectAnchor: 'Select an anchor episode to start annotating.',
        runPrototypeFirst: 'Run prototype discovery first so the system can generate one anchor episode per cluster.',
        anchors: 'Anchor Episodes',
        anchorsDesc: 'Each cluster exposes one representative episode as the manual-annotation entrypoint.',
        cluster: 'Cluster',
        members: 'Members',
        quality: 'Quality',
        annotated: 'Annotated',
        propagationDone: 'Propagated',
        propagationPending: 'Not propagated',
        loadingWorkspace: 'Loading annotation workspace...',
        saveAnnotationVersion: 'Save Annotations',
        saving: 'Saving...',
        runPropagation: 'Run Propagation',
        saveAndPropagate: 'Save & Propagate',
        propagating: 'Propagating...',
        streamLabel: 'Stream',
        syncedAxes: 'Action / State Joint Comparison',
        syncedAxesHint: 'Keep per-joint Action / State comparison and sync the cursor with the current video time.',
        currentCursor: 'Cursor',
        focusJoint: 'Focused Joint',
        focusActionValue: 'Action Value',
        focusStateValue: 'State Value',
        focusFrame: 'Frame Index',
        restoreSource: 'Restore Source',
        noJointData: 'No Action / State joint comparison is available for this episode.',
        actionSeries: 'Action',
        stateSeries: 'State',
        unknownJoint: 'Unknown Joint',
        workspaceStatus: 'Workspace Status',
        savedVersion: 'Saved Version',
        savedAt: 'Saved At',
        notSavedYet: 'Not saved yet',
        annotationCount: 'Annotations',
        noVideoData: 'No video stream is available for this episode.',
        saveBeforeSwitch: 'The current draft will be auto-saved before switching anchors.',
        targetCount: 'Targets',
        switchVideo: 'Switch Stream',
      }

  const anchorItems = useMemo(() => {
    const annotatedSet = new Set(workflowState?.stages.annotation.annotated_episodes || [])
    return (prototypeResults?.clusters || [])
      .map((cluster) => {
        const episodeIndex = Number(cluster.anchor_record_key)
        if (!Number.isFinite(episodeIndex)) return null
        const anchorMember =
          cluster.members.find((member) => member.record_key === cluster.anchor_record_key) ||
          cluster.members[0]
        return {
          episodeIndex,
          clusterIndex: cluster.cluster_index,
          memberCount: cluster.member_count,
          qualityScore: anchorMember?.quality?.score ?? null,
          qualityPassed: anchorMember?.quality?.passed ?? null,
          annotated: annotatedSet.has(episodeIndex),
          propagated: propagationResults?.source_episode_index === episodeIndex,
        }
      })
      .filter((item): item is NonNullable<typeof item> => item !== null)
  }, [prototypeResults, propagationResults, workflowState])

  const [selectedAnchorEpisode, setSelectedAnchorEpisode] = useState<number | null>(null)
  const [workspace, setWorkspace] = useState<AnnotationWorkspacePayload | null>(null)
  const [workspaceLoading, setWorkspaceLoading] = useState(false)
  const [workspaceError, setWorkspaceError] = useState('')
  const [annotations, setAnnotations] = useState<AnnotationItem[]>([])
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [selectedVideoPath, setSelectedVideoPath] = useState('')
  const [playbackState, setPlaybackState] = useState({ index: 0, time: 0 })
  const [isStudioPaused, setIsStudioPaused] = useState(true)
  const [selectedComparisonKey, setSelectedComparisonKey] = useState('')
  const [pendingRestoreContext, setPendingRestoreContext] = useState<SavedComparisonContext | null>(null)
  const [savedTaskContext, setSavedTaskContext] = useState<WorkflowTaskContext | null>(null)
  const [saveState, setSaveState] = useState({
    isSaving: false,
    error: '',
    versionNumber: 0,
    savedAt: '',
  })
  const [propagationState, setPropagationState] = useState({
    isRunning: false,
    error: '',
  })

  const annotationIdRef = useRef(0)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const effectiveSelectedVideo = useMemo(() => {
    if (!workspace?.videos.length) return null
    return (
      workspace.videos.find((video) => video.path === selectedVideoPath) ||
      workspace.videos[0]
    )
  }, [selectedVideoPath, workspace])

  const comparisonEntries = useMemo(
    () => buildJointComparisonEntries(workspace?.joint_trajectory || null),
    [workspace],
  )
  const activeComparisonEntry =
    comparisonEntries.find(
      (entry) => buildComparisonSelectionKey(entry) === selectedComparisonKey,
    ) ||
    comparisonEntries[0] ||
    null
  const frameValues = workspace?.joint_trajectory.frame_values || []
  const currentFrame = frameValues[playbackState.index] ?? null
  const timelineDuration = useMemo(() => {
    const clipStart = getClipStart(effectiveSelectedVideo)
    const clipEnd = getClipEnd(effectiveSelectedVideo)
    if (isFiniteNumber(clipEnd)) {
      return Math.max(clipEnd - clipStart, 0)
    }
    if (workspace?.summary.duration_s) {
      return workspace.summary.duration_s
    }
    const timeValues = workspace?.joint_trajectory.time_values || []
    if (timeValues.length > 1) {
      return Math.max(timeValues[timeValues.length - 1] - timeValues[0], 0)
    }
    return 0
  }, [effectiveSelectedVideo, workspace])

  const comparisonSnapshot = useMemo(() => {
    if (!activeComparisonEntry) {
      return {
        joint_name: '',
        time_s: Number(playbackState.time.toFixed(3)),
        frame_index: currentFrame,
        action_value: null,
        state_value: null,
        source: 'annotation_workspace',
      }
    }

    return {
      joint_name: activeComparisonEntry.label,
      time_s: Number(playbackState.time.toFixed(3)),
      frame_index: currentFrame,
      action_value: activeComparisonEntry.actionValues[playbackState.index] ?? null,
      state_value: activeComparisonEntry.stateValues[playbackState.index] ?? null,
      source: 'annotation_workspace',
    }
  }, [activeComparisonEntry, currentFrame, playbackState.index, playbackState.time])

  const taskContext = useMemo<WorkflowTaskContext>(() => {
    const defaultText = buildDefaultAnnotationText(workspace?.summary || null)
    return {
      label:
        workspace?.summary.task_label ||
        workspace?.summary.task_value ||
        'Task',
      text: defaultText,
      ...comparisonSnapshot,
    }
  }, [comparisonSnapshot, workspace])

  const latestPropagation =
    propagationResults?.source_episode_index === selectedAnchorEpisode
      ? propagationResults
      : workspace?.latest_propagation || null

  useEffect(() => {
    if (!anchorItems.length) {
      setSelectedAnchorEpisode(null)
      setWorkspace(null)
      return
    }
    setSelectedAnchorEpisode((currentValue) => {
      if (
        currentValue !== null &&
        anchorItems.some((item) => item.episodeIndex === currentValue)
      ) {
        return currentValue
      }
      return anchorItems[0].episodeIndex
    })
  }, [anchorItems])

  useEffect(() => {
    if (!workspace?.videos.length) {
      setSelectedVideoPath('')
      return
    }
    setSelectedVideoPath((currentPath) => {
      if (workspace.videos.some((video) => video.path === currentPath)) {
        return currentPath
      }
      return workspace.videos[0].path
    })
  }, [workspace])

  useEffect(() => {
    if (!selectedAnchorEpisode) return

    let active = true
    setWorkspaceLoading(true)
    setWorkspaceError('')

    void fetchAnnotationWorkspace(selectedAnchorEpisode)
      .then((payload) => {
        if (!active) return
        setWorkspace(payload)
        const savedAnnotations = payload.annotations.annotations || []
        annotationIdRef.current = savedAnnotations.length
        const normalizedAnnotations = savedAnnotations
          .map((item) => normalizeAnnotation(item, String(selectedAnchorEpisode)))
          .filter((item): item is AnnotationItem => item !== null)
        setAnnotations(normalizedAnnotations)
        setSelectedAnnotationId(normalizedAnnotations[0]?.id ?? null)
        setHasUnsavedChanges(false)
        setSavedTaskContext(payload.annotations.task_context || {})
        setPendingRestoreContext(
          normalizeSavedComparisonContext(payload.annotations.task_context),
        )
        setSaveState({
          isSaving: false,
          error: '',
          versionNumber: payload.annotations.version_number || 0,
          savedAt:
            payload.annotations.updated_at ||
            payload.annotations.created_at ||
            '',
        })
      })
      .catch((error: Error) => {
        if (!active) return
        setWorkspace(null)
        setAnnotations([])
        setSelectedAnnotationId(null)
        setWorkspaceError(error.message)
      })
      .finally(() => {
        if (!active) return
        setWorkspaceLoading(false)
      })

    return () => {
      active = false
    }
  }, [fetchAnnotationWorkspace, selectedAnchorEpisode])

  useEffect(() => {
    setPlaybackState({ index: 0, time: 0 })
    setSelectedComparisonKey('')
    setIsStudioPaused(true)
  }, [selectedAnchorEpisode, selectedVideoPath])

  useEffect(() => {
    if (!comparisonEntries.length) {
      setSelectedComparisonKey('')
      return
    }

    setSelectedComparisonKey((currentValue) => {
      if (
        currentValue &&
        comparisonEntries.some(
          (entry) => buildComparisonSelectionKey(entry) === currentValue,
        )
      ) {
        return currentValue
      }
      const restoredKey = matchComparisonSelectionKey(
        comparisonEntries,
        pendingRestoreContext?.jointName || '',
      )
      return restoredKey || buildComparisonSelectionKey(comparisonEntries[0])
    })
  }, [comparisonEntries, pendingRestoreContext])

  useEffect(() => {
    if (!annotations.length) {
      setSelectedAnnotationId(null)
      return
    }

    if (annotations.some((annotation) => annotation.id === selectedAnnotationId)) {
      return
    }

    setSelectedAnnotationId(annotations[0].id)
  }, [annotations, selectedAnnotationId])

  useEffect(() => {
    if (!pendingRestoreContext || !effectiveSelectedVideo) return
    const playerEl = videoRef.current
    const restoreContext = pendingRestoreContext
    if (!playerEl) return
    const player = playerEl

    function applyRestore(): void {
      const relativeTime = restoreContext.timeS
      if (Number.isFinite(relativeTime)) {
        const absoluteTime = getClipStart(effectiveSelectedVideo) + Number(relativeTime)
        const boundedTime = clampToClipWindow(
          effectiveSelectedVideo,
          absoluteTime,
          Number.isFinite(player.duration) ? player.duration : Number.POSITIVE_INFINITY,
        )
        player.currentTime = boundedTime
        const timeValues = workspace?.joint_trajectory.time_values || []
        const nextIndex = timeValues.length
          ? findClosestPlaybackIndex(
              timeValues,
              Number(relativeTime) + (timeValues[0] || 0),
            )
          : 0
        setPlaybackState({
          index: nextIndex,
          time: getRelativePlaybackTime(effectiveSelectedVideo, boundedTime),
        })
      }
      setPendingRestoreContext(null)
    }

    if (player.readyState >= 1) {
      applyRestore()
      return
    }

    player.addEventListener('loadedmetadata', applyRestore, { once: true })
    return () => {
      player.removeEventListener('loadedmetadata', applyRestore)
    }
  }, [effectiveSelectedVideo, pendingRestoreContext, workspace])

  useEffect(() => {
    const playerEl = videoRef.current
    if (!playerEl || !effectiveSelectedVideo) return
    const player = playerEl

    let rafId = 0
    const timeValues = workspace?.joint_trajectory.time_values || []

    function stopPolling(): void {
      if (!rafId) return
      window.cancelAnimationFrame(rafId)
      rafId = 0
    }

    function handlePlaybackTimeChange(currentTime: number): void {
      const nextIndex = timeValues.length
        ? findClosestPlaybackIndex(timeValues, currentTime + (timeValues[0] || 0))
        : 0
      setPlaybackState({ index: nextIndex, time: currentTime })
    }

    function poll(): void {
      const boundedTime = clampToClipWindow(
        effectiveSelectedVideo,
        player.currentTime,
        player.duration,
      )
      if (Math.abs(player.currentTime - boundedTime) > CLIP_TIME_EPSILON) {
        player.currentTime = boundedTime
      }

      const clipEnd = getClipEnd(effectiveSelectedVideo)
      if (isFiniteNumber(clipEnd) && boundedTime >= clipEnd - CLIP_TIME_EPSILON) {
        if (!player.paused) player.pause()
        handlePlaybackTimeChange(
          getRelativePlaybackTime(effectiveSelectedVideo, boundedTime),
        )
        stopPolling()
        return
      }

      handlePlaybackTimeChange(
        getRelativePlaybackTime(effectiveSelectedVideo, boundedTime),
      )
      if (!player.paused && !player.ended) {
        rafId = window.requestAnimationFrame(poll)
      } else {
        rafId = 0
      }
    }

    function startPolling(): void {
      stopPolling()
      rafId = window.requestAnimationFrame(poll)
    }

    function handleLoadedMetadata(): void {
      const clipStart = getClipStart(effectiveSelectedVideo)
      const nextTime = clampToClipWindow(
        effectiveSelectedVideo,
        clipStart,
        player.duration,
      )
      if (Math.abs(player.currentTime - nextTime) > 0.1) {
        player.currentTime = nextTime
      }
      setIsStudioPaused(player.paused)
      handlePlaybackTimeChange(
        getRelativePlaybackTime(effectiveSelectedVideo, player.currentTime),
      )
    }

    function handlePlay(): void {
      setIsStudioPaused(false)
      startPolling()
    }

    function handlePause(): void {
      setIsStudioPaused(true)
      handlePlaybackTimeChange(
        getRelativePlaybackTime(effectiveSelectedVideo, player.currentTime),
      )
      stopPolling()
    }

    function handleSeeking(): void {
      const nextTime = clampToClipWindow(
        effectiveSelectedVideo,
        player.currentTime,
        player.duration,
      )
      if (Math.abs(player.currentTime - nextTime) > CLIP_TIME_EPSILON) {
        player.currentTime = nextTime
      }
      handlePlaybackTimeChange(
        getRelativePlaybackTime(effectiveSelectedVideo, player.currentTime),
      )
    }

    player.addEventListener('loadedmetadata', handleLoadedMetadata)
    player.addEventListener('play', handlePlay)
    player.addEventListener('pause', handlePause)
    player.addEventListener('ended', handlePause)
    player.addEventListener('seeking', handleSeeking)

    if (player.readyState >= 1) {
      handleLoadedMetadata()
    }

    return () => {
      stopPolling()
      player.removeEventListener('loadedmetadata', handleLoadedMetadata)
      player.removeEventListener('play', handlePlay)
      player.removeEventListener('pause', handlePause)
      player.removeEventListener('ended', handlePause)
      player.removeEventListener('seeking', handleSeeking)
    }
  }, [effectiveSelectedVideo, workspace])

  useEffect(() => {
    const status = workflowState?.stages.annotation.status
    if (status !== 'running' && propagationState.isRunning) {
      setPropagationState((current) => ({ ...current, isRunning: false }))
    }
  }, [propagationState.isRunning, workflowState])

  function createAnnotation(seedTime = playbackState.time): void {
    if (!selectedAnchorEpisode) return

    annotationIdRef.current += 1
    const startTime = clampAnnotationTime(seedTime, Number.POSITIVE_INFINITY)
    const fallbackLabel = `Annotation ${annotationIdRef.current}`
    const nextAnnotation = normalizeAnnotation(
      {
        id: `${selectedAnchorEpisode}-annotation-${annotationIdRef.current}`,
        label: fallbackLabel,
        text: '',
        category: 'movement',
        color:
          ANNOTATION_SEED_COLORS[
            annotationIdRef.current % ANNOTATION_SEED_COLORS.length
          ],
        startTime,
        endTime: Number((startTime + 1).toFixed(2)),
        tags: ['manual', 'language'],
        source: 'user',
      },
      String(selectedAnchorEpisode),
    )

    if (!nextAnnotation) return

    setAnnotations((current) => [...current, nextAnnotation])
    setSelectedAnnotationId(nextAnnotation.id)
    setHasUnsavedChanges(true)
  }

  function updateAnnotation(
    annotationId: string,
    patch: Partial<AnnotationItem>,
  ): void {
    setAnnotations((currentAnnotations) =>
      currentAnnotations.map((annotation) => {
        if (annotation.id !== annotationId) return annotation

        const nextText =
          patch.text !== undefined ? patch.text : annotation.text
        const nextStartTime =
          patch.startTime !== undefined
            ? Math.max(Number(patch.startTime) || 0, 0)
            : annotation.startTime
        const rawEndTime =
          patch.endTime !== undefined
            ? patch.endTime === null
              ? null
              : Math.max(Number(patch.endTime) || 0, 0)
            : annotation.endTime
        const nextEndTime =
          rawEndTime === null ? null : Math.max(rawEndTime, nextStartTime)

        return {
          ...annotation,
          ...patch,
          text: nextText,
          startTime: nextStartTime,
          endTime: nextEndTime,
          label: deriveAnnotationLabel(
            nextText,
            patch.label || annotation.label || `Annotation ${annotationId}`,
          ),
        }
      }),
    )
    setHasUnsavedChanges(true)
  }

  function deleteAnnotation(annotationId: string): void {
    setAnnotations((current) =>
      current.filter((annotation) => annotation.id !== annotationId),
    )
    setHasUnsavedChanges(true)
  }

  function jumpToTime(timeValue: number): void {
    const player = videoRef.current
    if (!player || !effectiveSelectedVideo) return

    const boundedTime = clampToClipWindow(
      effectiveSelectedVideo,
      getClipStart(effectiveSelectedVideo) +
        clampAnnotationTime(timeValue, Number.POSITIVE_INFINITY),
      player.duration,
    )
    player.currentTime = boundedTime
  }

  async function handleSaveAnnotations(): Promise<boolean> {
    if (!selectedAnchorEpisode) return false

    setSaveState((current) => ({ ...current, isSaving: true, error: '' }))

    try {
      const saved = await saveAnnotations(selectedAnchorEpisode, taskContext, annotations)
      const normalizedAnnotations = (saved.annotations || [])
        .map((item) => normalizeAnnotation(item, String(selectedAnchorEpisode)))
        .filter((item): item is AnnotationItem => item !== null)
      annotationIdRef.current = normalizedAnnotations.length
      setAnnotations(normalizedAnnotations)
      setSelectedAnnotationId((currentValue) =>
        normalizedAnnotations.some((annotation) => annotation.id === currentValue)
          ? currentValue
          : normalizedAnnotations[0]?.id ?? null,
      )
      setSavedTaskContext(saved.task_context || taskContext)
      setSaveState({
        isSaving: false,
        error: '',
        versionNumber: saved.version_number || 0,
        savedAt: saved.updated_at || saved.created_at || '',
      })
      setHasUnsavedChanges(false)
      return true
    } catch (error) {
      setSaveState((current) => ({
        ...current,
        isSaving: false,
        error: error instanceof Error ? error.message : 'Failed to save annotations',
      }))
      return false
    }
  }

  async function handleRunPropagation(): Promise<void> {
    if (!selectedAnchorEpisode) return

    setPropagationState({ isRunning: true, error: '' })

    try {
      if (hasUnsavedChanges || saveState.versionNumber === 0) {
        const saved = await handleSaveAnnotations()
        if (!saved) {
          setPropagationState({ isRunning: false, error: '' })
          return
        }
      }

      await runPropagation(selectedAnchorEpisode)
      await loadPropagationResults()
    } catch (error) {
      setPropagationState({
        isRunning: false,
        error: error instanceof Error ? error.message : 'Failed to run propagation',
      })
    }
  }

  async function focusAnchorEpisode(nextEpisode: number): Promise<void> {
    if (nextEpisode === selectedAnchorEpisode) return

    if (hasUnsavedChanges) {
      const saved = await handleSaveAnnotations()
      if (!saved) return
    }

    setSelectedAnchorEpisode(nextEpisode)
  }

  if (!prototypeResults?.clusters.length) {
    return (
      <div className="annotation-panel__empty">
        <p>{copy.runPrototypeFirst}</p>
      </div>
    )
  }

  if (!selectedAnchorEpisode) {
    return (
      <div className="annotation-panel__empty">
        <p>{copy.selectAnchor}</p>
      </div>
    )
  }

  return (
    <div className="annotation-panel">
      <div className="annotation-panel__anchor-strip">
        <div className="annotation-panel__anchor-head">
          <div>
            <h4>{copy.anchors}</h4>
            <p>{copy.anchorsDesc}</p>
          </div>
          {hasUnsavedChanges ? (
            <span className="annotation-pill annotation-pill--warn">
              {copy.saveBeforeSwitch}
            </span>
          ) : null}
        </div>
        <div className="annotation-panel__anchor-list">
          {anchorItems.map((item) => (
            <button
              key={item.episodeIndex}
              type="button"
              className={
                item.episodeIndex === selectedAnchorEpisode
                  ? 'annotation-anchor-card is-selected'
                  : 'annotation-anchor-card'
              }
              onClick={() => void focusAnchorEpisode(item.episodeIndex)}
            >
              <div className="annotation-anchor-card__head">
                <span>{copy.cluster} {item.clusterIndex + 1}</span>
                <strong>EP {item.episodeIndex}</strong>
              </div>
              <div className="annotation-anchor-card__meta">
                <span>{copy.members}: {item.memberCount}</span>
                <span>{copy.quality}: {item.qualityScore?.toFixed(1) ?? '-'}</span>
              </div>
              <div className="annotation-anchor-card__status">
                {item.annotated ? (
                  <span className="annotation-pill annotation-pill--ok">
                    {copy.annotated}
                  </span>
                ) : null}
                <span className={item.propagated ? 'annotation-pill annotation-pill--ok' : 'annotation-pill'}>
                  {item.propagated ? copy.propagationDone : copy.propagationPending}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="annotation-panel__toolbar">
        <div className="annotation-panel__toolbar-status">
          <span className="annotation-pill">
            {copy.workspaceStatus}
          </span>
          <span className="annotation-pill">
            {copy.savedVersion}: {saveState.versionNumber || copy.notSavedYet}
          </span>
          <span className="annotation-pill">
            {copy.annotationCount}: {annotations.length}
          </span>
          {latestPropagation ? (
            <span className="annotation-pill annotation-pill--ok">
              {copy.targetCount}: {latestPropagation.target_count}
            </span>
          ) : null}
        </div>
        <div className="annotation-panel__toolbar-actions">
          <button
            type="button"
            className="annotation-primary-button"
            onClick={() => void handleSaveAnnotations()}
            disabled={saveState.isSaving || workspaceLoading}
          >
            {saveState.isSaving ? copy.saving : copy.saveAnnotationVersion}
          </button>
          <button
            type="button"
            className="annotation-primary-button"
            onClick={() => void handleRunPropagation()}
            disabled={saveState.isSaving || workspaceLoading || propagationState.isRunning}
          >
            {propagationState.isRunning
              ? copy.propagating
              : hasUnsavedChanges || saveState.versionNumber === 0
                ? copy.saveAndPropagate
                : copy.runPropagation}
          </button>
        </div>
      </div>

      {workspaceError ? <div className="status-panel error">{workspaceError}</div> : null}
      {saveState.error ? <div className="status-panel error">{saveState.error}</div> : null}
      {propagationState.error ? (
        <div className="status-panel error">{propagationState.error}</div>
      ) : null}
      {workspaceLoading ? <div className="status-panel">{copy.loadingWorkspace}</div> : null}

      {workspace && !workspaceLoading ? (
        <>
          <AnnotationWorkspaceCard
            videoRef={videoRef}
            videoSource={effectiveSelectedVideo?.url || ''}
            videoTitle={effectiveSelectedVideo?.path || ''}
            fps={Number(workspace.summary.fps) || 30}
            streamLabel={effectiveSelectedVideo?.stream || ''}
            chunkLabel={
              effectiveSelectedVideo
                ? effectiveSelectedVideo.path.split('/').slice(-2, -1)[0] || ''
                : ''
            }
            currentFrame={currentFrame}
            isPaused={isStudioPaused}
            videoCurrentTime={playbackState.time}
            timelineDuration={timelineDuration}
            annotations={annotations}
            selectedAnnotationId={selectedAnnotationId}
            onSelectAnnotation={setSelectedAnnotationId}
            onCreateAnnotation={createAnnotation}
            onUpdateAnnotation={updateAnnotation}
            onDeleteAnnotation={deleteAnnotation}
            onJumpToTime={jumpToTime}
          />

          <section className="annotation-stream-switcher">
            <div className="annotation-stream-switcher__head">
              <span>{copy.switchVideo}</span>
            </div>
            <div className="annotation-stream-switcher__list">
              {workspace.videos.length ? (
                workspace.videos.map((video) => (
                  <button
                    key={video.path}
                    type="button"
                    className={
                      video.path === effectiveSelectedVideo?.path
                        ? 'annotation-stream-pill is-selected'
                        : 'annotation-stream-pill'
                    }
                    onClick={() => setSelectedVideoPath(video.path)}
                  >
                    {video.stream || video.path}
                  </button>
                ))
              ) : (
                <span className="annotation-stream-switcher__empty">
                  {copy.noVideoData}
                </span>
              )}
            </div>
          </section>

          <section className="episode-preview-trajectory-panel">
            <div className="episode-preview-trajectory-head">
              <span>{copy.syncedAxes}</span>
              <strong>{copy.syncedAxesHint}</strong>
            </div>
            <div className="joint-comparison-focus-strip">
              <div className="joint-comparison-focus-metric">
                <span>{copy.focusJoint}</span>
                <strong>{activeComparisonEntry?.label || copy.unknownJoint}</strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.currentCursor}</span>
                <strong>{formatSeconds(playbackState.time)}s</strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.focusFrame}</span>
                <strong>{comparisonSnapshot.frame_index ?? '-'}</strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.focusActionValue}</span>
                <strong>{formatValue(comparisonSnapshot.action_value)}</strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.focusStateValue}</span>
                <strong>{formatValue(comparisonSnapshot.state_value)}</strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.restoreSource}</span>
                <strong>
                  {savedTaskContext?.source || comparisonSnapshot.source || '-'}
                </strong>
              </div>
              <div className="joint-comparison-focus-metric">
                <span>{copy.savedAt}</span>
                <strong>{saveState.savedAt || copy.notSavedYet}</strong>
              </div>
            </div>
            <JointComparisonGrid
              jointTrajectory={workspace.joint_trajectory}
              currentTime={playbackState.time}
              copy={{
                noJointData: copy.noJointData,
                actionSeries: copy.actionSeries,
                stateSeries: copy.stateSeries,
              }}
              activeKey={selectedComparisonKey}
              onSelectEntry={setSelectedComparisonKey}
            />
          </section>
        </>
      ) : null}
    </div>
  )
}
