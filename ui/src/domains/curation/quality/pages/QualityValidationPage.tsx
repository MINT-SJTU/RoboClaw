import { useEffect, useMemo, useState } from 'react'
import { useI18n } from '@/i18n'
import { useWorkflow, type QualityEpisodeResult } from '@/domains/curation/store/useCurationStore'
import { ActionButton, GlassPanel } from '@/shared/ui'

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

function formatIssueLabel(checkName: string, locale: 'zh' | 'en'): string {
  const labels: Record<string, { zh: string; en: string }> = {
    'info.json': { zh: '缺少信息文件', en: 'Missing info.json' },
    'episode identity': { zh: '回合索引缺失', en: 'Missing episode identity' },
    robot_type: { zh: '机器人类型缺失', en: 'Missing robot type' },
    fps: { zh: '帧率缺失', en: 'Missing FPS' },
    features: { zh: '特征定义缺失', en: 'Missing feature schema' },
    parquet_data: { zh: 'Parquet 数据缺失', en: 'Missing parquet data' },
    videos: { zh: '视频文件缺失', en: 'Missing video files' },
    task_description: { zh: '任务描述缺失', en: 'Missing task description' },
    length: { zh: '回合时长异常', en: 'Episode duration issue' },
    timestamps: { zh: '时间戳不足', en: 'Insufficient timestamps' },
    monotonicity: { zh: '时间戳不单调', en: 'Timestamp monotonicity issue' },
    interval_cv: { zh: '采样间隔不稳定', en: 'Sampling interval variance' },
    estimated_frequency: { zh: '采样频率异常', en: 'Estimated frequency issue' },
    gap_ratio: { zh: '大时间间隔过多', en: 'Too many timestamp gaps' },
    frequency_consistency: { zh: '频率一致性差', en: 'Poor frequency consistency' },
    joint_series: { zh: '缺少关节序列', en: 'Missing joint series' },
    all_static_duration: { zh: '整体静止时间过长', en: 'All-joint static too long' },
    key_static_duration: { zh: '关键关节静止过长', en: 'Key-joint static too long' },
    max_velocity: { zh: '速度过高', en: 'Velocity too high' },
    duration: { zh: '动作时长异常', en: 'Action duration issue' },
    nan_ratio: { zh: '缺失值过多', en: 'Too many missing values' },
    video_count: { zh: '视频数量异常', en: 'Unexpected video count' },
    video_accessibility: { zh: '视频不可访问', en: 'Video accessibility issue' },
    video_resolution: { zh: '视频分辨率不足', en: 'Video resolution issue' },
    video_fps: { zh: '视频帧率不足', en: 'Video FPS issue' },
    overexposure_ratio: { zh: '过曝比例过高', en: 'Overexposure ratio too high' },
    underexposure_ratio: { zh: '欠曝比例过高', en: 'Underexposure ratio too high' },
    abnormal_frame_ratio: { zh: '异常黑白帧过多', en: 'Too many abnormal black/white frames' },
    color_shift: { zh: '色彩偏移过大', en: 'Color shift too high' },
    depth_streams: { zh: '缺少深度流', en: 'Missing depth streams' },
    depth_accessibility: { zh: '深度资源不可访问', en: 'Depth accessibility issue' },
    depth_invalid_ratio: { zh: '深度无效像素过多', en: 'Too many invalid depth pixels' },
    depth_continuity: { zh: '深度连续性不足', en: 'Depth continuity too low' },
    grasp_event_count: { zh: '抓放事件不足', en: 'Too few grasp/place events' },
    gripper_motion_span: { zh: '夹爪运动幅度不足', en: 'Gripper motion span too small' },
  }
  const label = labels[checkName]
  return label ? label[locale] : checkName
}

function formatCheckLabel(checkName: string, locale: 'zh' | 'en'): string {
  const labels: Record<string, { zh: string; en: string }> = {
    'info.json': { zh: '信息文件', en: 'Info file' },
    'episode identity': { zh: '回合索引', en: 'Episode identity' },
    robot_type: { zh: '机器人类型', en: 'Robot type' },
    fps: { zh: '数据帧率', en: 'Dataset FPS' },
    features: { zh: '特征定义', en: 'Feature schema' },
    parquet_data: { zh: 'Parquet 数据', en: 'Parquet data' },
    videos: { zh: '视频文件', en: 'Video files' },
    task_description: { zh: '任务描述', en: 'Task description' },
    length: { zh: '回合时长', en: 'Episode duration' },
    timestamps: { zh: '时间戳', en: 'Timestamps' },
    monotonicity: { zh: '时间戳单调性', en: 'Timestamp monotonicity' },
    interval_cv: { zh: '采样间隔 CV', en: 'Sampling interval CV' },
    estimated_frequency: { zh: '估算采样频率', en: 'Estimated frequency' },
    gap_ratio: { zh: '大间隔比例', en: 'Large gap ratio' },
    frequency_consistency: { zh: '频率一致性', en: 'Frequency consistency' },
    joint_series: { zh: '关节序列', en: 'Joint series' },
    all_static_duration: { zh: '整体最长静止', en: 'All-joint static duration' },
    key_static_duration: { zh: '关键关节最长静止', en: 'Key-joint static duration' },
    max_velocity: { zh: '最大速度', en: 'Maximum velocity' },
    duration: { zh: '动作时长', en: 'Action duration' },
    nan_ratio: { zh: '缺失值比例', en: 'Missing value ratio' },
    video_count: { zh: '视频数量', en: 'Video count' },
    video_accessibility: { zh: '视频可访问性', en: 'Video accessibility' },
    video_resolution: { zh: '视频分辨率', en: 'Video resolution' },
    video_fps: { zh: '视频帧率', en: 'Video FPS' },
    overexposure_ratio: { zh: '过曝比例', en: 'Overexposure ratio' },
    underexposure_ratio: { zh: '欠曝比例', en: 'Underexposure ratio' },
    abnormal_frame_ratio: { zh: '异常黑白帧比例', en: 'Abnormal black/white frame ratio' },
    color_shift: { zh: '色彩偏移', en: 'Color shift' },
    depth_streams: { zh: '深度流', en: 'Depth streams' },
    depth_accessibility: { zh: '深度可访问性', en: 'Depth accessibility' },
    depth_invalid_ratio: { zh: '深度无效像素比例', en: 'Invalid depth pixel ratio' },
    depth_continuity: { zh: '深度连续性', en: 'Depth continuity' },
    grasp_event_count: { zh: '抓放事件数', en: 'Grasp/place event count' },
    gripper_motion_span: { zh: '夹爪运动幅度', en: 'Gripper motion span' },
  }
  return labels[checkName]?.[locale] ?? checkName
}

function formatIssueDetail(issue: Record<string, unknown>): string {
  const message = issue['message']
  return typeof message === 'string' && message.trim() ? message : ''
}

function formatValidatorLabel(name: string, locale: 'zh' | 'en'): string {
  const labels: Record<string, { zh: string; en: string }> = {
    metadata: { zh: '元数据', en: 'Metadata' },
    timing: { zh: '时序', en: 'Timing' },
    action: { zh: '动作连续性', en: 'Action continuity' },
    visual: { zh: '视觉质量', en: 'Visual quality' },
    depth: { zh: '深度', en: 'Depth' },
    ee_trajectory: { zh: '末端轨迹', en: 'EE trajectory' },
  }
  return labels[name]?.[locale] ?? name
}

function formatQualityValue(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : Number(value.toFixed(6)).toString()
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  if (typeof value === 'string') {
    return value
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function hasQualityValue(value: unknown): boolean {
  if (value === null || value === undefined) {
    return false
  }
  if (typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).length > 0
  }
  return String(value).trim().length > 0
}

function groupQualityIssues(issues: Array<Record<string, unknown>>): Array<{
  validator: string
  checks: Array<Record<string, unknown>>
}> {
  const groups = new Map<string, Array<Record<string, unknown>>>()
  issues.forEach((issue) => {
    const operatorName = issue['operator_name']
    const validator = typeof operatorName === 'string' && operatorName.trim()
      ? operatorName
      : 'unknown'
    groups.set(validator, [...(groups.get(validator) || []), issue])
  })
  return Array.from(groups.entries()).map(([validator, checks]) => ({ validator, checks }))
}

function isFailingIssue(issue: Record<string, unknown>): boolean {
  return issue['passed'] !== true
}

function collectIssueTypes(episodes: QualityEpisodeResult[]): string[] {
  const issueTypes = new Set<string>()
  episodes.forEach((episode) => {
    ;(episode.issues || []).forEach((issue) => {
      if (!isFailingIssue(issue)) {
        return
      }
      const checkName = issue['check_name']
      if (typeof checkName === 'string' && checkName.trim()) {
        issueTypes.add(checkName)
      }
    })
  })
  return Array.from(issueTypes).sort()
}

function issueDistribution(episodes: QualityEpisodeResult[]): Array<{ label: string; count: number }> {
  const counts = new Map<string, number>()
  episodes.forEach((episode) => {
    ;(episode.issues || []).forEach((issue) => {
      if (!isFailingIssue(issue)) {
        return
      }
      const checkName = issue['check_name']
      if (typeof checkName !== 'string' || !checkName.trim()) {
        return
      }
      counts.set(checkName, (counts.get(checkName) || 0) + 1)
    })
  })
  return Array.from(counts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, 12)
}

function scoreHistogram(episodes: QualityEpisodeResult[]): Array<{ label: string; count: number }> {
  const bins = [
    { label: '0-20', min: 0, max: 20 },
    { label: '20-40', min: 20, max: 40 },
    { label: '40-60', min: 40, max: 60 },
    { label: '60-80', min: 60, max: 80 },
    { label: '80-100', min: 80, max: 101 },
  ]
  return bins.map((bin) => ({
    label: bin.label,
    count: episodes.filter((episode) => episode.score >= bin.min && episode.score < bin.max).length,
  }))
}

interface PieSegment {
  label: string
  count: number
  color: string
}

function buildPieGradient(segments: PieSegment[]): string {
  const total = segments.reduce((sum, segment) => sum + segment.count, 0)
  if (total <= 0) {
    return 'conic-gradient(rgba(47,111,228,0.08) 0deg 360deg)'
  }

  let current = 0
  const stops = segments.map((segment) => {
    const start = current
    current += (segment.count / total) * 360
    return `${segment.color} ${start}deg ${current}deg`
  })
  return `conic-gradient(${stops.join(', ')})`
}

function clampPieSegments(
  segments: PieSegment[],
  options: {
    maxSegments?: number
    otherLabel: string
    otherColor: string
  },
): PieSegment[] {
  const {
    maxSegments = 4,
    otherLabel,
    otherColor,
  } = options
  const nonZero = segments.filter((segment) => segment.count > 0)
  if (nonZero.length <= maxSegments) {
    return nonZero
  }
  const head = nonZero.slice(0, maxSegments - 1)
  const tail = nonZero.slice(maxSegments - 1)
  return [
    ...head,
    {
      label: otherLabel,
      count: tail.reduce((sum, item) => sum + item.count, 0),
      color: otherColor,
    },
  ]
}

function PieChartCard({
  title,
  segments,
  centerLabel,
}: {
  title: string
  segments: PieSegment[]
  centerLabel: string
}) {
  const total = segments.reduce((sum, segment) => sum + segment.count, 0)
  const gradient = buildPieGradient(segments)

  return (
    <GlassPanel className="quality-pie-card">
      <div className="quality-pie-card__title">{title}</div>
      <div className="quality-pie-card__body">
        <div className="quality-pie-card__chart" style={{ backgroundImage: gradient }}>
          <div className="quality-pie-card__inner">
            <div className="quality-pie-card__total">{total}</div>
            <div className="quality-pie-card__caption">{centerLabel}</div>
          </div>
        </div>
        <div className="quality-pie-card__legend">
          {segments.length === 0 ? (
            <div className="quality-pie-card__empty">No data</div>
          ) : (
            segments.map((segment) => {
              const percent = total > 0 ? (segment.count / total) * 100 : 0
              return (
                <div key={segment.label} className="quality-pie-card__legend-item">
                  <span
                    className="quality-pie-card__dot"
                    style={{ backgroundColor: segment.color }}
                  />
                  <span className="quality-pie-card__legend-label">{segment.label}</span>
                  <span className="quality-pie-card__legend-value">
                    {segment.count} · {percent.toFixed(0)}%
                  </span>
                </div>
              )
            })
          )}
        </div>
      </div>
    </GlassPanel>
  )
}

function QualityDetailInspector({
  episode,
  locale,
}: {
  episode: QualityEpisodeResult
  locale: 'zh' | 'en'
}) {
  const copy = locale === 'zh'
    ? {
      title: '检测详情',
      validatorSummary: '验证器汇总',
      checks: '检查项',
      actualValue: '实际检测值',
      noDetails: '没有详细检查记录',
    }
    : {
      title: 'Validation Details',
      validatorSummary: 'Validator Summary',
      checks: 'Checks',
      actualValue: 'Measured Value',
      noDetails: 'No detailed check records',
    }
  const issueGroups = groupQualityIssues(episode.issues || [])
  const validatorEntries = Object.entries(episode.validators || {})

  return (
    <div className="quality-detail-inspector">
      <div className="quality-detail-inspector__head">
        <div>
          <div className="quality-detail-inspector__eyebrow">{copy.title}</div>
          <h4>Episode {episode.episode_index}</h4>
        </div>
        <div className="quality-detail-inspector__score">
          <span>{episode.score.toFixed(1)}</span>
          <span className={cn(episode.passed ? 'is-pass' : 'is-fail')}>
            {episode.passed ? (locale === 'zh' ? '通过' : 'Passed') : (locale === 'zh' ? '未通过' : 'Failed')}
          </span>
        </div>
      </div>

      {validatorEntries.length > 0 && (
        <div className="quality-detail-summary" aria-label={copy.validatorSummary}>
          {validatorEntries.map(([name, validator]) => (
            <div
              key={name}
              className={cn('quality-detail-summary__item', validator.passed ? 'is-pass' : 'is-fail')}
            >
              <span>{formatValidatorLabel(name, locale)}</span>
              <strong>{validator.score.toFixed(1)}</strong>
            </div>
          ))}
        </div>
      )}

      {issueGroups.length > 0 ? (
        <div className="quality-detail-groups">
          {issueGroups.map((group) => (
            <section key={group.validator} className="quality-detail-group">
              <div className="quality-detail-group__title">
                <span>{formatValidatorLabel(group.validator, locale)}</span>
                <span>{group.checks.length} {copy.checks}</span>
              </div>
              <div className="quality-detail-checks">
                {group.checks.map((issue, index) => {
                  const checkName = issue['check_name']
                  const checkKey = typeof checkName === 'string' && checkName.trim()
                    ? checkName
                    : `check-${index}`
                  const passed = issue['passed'] === true
                  const detail = formatIssueDetail(issue)
                  const value = issue['value']
                  const hasValue = hasQualityValue(value)
                  return (
                    <div
                      key={`${group.validator}-${checkKey}-${index}`}
                      className={cn('quality-detail-check', passed ? 'is-pass' : 'is-fail')}
                    >
                      <div className="quality-detail-check__head">
                        <span className={cn('quality-detail-check__status', passed ? 'is-pass' : 'is-fail')}>
                          {passed ? (locale === 'zh' ? '通过' : 'Pass') : (locale === 'zh' ? '未通过' : 'Fail')}
                        </span>
                        <span className="quality-detail-check__name">
                          {formatCheckLabel(checkKey, locale)}
                        </span>
                        <span className="quality-detail-check__raw">{checkKey}</span>
                      </div>
                      {detail && <div className="quality-detail-check__message">{detail}</div>}
                      {hasValue && (
                        <div className="quality-detail-value">
                          <div className="quality-detail-value__label">{copy.actualValue}</div>
                          <pre>{formatQualityValue(value)}</pre>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="quality-detail-inspector__empty">{copy.noDetails}</div>
      )}
    </div>
  )
}

export default function QualityValidationView() {
  const { t, locale } = useI18n()
  const {
    selectedDataset,
    datasetInfo,
    selectedValidators,
    toggleValidator,
    runQualityValidation,
    pauseQualityValidation,
    resumeQualityValidation,
    qualityRunning,
    qualityDefaults,
    qualityResults,
    workflowState,
    deleteQualityResults,
    publishQualityParquet,
    getQualityCsvUrl,
    fetchAnnotationWorkspace,
    qualityThresholds,
    setQualityThreshold,
    selectDataset,
    prepareRemoteDatasetForWorkflow,
    stopPolling,
    selectedDatasetIsRemotePrepared,
  } = useWorkflow()
  const [failureOnly, setFailureOnly] = useState(false)
  const [issueType, setIssueType] = useState('')
  const [publishing, setPublishing] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [publishError, setPublishError] = useState('')
  const [publishMessage, setPublishMessage] = useState('')
  const [selectedEpisodeForReview, setSelectedEpisodeForReview] = useState<number | null>(null)
  const [reviewVideoUrl, setReviewVideoUrl] = useState('')
  const [reviewVideoLabel, setReviewVideoLabel] = useState('')
  const [reviewLoading, setReviewLoading] = useState(false)
  const [reviewError, setReviewError] = useState('')
  const [runQualityError, setRunQualityError] = useState('')
  const [rightRailCollapsed, setRightRailCollapsed] = useState(false)
  const [hoveredEpisodeIndex, setHoveredEpisodeIndex] = useState<number | null>(null)
  const [collapsedThresholdValidators, setCollapsedThresholdValidators] = useState<string[]>([
    'metadata',
    'timing',
    'action',
    'visual',
    'depth',
  ])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  useEffect(() => {
    if (selectedDataset && !datasetInfo) {
      void selectDataset(selectedDataset)
    }
  }, [selectedDataset, datasetInfo, selectDataset])

  const qStage = workflowState?.stages.quality_validation
  const isRunning = qualityRunning || qStage?.status === 'running'
  const isPaused = qStage?.status === 'paused'
  const controlsLocked = isRunning || isPaused
  const datasetIsWorkflowReady = Boolean(workflowState) || selectedDatasetIsRemotePrepared
  const episodes = qualityResults?.episodes || []
  const canDeleteResults =
    Boolean(selectedDataset)
    && !isRunning
    && (
      episodes.length > 0
      || qStage?.status === 'completed'
      || qStage?.status === 'paused'
      || qStage?.status === 'error'
    )
  const availableIssueTypes = useMemo(() => collectIssueTypes(episodes), [episodes])
  const filteredEpisodes = useMemo(() => {
    return episodes.filter((episode) => {
      if (failureOnly && episode.passed) {
        return false
      }
      if (issueType) {
        return (episode.issues || []).some(
          (issue) => isFailingIssue(issue) && issue.check_name === issueType,
        )
      }
      return true
    })
  }, [episodes, failureOnly, issueType])
  const detailEpisode = useMemo(() => {
    if (hoveredEpisodeIndex === null) {
      return null
    }
    return filteredEpisodes.find((episode) => episode.episode_index === hoveredEpisodeIndex) || null
  }, [filteredEpisodes, hoveredEpisodeIndex])
  const displayedEpisodeCount = useMemo(() => {
    if (failureOnly || issueType) {
      return filteredEpisodes.length
    }
    const completed = qStage?.summary?.['completed']
    if (typeof completed === 'number') {
      return completed
    }
    if (episodes.length > 0) {
      return episodes.length
    }
    return qualityResults?.total ?? '--'
  }, [episodes.length, failureOnly, filteredEpisodes.length, issueType, qStage?.summary, qualityResults?.total])

  const otherLabel = locale === 'zh' ? '其他' : 'Other'
  const qualityPieSegments = useMemo<PieSegment[]>(() => ([
    { label: t('passedEpisodes'), count: qualityResults?.passed ?? 0, color: '#33c36b' },
    { label: t('failedEpisodes'), count: qualityResults?.failed ?? 0, color: '#f26b6b' },
  ]).filter((segment) => segment.count > 0), [qualityResults, t])
  const issuePieSegments = useMemo<PieSegment[]>(
    () =>
      clampPieSegments(
        issueDistribution(episodes).map((item, index) => ({
          label: formatIssueLabel(item.label, locale),
          count: item.count,
          color: ['#4d87ff', '#7c68ff', '#f59e0b', '#ec4899', '#14b8a6'][index % 5],
        })),
        { maxSegments: 4, otherLabel, otherColor: '#94a3b8' },
      ),
    [episodes, locale, otherLabel],
  )
  const scorePieSegments = useMemo<PieSegment[]>(
    () =>
      scoreHistogram(episodes)
        .map((item, index) => ({
          label: item.label,
          count: item.count,
          color: ['#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'][index % 5],
        }))
        .filter((segment) => segment.count > 0),
    [episodes],
  )

  async function handlePublishParquet(): Promise<void> {
    setPublishing(true)
    setPublishError('')
    setPublishMessage('')
    try {
      const result = await publishQualityParquet()
      setPublishMessage(`${t('qualityParquet')}: ${result.path}`)
    } catch (error) {
      setPublishError(error instanceof Error ? error.message : 'Publish failed')
    } finally {
      setPublishing(false)
    }
  }

  async function handleDeleteQualityResults(): Promise<void> {
    if (!selectedDataset || !window.confirm(t('deleteQualityResultsConfirm'))) {
      return
    }
    setDeleting(true)
    setPublishError('')
    setPublishMessage('')
    try {
      await deleteQualityResults()
      setFailureOnly(false)
      setIssueType('')
      setSelectedEpisodeForReview(null)
      setReviewVideoUrl('')
      setReviewVideoLabel('')
      setReviewError('')
      setPublishMessage(t('deleteQualityResultsSuccess'))
    } catch (error) {
      setPublishError(error instanceof Error ? error.message : 'Delete failed')
    } finally {
      setDeleting(false)
    }
  }

  async function handleRunQualityAction(): Promise<void> {
    setRunQualityError('')
    try {
      if (!datasetIsWorkflowReady && selectedDataset) {
        await prepareRemoteDatasetForWorkflow(selectedDataset, false)
      }
      if (isPaused) {
        await resumeQualityValidation()
      } else {
        await runQualityValidation()
      }
    } catch (error) {
      setRunQualityError(error instanceof Error ? error.message : t('qualityRunFailed'))
    }
  }

  async function handleReviewEpisode(episodeIndex: number): Promise<void> {
    setSelectedEpisodeForReview(episodeIndex)
    setReviewLoading(true)
    setReviewError('')
    try {
      const workspace = await fetchAnnotationWorkspace(episodeIndex)
      const firstVideo = workspace.videos[0]
      if (!firstVideo) {
        setReviewVideoUrl('')
        setReviewVideoLabel('')
        setReviewError('No video available for this episode')
        return
      }
      setReviewVideoUrl(firstVideo.url)
      setReviewVideoLabel(firstVideo.path)
    } catch (error) {
      setReviewVideoUrl('')
      setReviewVideoLabel('')
      setReviewError(error instanceof Error ? error.message : 'Failed to load episode video')
    } finally {
      setReviewLoading(false)
    }
  }

  const thresholdGroups: Array<{
    validator: string
    fields: Array<{ key: string; label: string; step: number; kind?: 'boolean' }>
  }> = [
    {
      validator: 'metadata',
      fields: [
        { key: 'metadata_require_info_json', label: '检查 meta/info.json', step: 1, kind: 'boolean' },
        { key: 'metadata_require_episode_metadata', label: '检查 episode 元数据', step: 1, kind: 'boolean' },
        { key: 'metadata_require_data_files', label: '检查数据文件缺失', step: 1, kind: 'boolean' },
        { key: 'metadata_require_videos', label: '检查视频文件缺失', step: 1, kind: 'boolean' },
        { key: 'metadata_require_task_description', label: '检查任务描述', step: 1, kind: 'boolean' },
        { key: 'metadata_min_duration_s', label: '最小时长 (s)', step: 0.1 },
      ],
    },
    {
      validator: 'timing',
      fields: [
        { key: 'timing_min_monotonicity', label: '最小单调性', step: 0.001 },
        { key: 'timing_max_interval_cv', label: '最大间隔 CV', step: 0.001 },
        { key: 'timing_min_frequency_hz', label: '最小频率 (Hz)', step: 0.1 },
        { key: 'timing_max_gap_ratio', label: '最大 gap 比例', step: 0.001 },
        { key: 'timing_min_frequency_consistency', label: '最小频率一致性', step: 0.001 },
      ],
    },
    {
      validator: 'action',
      fields: [
        { key: 'action_static_threshold', label: '静止阈值', step: 0.0001 },
        { key: 'action_max_all_static_s', label: '整体最长静止 (s)', step: 0.1 },
        { key: 'action_max_key_static_s', label: '关键关节最长静止 (s)', step: 0.1 },
        { key: 'action_max_velocity_rad_s', label: '最大速度 (rad/s)', step: 0.01 },
        { key: 'action_min_duration_s', label: '动作最小时长 (s)', step: 0.1 },
        { key: 'action_max_nan_ratio', label: '最大缺失比例', step: 0.001 },
      ],
    },
    {
      validator: 'visual',
      fields: [
        { key: 'visual_min_resolution_width', label: '最小宽度', step: 1 },
        { key: 'visual_min_resolution_height', label: '最小高度', step: 1 },
        { key: 'visual_min_frame_rate', label: '最小帧率 (Hz)', step: 0.1 },
        { key: 'visual_frame_rate_tolerance', label: '帧率容差', step: 0.1 },
        { key: 'visual_color_shift_max', label: '最大色偏', step: 0.01 },
        { key: 'visual_overexposure_ratio_max', label: '最大过曝比例', step: 0.01 },
        { key: 'visual_underexposure_ratio_max', label: '最大欠曝比例', step: 0.01 },
        { key: 'visual_abnormal_black_ratio_max', label: '最大黑帧比例', step: 0.01 },
        { key: 'visual_abnormal_white_ratio_max', label: '最大白帧比例', step: 0.01 },
        { key: 'visual_min_video_count', label: '最少视频数量', step: 1 },
        { key: 'visual_min_accessible_ratio', label: '最小可访问比例', step: 0.01 },
      ],
    },
    {
      validator: 'depth',
      fields: [
        { key: 'depth_min_stream_count', label: '最少深度流数量', step: 1 },
        { key: 'depth_min_accessible_ratio', label: '最小可访问比例', step: 0.01 },
        { key: 'depth_invalid_pixel_max', label: '最大无效像素比例', step: 0.01 },
        { key: 'depth_continuity_min', label: '最小连续性', step: 0.01 },
      ],
    },
    {
      validator: 'ee_trajectory',
      fields: [
        { key: 'ee_min_event_count', label: '最少抓放事件数', step: 1 },
        { key: 'ee_min_gripper_span', label: '最小夹爪幅度', step: 0.01 },
      ],
    },
  ] as const

  function toggleThresholdValidator(validator: string): void {
    setCollapsedThresholdValidators((current) =>
      current.includes(validator)
        ? current.filter((item) => item !== validator)
        : [...current, validator],
    )
  }

  return (
    <div className="page-enter quality-view pipeline-page quality-validation-page">
      {selectedDataset && datasetInfo ? (
        <div className="workflow-view__info-bar">
          <span>{datasetInfo.label}</span>
          <span>{datasetInfo.stats.total_episodes} {t('episodes')}</span>
          <span>{datasetInfo.stats.fps} fps</span>
          <span>{datasetInfo.stats.robot_type}</span>
        </div>
      ) : (
        <GlassPanel className="quality-view__empty">
          {t('noWorkflowDataset')}
        </GlassPanel>
      )}

      <div className={cn('quality-validation-shell', rightRailCollapsed && 'is-rail-collapsed')}>
        <div className="quality-validation-shell__main">
          <div className="quality-validation-overview">
            <GlassPanel className="quality-total-card">
              <div className="quality-total-card__eyebrow">{t('totalEpisodes')}</div>
              <div className="quality-total-card__value">{displayedEpisodeCount}</div>
            </GlassPanel>

            <div className="quality-validation-pies">
              <PieChartCard
                title={`${t('passedEpisodes')} / ${t('failedEpisodes')}`}
                segments={qualityPieSegments}
                centerLabel={t('episodes')}
              />
              <PieChartCard
                title={t('issueDistribution')}
                segments={issuePieSegments}
                centerLabel={locale === 'zh' ? '问题' : 'Issues'}
              />
              <PieChartCard
                title={t('scoreDistribution')}
                segments={scorePieSegments}
                centerLabel={locale === 'zh' ? '区间' : 'Bands'}
              />
            </div>
          </div>

          {runQualityError && (
            <GlassPanel className="quality-results-card">
              <div className="quality-sidebar__error">{runQualityError}</div>
            </GlassPanel>
          )}

          <GlassPanel
            className="quality-results-card"
            onMouseLeave={() => setHoveredEpisodeIndex(null)}
          >
            <div className="quality-results-card__head">
              <div>
                <h3>{t('qualityResults')}</h3>
                <p>
                  {filteredEpisodes.length} / {episodes.length} rows
                </p>
              </div>
              <div className="quality-results-card__filters">
                <label className="quality-checkbox">
                  <input
                    type="checkbox"
                    checked={failureOnly}
                    onChange={() => setFailureOnly((value) => !value)}
                  />
                  <span>{t('failureOnly')}</span>
                </label>
                <select
                  className="dataset-selector__select"
                  value={issueType}
                  onChange={(event) => setIssueType(event.target.value)}
                >
                  <option value="">{t('allIssues')}</option>
                  {availableIssueTypes.map((type) => (
                    <option key={type} value={type}>
                      {formatIssueLabel(type, locale)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="quality-table-wrap quality-results-table-wrap">
              <table className="quality-table">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>{t('score')}</th>
                    <th>{t('passed')}</th>
                    <th>{t('validators')}</th>
                    <th>{t('issueType')}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEpisodes.map((episode) => {
                    const failedValidators = Object.entries(episode.validators || {})
                      .filter(([, validator]) => !validator.passed)
                      .map(([name]) => name)
                    const issueNames = Array.from(
                      new Set(
                        (episode.issues || [])
                          .filter((issue) => isFailingIssue(issue))
                          .map((issue) => issue['check_name'])
                          .filter((name): name is string => typeof name === 'string' && Boolean(name)),
                      ),
                    )
                    const issueDetails = (episode.issues || [])
                      .filter((issue) => isFailingIssue(issue))
                      .map((issue) => {
                        const checkName = issue['check_name']
                        if (typeof checkName !== 'string' || !checkName) {
                          return null
                        }
                        return {
                          key: checkName,
                          label: formatIssueLabel(checkName, locale),
                          detail: formatIssueDetail(issue),
                        }
                      })
                      .filter((item): item is { key: string; label: string; detail: string } => Boolean(item))
                    return (
                      <tr
                        key={episode.episode_index}
                        className={cn(
                          'quality-result-row',
                          detailEpisode?.episode_index === episode.episode_index && 'is-inspected',
                        )}
                        tabIndex={0}
                        onMouseEnter={() => setHoveredEpisodeIndex(episode.episode_index)}
                        onFocus={() => setHoveredEpisodeIndex(episode.episode_index)}
                      >
                        <td>
                          <button
                            type="button"
                            className="quality-episode-link"
                            onMouseEnter={() => setHoveredEpisodeIndex(episode.episode_index)}
                            onFocus={() => setHoveredEpisodeIndex(episode.episode_index)}
                            onClick={() => {
                              setHoveredEpisodeIndex(episode.episode_index)
                              void handleReviewEpisode(episode.episode_index)
                            }}
                          >
                            {episode.episode_index}
                          </button>
                        </td>
                        <td>{episode.score.toFixed(1)}</td>
                        <td className={cn(episode.passed ? 'is-pass' : 'is-fail')}>
                          {episode.passed ? t('passed') : t('failed')}
                        </td>
                        <td>{failedValidators.join(', ') || '-'}</td>
                        <td>
                          {issueDetails.length > 0 ? (
                            <div className="quality-issue-list">
                              {issueDetails.map((issue) => (
                                <div key={`${episode.episode_index}-${issue.key}`} className="quality-issue-item">
                                  <div className="quality-issue-item__label">{issue.label}</div>
                                  {issue.detail && (
                                    <div className="quality-issue-item__detail">{issue.detail}</div>
                                  )}
                                </div>
                              ))}
                            </div>
                          ) : (
                            issueNames.join(', ') || '-'
                          )}
                        </td>
                      </tr>
                    )
                  })}
                  {filteredEpisodes.length === 0 && (
                    <tr>
                      <td colSpan={5} className="quality-table__empty">
                        No results
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {detailEpisode && (
              <QualityDetailInspector episode={detailEpisode} locale={locale} />
            )}
          </GlassPanel>
        </div>

        <aside className={cn('quality-validation-rail', rightRailCollapsed && 'is-collapsed')}>
          <GlassPanel className="quality-validation-rail__card">
            <button
              type="button"
              className={cn(
                'quality-validation-rail__toggle',
                rightRailCollapsed && 'is-collapsed',
              )}
              onClick={() => setRightRailCollapsed((value) => !value)}
              aria-expanded={!rightRailCollapsed}
              aria-label={rightRailCollapsed ? 'Expand quality rail' : 'Collapse quality rail'}
            >
              <span className="quality-validation-rail__toggle-icon">‹</span>
              <span className="quality-validation-rail__toggle-label">{t('qualityValidation')}</span>
            </button>

            <div
              className={cn(
                'quality-validation-rail__panel',
                rightRailCollapsed && 'is-collapsed',
              )}
              aria-hidden={rightRailCollapsed}
            >
              <div className="quality-sidebar__section">
                <h3>{t('qualityValidation')}</h3>
                <p>{t('qualityOverview')}</p>
                {qualityDefaults && (
                  <div className="quality-sidebar__path">
                    自动默认值:
                    {' '}
                    {qualityDefaults.profile.fps > 0 ? `${qualityDefaults.profile.fps} fps` : 'fps --'}
                    {qualityDefaults.profile.video_resolution
                      ? ` · ${qualityDefaults.profile.video_resolution.width}x${qualityDefaults.profile.video_resolution.height}`
                      : ''}
                    {' · '}
                    {qualityDefaults.checks.task_descriptions_present ? '任务描述存在' : '任务描述缺失'}
                  </div>
                )}
              </div>

              <div className="quality-sidebar__section">
                <div className="quality-sidebar__label">{t('validators')}</div>
                <div className="quality-threshold-groups">
                  {thresholdGroups.map((group) => {
                    const collapsed = collapsedThresholdValidators.includes(group.validator)
                    const enabled = selectedValidators.includes(group.validator)
                    return (
                      <div
                        key={group.validator}
                        className={cn(
                          'quality-threshold-group',
                          !enabled && 'is-disabled',
                        )}
                      >
                        <div className="quality-threshold-group__toggle">
                          <label className="quality-threshold-group__check">
                            <input
                              type="checkbox"
                              checked={enabled}
                              onChange={() => toggleValidator(group.validator)}
                              disabled={controlsLocked || !selectedDataset}
                            />
                            <span>
                              {t(group.validator as 'metadata' | 'timing' | 'action' | 'visual' | 'depth' | 'ee_trajectory')}
                            </span>
                          </label>
                          <button
                            type="button"
                            className="quality-threshold-group__chevron-btn"
                            onClick={() => toggleThresholdValidator(group.validator)}
                          >
                            <span className={cn('quality-threshold-group__chevron', !collapsed && 'is-open')}>
                              ▾
                            </span>
                          </button>
                        </div>
                        {!collapsed && (
                          <div className="quality-threshold-group__body">
                            {group.fields.length > 0 ? (
                              <div className="quality-threshold-list">
                                {group.fields.map((field) => {
                                  const value = qualityThresholds[field.key] ?? 0
                                  return (
                                    <label key={field.key} className="quality-threshold-field">
                                      <span>{field.label}</span>
                                      {field.kind === 'boolean' ? (
                                        <input
                                          type="checkbox"
                                          checked={value >= 0.5}
                                          disabled={!enabled || controlsLocked}
                                          onChange={(event) =>
                                            setQualityThreshold(field.key, event.target.checked ? 1 : 0)
                                          }
                                        />
                                      ) : (
                                        <input
                                          type="number"
                                          step={field.step}
                                          value={value}
                                          disabled={!enabled || controlsLocked}
                                          onChange={(event) =>
                                            setQualityThreshold(field.key, Number(event.target.value))
                                          }
                                        />
                                      )}
                                    </label>
                                  )
                                })}
                              </div>
                            ) : (
                              <div className="quality-threshold-empty">
                                这个验证器当前没有可调阈值
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              <div className="quality-sidebar__section">
                {!datasetIsWorkflowReady && selectedDataset && (
                  <div className="quality-sidebar__error">{t('qualityRequiresImportedDataset')}</div>
                )}
                <ActionButton
                  type="button"
                  disabled={
                    !selectedDataset
                    || isRunning
                    || (!isPaused && selectedValidators.length === 0)
                  }
                  onClick={() => void handleRunQualityAction()}
                  className="w-full justify-center"
                >
                  {isRunning ? t('running') : isPaused ? t('resumeQuality') : t('runQuality')}
                </ActionButton>
                {isRunning && (
                  <ActionButton
                    type="button"
                    variant="warning"
                    disabled={!selectedDataset}
                    onClick={() => void pauseQualityValidation()}
                    className="mt-3 w-full justify-center"
                  >
                    {t('pauseQuality')}
                  </ActionButton>
                )}
                {isPaused && (
                  <div className="quality-sidebar__path">
                    {t('paused')}
                    {typeof qStage?.summary?.['completed'] === 'number' && typeof qStage?.summary?.['total'] === 'number'
                      ? ` · ${qStage.summary['completed']} / ${qStage.summary['total']}`
                      : ''}
                  </div>
                )}
              </div>

              <div className="quality-sidebar__section">
                <a
                  href={getQualityCsvUrl(failureOnly)}
                  className={cn(
                    'quality-sidebar__link',
                    !selectedDataset && 'is-disabled',
                  )}
                  onClick={(event) => {
                    if (!selectedDataset) {
                      event.preventDefault()
                    }
                  }}
                >
                  {t('exportCsv')}
                </a>
                <ActionButton
                  type="button"
                  variant="secondary"
                  disabled={!selectedDataset || publishing}
                  onClick={() => void handlePublishParquet()}
                  className="w-full justify-center"
                >
                  {publishing ? t('publishing') : t('publishQualityParquet')}
                </ActionButton>
                <ActionButton
                  type="button"
                  variant="danger"
                  disabled={!canDeleteResults || deleting}
                  onClick={() => void handleDeleteQualityResults()}
                  className="w-full justify-center"
                >
                  {deleting ? t('deleting') : t('deleteQualityResults')}
                </ActionButton>
                {qualityResults?.working_parquet_path && (
                  <div className="quality-sidebar__path">
                    working: {qualityResults.working_parquet_path}
                  </div>
                )}
                {qualityResults?.published_parquet_path && (
                  <div className="quality-sidebar__path">
                    published: {qualityResults.published_parquet_path}
                  </div>
                )}
                {publishMessage && (
                  <div className="quality-sidebar__path">{publishMessage}</div>
                )}
                {publishError && (
                  <div className="quality-sidebar__error">{publishError}</div>
                )}
              </div>

              <div className="quality-sidebar__section">
                <div className="quality-sidebar__label">视频验证</div>
                {reviewLoading ? (
                  <div className="quality-sidebar__path">加载视频中...</div>
                ) : reviewError ? (
                  <div className="quality-sidebar__error">{reviewError}</div>
                ) : reviewVideoUrl ? (
                  <div className="quality-review-video">
                    <video
                      className="quality-review-video__player"
                      controls
                      preload="metadata"
                      playsInline
                      src={reviewVideoUrl}
                    />
                    <div className="quality-sidebar__path">
                      episode {selectedEpisodeForReview} · {reviewVideoLabel}
                    </div>
                  </div>
                ) : (
                  <div className="quality-sidebar__path">点击结果表中的 episode 编号开始验证视频</div>
                )}
              </div>
            </div>
          </GlassPanel>
        </aside>
      </div>
    </div>
  )
}
