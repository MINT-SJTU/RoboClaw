import { useEffect, useMemo, useState } from 'react'
import { useI18n } from '../controllers/i18n'
import { useWorkflow, type QualityEpisodeResult } from '../controllers/curation'
import { ActionButton, GlassPanel, MetricCard } from '../components/ux'

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

function formatIssueDetail(issue: Record<string, unknown>): string {
  const message = issue['message']
  return typeof message === 'string' && message.trim() ? message : ''
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

function MiniBarChart({
  title,
  items,
}: {
  title: string
  items: Array<{ label: string; count: number }>
}) {
  const maxValue = Math.max(...items.map((item) => item.count), 1)

  return (
    <GlassPanel className="quality-chart-card">
      <div className="quality-chart-card__title">{title}</div>
      <div className="quality-chart-card__bars">
        {items.length === 0 ? (
          <div className="quality-chart-card__empty">No data</div>
        ) : (
          items.map((item) => (
            <div key={item.label} className="quality-chart-card__row">
              <div className="quality-chart-card__label">{item.label}</div>
              <div className="quality-chart-card__track">
                <div
                  className="quality-chart-card__fill"
                  style={{ width: `${(item.count / maxValue) * 100}%` }}
                />
              </div>
              <div className="quality-chart-card__value">{item.count}</div>
            </div>
          ))
        )}
      </div>
    </GlassPanel>
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
    qualityResults,
    workflowState,
    deleteQualityResults,
    publishQualityParquet,
    getQualityCsvUrl,
    fetchAnnotationWorkspace,
    qualityThresholds,
    setQualityThreshold,
    selectDataset,
    stopPolling,
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
  const isRunning = qStage?.status === 'running'
  const isPaused = qStage?.status === 'paused'
  const controlsLocked = isRunning || isPaused
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

  const thresholdGroups = [
    {
      validator: 'metadata',
      fields: [
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
    <div className="page-enter quality-view">
      <div className="quality-view__hero">
        <div>
          <h2 className="quality-view__title">{t('qualityPageTitle')}</h2>
          <p className="quality-view__desc">{t('qualityPageDesc')}</p>
        </div>
      </div>

      {selectedDataset && datasetInfo ? (
        <div className="workflow-view__info-bar">
          <span>{datasetInfo.name}</span>
          <span>{datasetInfo.total_episodes} {t('episodes')}</span>
          <span>{datasetInfo.fps} fps</span>
          <span>{datasetInfo.robot_type}</span>
        </div>
      ) : (
        <GlassPanel className="quality-view__empty">
          {t('noWorkflowDataset')}
        </GlassPanel>
      )}

      <div className="quality-layout">
        <div className="quality-layout__main">
          <div className="quality-kpis">
            <MetricCard
              label={t('totalEpisodes')}
              value={qualityResults?.total ?? '--'}
            />
            <MetricCard
              label={t('passedEpisodes')}
              value={qualityResults?.passed ?? '--'}
              accent="sage"
            />
            <MetricCard
              label={t('failedEpisodes')}
              value={qualityResults?.failed ?? '--'}
              accent="coral"
            />
            <MetricCard
              label={t('score')}
              value={qualityResults ? qualityResults.overall_score.toFixed(1) : '--'}
              accent="amber"
            />
          </div>

          <div className="quality-charts">
            <MiniBarChart
              title={t('issueDistribution')}
              items={issueDistribution(filteredEpisodes)}
            />
            <MiniBarChart
              title={t('scoreDistribution')}
              items={scoreHistogram(filteredEpisodes)}
            />
          </div>

	          <GlassPanel className="quality-results-card">
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
	                      <tr key={episode.episode_index}>
	                        <td>
                            <button
                              type="button"
                              className="quality-episode-link"
                              onClick={() => void handleReviewEpisode(episode.episode_index)}
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
          </GlassPanel>
        </div>

	        <GlassPanel className="quality-layout__sidebar">
          <div className="quality-sidebar__section">
            <h3>{t('qualityValidation')}</h3>
            <p>{t('qualityOverview')}</p>
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
                            {group.fields.map((field) => (
                              <label key={field.key} className="quality-threshold-field">
                                <span>{field.label}</span>
                                <input
                                  type="number"
                                  step={field.step}
                                  value={qualityThresholds[field.key]}
                                  disabled={!enabled || controlsLocked}
                                  onChange={(event) =>
                                    setQualityThreshold(field.key, Number(event.target.value))
                                  }
                                />
                              </label>
                            ))}
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
	            <ActionButton
	              type="button"
	              disabled={
                  !selectedDataset
                  || isRunning
                  || (!isPaused && selectedValidators.length === 0)
                }
              onClick={() =>
                void (isPaused ? resumeQualityValidation() : runQualityValidation())
              }
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
	        </GlassPanel>
	      </div>
	    </div>
  )
}
