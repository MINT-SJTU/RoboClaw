import { useEffect, useMemo, useState } from 'react'
import { ActionButton, GlassPanel, MetricCard } from '@/shared/ui'
import { useI18n } from '@/i18n'
import {
  useWorkflow,
  type AlignmentOverviewRow,
  type AlignmentOverviewSpan,
} from '@/domains/curation/store/useCurationStore'
import { cn } from '@/shared/lib/cn'

type ChartDimension =
  | 'alignment_status'
  | 'issue_types'
  | 'score_bands'
  | 'failed_validators'
  | 'tasks'

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

function formatIssueDetail(issue: Record<string, unknown>): string {
  const message = issue['message']
  return typeof message === 'string' && message.trim() ? message : ''
}

function formatQualityScalar(value: unknown, locale: 'zh' | 'en'): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : Number(value.toFixed(6)).toString()
  }
  if (typeof value === 'boolean') {
    return value ? (locale === 'zh' ? '是' : 'true') : (locale === 'zh' ? '否' : 'false')
  }
  if (typeof value === 'string') return value
  return String(value)
}

function isQualityRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatInlineQualityValue(value: unknown, locale: 'zh' | 'en'): string {
  if (Array.isArray(value)) {
    if (value.length === 0) return locale === 'zh' ? '空' : 'empty'
    if (value.length > 8) return locale === 'zh' ? `${value.length} 项` : `${value.length} items`
    return value.map((item) => formatInlineQualityValue(item, locale)).filter(Boolean).join(', ')
  }
  return formatQualityScalar(value, locale)
}

function canFormatInlineQualityValue(value: unknown): boolean {
  if (Array.isArray(value)) return value.every(canFormatInlineQualityValue)
  return !isQualityRecord(value)
}

function formatQualityKey(key: string): string {
  return key.replace(/_/g, ' ')
}

function formatQualityValueSummary(value: unknown, locale: 'zh' | 'en'): string {
  if (Array.isArray(value)) {
    return canFormatInlineQualityValue(value)
      ? formatInlineQualityValue(value, locale)
      : (locale === 'zh' ? `${value.length} 项` : `${value.length} items`)
  }

  if (isQualityRecord(value)) {
    if (Object.keys(value).length === 0) return ''
    const directValue = value['value']
    if (directValue !== undefined) {
      if (isQualityRecord(directValue)) return locale === 'zh' ? '存在' : 'present'
      return formatQualityValueSummary(directValue, locale)
    }

    const width = value['width']
    const height = value['height']
    if (typeof width === 'number' && typeof height === 'number') return `${width}x${height}`

    const entries = Object.entries(value)
      .filter(([, nestedValue]) => !isQualityRecord(nestedValue) && !Array.isArray(nestedValue))
      .slice(0, 3)
    if (entries.length > 0) {
      return entries
        .map(([key, nestedValue]) => `${formatQualityKey(key)}=${formatQualityScalar(nestedValue, locale)}`)
        .join(', ')
    }

    return locale === 'zh' ? '存在' : 'present'
  }

  return formatQualityScalar(value, locale)
}

function isPresenceDetail(detail: string): boolean {
  return /(present|exists|found|missing)$/i.test(detail.trim())
}

function formatQualityCheckDetail(issue: Record<string, unknown>, locale: 'zh' | 'en'): string {
  const detail = formatIssueDetail(issue)
  const valueSummary = formatQualityValueSummary(issue['value'], locale)
  if (!detail) return valueSummary
  if (!valueSummary) return detail
  if (isPresenceDetail(detail)) return valueSummary
  if (detail.toLowerCase().includes(valueSummary.toLowerCase())) return detail
  return detail
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

function formatSeconds(value: number | null | undefined, locale: 'zh' | 'en'): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return locale === 'zh' ? '无数据' : 'No data'
  const absValue = Math.abs(value)
  const formatted = absValue >= 10 ? value.toFixed(2) : value.toFixed(3)
  return `${Number(formatted)}s`
}

function formatSignedSeconds(value: number | null | undefined, locale: 'zh' | 'en'): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return locale === 'zh' ? '无延迟数据' : 'No delay data'
  const formatted = Math.abs(value) >= 10 ? value.toFixed(2) : value.toFixed(3)
  const prefix = value > 0 ? '+' : ''
  return `${prefix}${Number(formatted)}s`
}

function formatTimeWindow(span: AlignmentOverviewSpan, locale: 'zh' | 'en'): string {
  const start = formatSeconds(span.startTime, locale)
  const end = typeof span.endTime === 'number' ? formatSeconds(span.endTime, locale) : (locale === 'zh' ? '未结束' : 'open')
  return `${start}-${end}`
}

function formatSourceTimeWindow(span: AlignmentOverviewSpan, locale: 'zh' | 'en'): string {
  const start = formatSeconds(span.source_start_time, locale)
  const end = typeof span.source_end_time === 'number'
    ? formatSeconds(span.source_end_time, locale)
    : (locale === 'zh' ? '未结束' : 'open')
  return `${start}-${end}`
}

function formatSpanTitle(span: AlignmentOverviewSpan, locale: 'zh' | 'en'): string {
  const title = span.text || span.label || span.category || span.id
  return title ? String(title) : (locale === 'zh' ? '未命名片段' : 'Untitled span')
}

function formatAlignmentMethod(method: string | null | undefined, locale: 'zh' | 'en'): string {
  if (method === 'dtw') return 'DTW'
  if (method === 'scale') return locale === 'zh' ? '时长缩放' : 'Duration scale'
  return locale === 'zh' ? '未记录' : 'Not recorded'
}

function formatSpanSource(source: string | null | undefined, locale: 'zh' | 'en'): string {
  if (source === 'dtw_propagated') return locale === 'zh' ? 'DTW 传播' : 'DTW propagated'
  if (source === 'duration_scaled') return locale === 'zh' ? '时长缩放' : 'Duration scaled'
  if (source === 'user') return locale === 'zh' ? '人工标注' : 'Manual'
  return source || (locale === 'zh' ? '未记录' : 'Not recorded')
}

function alignmentStatusKey(
  status: AlignmentOverviewRow['alignment_status'],
): 'alignmentPropagated' | 'alignmentAnnotated' | 'alignmentNotStarted' {
  if (status === 'propagated') return 'alignmentPropagated'
  if (status === 'annotated') return 'alignmentAnnotated'
  return 'alignmentNotStarted'
}

function collectCounts(values: string[]): Array<{ label: string; count: number }> {
  const counts = new Map<string, number>()
  values.forEach((value) => {
    const normalized = String(value || '').trim()
    if (!normalized) return
    counts.set(normalized, (counts.get(normalized) || 0) + 1)
  })
  return Array.from(counts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count)
}

function scoreBandLabel(score: number): string {
  if (score >= 95) return '95-100'
  if (score >= 85) return '85-94'
  if (score >= 70) return '70-84'
  if (score >= 50) return '50-69'
  return '0-49'
}

function DistributionChart({
  title,
  items,
}: {
  title: string
  items: Array<{ label: string; count: number }>
}) {
  const maxValue = Math.max(...items.map((item) => item.count), 1)

  return (
    <GlassPanel className="quality-chart-card overview-chart-card">
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

function buildExportRows(
  rows: AlignmentOverviewRow[],
  locale: 'zh' | 'en',
  t: (key: 'passed' | 'failed' | 'untitledTask' | 'alignmentPropagated' | 'alignmentAnnotated' | 'alignmentNotStarted') => string,
) {
  return rows.map((row) => ({
    episode_index: row.episode_index,
    record_key: row.record_key,
    task: row.task || '',
    quality_status: row.quality_passed ? t('passed') : t('failed'),
    quality_score: Number(row.quality_score.toFixed(1)),
    failed_validators: row.failed_validators.join(', '),
    issue_types: row.issues
      .map((issue) => {
        const checkName = issue['check_name']
        return typeof checkName === 'string' ? formatIssueLabel(checkName, locale) : ''
      })
      .filter(Boolean)
      .join(', '),
    alignment_status: t(alignmentStatusKey(row.alignment_status)),
    annotation_count: row.annotation_count,
    propagated_count: row.propagated_count,
    prototype_score:
      typeof row.prototype_score === 'number' ? Number(row.prototype_score.toFixed(4)) : '',
    updated_at: row.updated_at || '',
  }))
}

function escapeCsvValue(value: unknown): string {
  const text = String(value ?? '')
  if (text.includes('"') || text.includes(',') || text.includes('\n')) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

function OverviewRowDetailPopover({
  row,
  locale,
}: {
  row: AlignmentOverviewRow
  locale: 'zh' | 'en'
}) {
  const copy = locale === 'zh'
    ? {
      title: '数据纵览',
      quality: '质量验证结果',
      validators: '验证器',
      checks: '检查项',
      dtw: 'DTW 延迟结果',
      semantic: '语义对齐',
      passed: '通过',
      failed: '未通过',
      noChecks: '没有详细检查记录',
      noPropagation: '暂无传播或 DTW 延迟结果',
      noSemantic: '暂无语义标注',
      method: '方法',
      sourceEpisode: '源回合',
      targetWindow: '目标区间',
      sourceWindow: '源区间',
      startDelay: '起点延迟',
      endDelay: '终点延迟',
      durationDelta: '时长差',
      confidence: '置信度',
      status: '状态',
      annotationCount: '标注数',
      propagatedCount: '传播数',
      source: '来源',
    }
    : {
      title: 'Data Overview',
      quality: 'Quality validation result',
      validators: 'Validators',
      checks: 'Checks',
      dtw: 'DTW delay result',
      semantic: 'Semantic alignment',
      passed: 'Passed',
      failed: 'Failed',
      noChecks: 'No detailed check records',
      noPropagation: 'No propagation or DTW delay result',
      noSemantic: 'No semantic annotation',
      method: 'Method',
      sourceEpisode: 'Source episode',
      targetWindow: 'Target window',
      sourceWindow: 'Source window',
      startDelay: 'Start delay',
      endDelay: 'End delay',
      durationDelta: 'Duration delta',
      confidence: 'Confidence',
      status: 'Status',
      annotationCount: 'Annotations',
      propagatedCount: 'Propagated',
      source: 'Source',
    }
  const validatorEntries = Object.entries(row.validator_scores || {})
  const issueGroups = groupQualityIssues(row.issues || [])
  const propagationSpans = row.propagation_spans || []
  const annotationSpans = row.annotation_spans || []
  const semanticSpans = propagationSpans.length > 0 ? propagationSpans : annotationSpans

  return (
    <div className="quality-detail-inspector overview-row-detail-popover" role="status">
      <div className="quality-detail-inspector__head">
        <div>
          <div className="quality-detail-inspector__eyebrow">{copy.title}</div>
          <h4>Episode {row.episode_index}</h4>
        </div>
        <div className="quality-detail-inspector__score">
          <span>{row.quality_score.toFixed(1)}</span>
          <span className={cn(row.quality_passed ? 'is-pass' : 'is-fail')}>
            {row.quality_passed ? copy.passed : copy.failed}
          </span>
        </div>
      </div>

      <div className="overview-detail-sections">
        <section className="overview-detail-section">
          <div className="overview-detail-section__title">{copy.quality}</div>
          {validatorEntries.length > 0 && (
            <div className="overview-detail-lines" aria-label={copy.validators}>
              {validatorEntries.map(([name, score]) => {
                const failed = row.failed_validators.includes(name)
                return (
                  <div key={name} className="overview-detail-line">
                    <strong>{formatValidatorLabel(name, locale)}</strong>
                    <span className={cn(failed ? 'is-fail' : 'is-pass')}>
                      {Number(score.toFixed(1))} · {failed ? copy.failed : copy.passed}
                    </span>
                  </div>
                )
              })}
            </div>
          )}

          {issueGroups.length > 0 ? (
            <div className="overview-detail-check-list" aria-label={copy.checks}>
              {issueGroups.map((group) => (
                <div key={group.validator} className="overview-detail-check-group">
                  <div className="overview-detail-check-group__name">
                    {formatValidatorLabel(group.validator, locale)}
                  </div>
                  {group.checks.map((issue, index) => {
                    const checkName = issue['check_name']
                    const checkKey = typeof checkName === 'string' && checkName.trim()
                      ? checkName
                      : `check-${index}`
                    const passed = issue['passed'] === true
                    const detail = formatQualityCheckDetail(issue, locale)
                    return (
                      <div
                        key={`${group.validator}-${checkKey}-${index}`}
                        className={cn('overview-detail-check', passed ? 'is-pass' : 'is-fail')}
                      >
                        <span>{passed ? '✓' : '×'}</span>
                        <span>{formatCheckLabel(checkKey, locale)}</span>
                        {detail && <em>{detail}</em>}
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          ) : (
            <div className="overview-detail-empty">{copy.noChecks}</div>
          )}
        </section>

        <section className="overview-detail-section">
          <div className="overview-detail-section__title">{copy.dtw}</div>
          {propagationSpans.length > 0 ? (
            <>
              <div className="overview-detail-lines">
                <div className="overview-detail-line">
                  <strong>{copy.method}</strong>
                  <span>{formatAlignmentMethod(row.propagation_alignment_method, locale)}</span>
                </div>
                {row.propagation_source_episode_index !== null
                  && row.propagation_source_episode_index !== undefined && (
                  <div className="overview-detail-line">
                    <strong>{copy.sourceEpisode}</strong>
                    <span>Episode {row.propagation_source_episode_index}</span>
                  </div>
                )}
              </div>
              <div className="overview-detail-span-list">
                {propagationSpans.map((span, index) => (
                  <div key={`${span.id || span.label || 'span'}-${index}`} className="overview-detail-span">
                    <div className="overview-detail-span__title">{formatSpanTitle(span, locale)}</div>
                    <div className="overview-detail-span__meta">
                      <span>{copy.targetWindow}: {formatTimeWindow(span, locale)}</span>
                      <span>{copy.sourceWindow}: {formatSourceTimeWindow(span, locale)}</span>
                      <span>{copy.startDelay}: {formatSignedSeconds(span.dtw_start_delay_s, locale)}</span>
                      <span>{copy.endDelay}: {formatSignedSeconds(span.dtw_end_delay_s, locale)}</span>
                      <span>{copy.durationDelta}: {formatSignedSeconds(span.duration_delta_s, locale)}</span>
                      {typeof span.prototype_score === 'number' && (
                        <span>{copy.confidence}: {Number(span.prototype_score.toFixed(3))}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="overview-detail-empty">{copy.noPropagation}</div>
          )}
        </section>

        <section className="overview-detail-section">
          <div className="overview-detail-section__title">{copy.semantic}</div>
          <div className="overview-detail-lines">
            <div className="overview-detail-line">
              <strong>{copy.status}</strong>
              <span>{locale === 'zh'
                ? (row.alignment_status === 'propagated' ? '已自动传播' : row.alignment_status === 'annotated' ? '已人工标注' : '未开始对齐')
                : (row.alignment_status === 'propagated' ? 'Propagated' : row.alignment_status === 'annotated' ? 'Annotated' : 'Not started')}
              </span>
            </div>
            <div className="overview-detail-line">
              <strong>{copy.annotationCount}</strong>
              <span>{row.annotation_count}</span>
            </div>
            <div className="overview-detail-line">
              <strong>{copy.propagatedCount}</strong>
              <span>{row.propagated_count}</span>
            </div>
          </div>
          {semanticSpans.length > 0 ? (
            <div className="overview-detail-span-list overview-detail-span-list--compact">
              {semanticSpans.map((span, index) => (
                <div key={`${span.id || span.label || 'semantic'}-${index}`} className="overview-detail-span">
                  <div className="overview-detail-span__title">{formatSpanTitle(span, locale)}</div>
                  <div className="overview-detail-span__meta">
                    <span>{copy.targetWindow}: {formatTimeWindow(span, locale)}</span>
                    <span>{copy.source}: {formatSpanSource(span.source, locale)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="overview-detail-empty">{copy.noSemantic}</div>
          )}
        </section>
      </div>
    </div>
  )
}

export default function DataOverviewPage() {
  const { t, locale } = useI18n()
  const {
    selectedDataset,
    datasetInfo,
    alignmentOverview,
    loadAlignmentOverview,
    selectDataset,
  } = useWorkflow()
  const [qualityFilter, setQualityFilter] = useState<'all' | 'passed' | 'failed'>('all')
  const [alignmentFilter, setAlignmentFilter] = useState<
    'all' | 'not_started' | 'annotated' | 'propagated'
  >('all')
  const [chartDimension, setChartDimension] = useState<ChartDimension>('alignment_status')
  const [selectedEpisodeIds, setSelectedEpisodeIds] = useState<number[]>([])
  const [inspectedEpisodeId, setInspectedEpisodeId] = useState<number | null>(null)

  useEffect(() => {
    if (selectedDataset && !datasetInfo) {
      void selectDataset(selectedDataset)
    }
  }, [selectedDataset, datasetInfo, selectDataset])

  useEffect(() => {
    if (selectedDataset) {
      void loadAlignmentOverview()
    }
  }, [selectedDataset, loadAlignmentOverview])

  const rows = alignmentOverview?.rows || []
  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (qualityFilter === 'passed' && !row.quality_passed) return false
      if (qualityFilter === 'failed' && row.quality_passed) return false
      if (alignmentFilter !== 'all' && row.alignment_status !== alignmentFilter) return false
      return true
    })
  }, [rows, qualityFilter, alignmentFilter])

  const visibleSelectedRows = useMemo(
    () => filteredRows.filter((row) => selectedEpisodeIds.includes(row.episode_index)),
    [filteredRows, selectedEpisodeIds],
  )
  const exportSourceRows = visibleSelectedRows.length > 0 ? visibleSelectedRows : filteredRows
  const exportRows = useMemo(
    () => buildExportRows(exportSourceRows, locale, t),
    [exportSourceRows, locale, t],
  )
  const inspectedRow = useMemo(
    () => filteredRows.find((row) => row.episode_index === inspectedEpisodeId) || null,
    [filteredRows, inspectedEpisodeId],
  )

  const chartItems = useMemo(() => {
    if (chartDimension === 'alignment_status') {
      return collectCounts(filteredRows.map((row) => t(alignmentStatusKey(row.alignment_status))))
    }
    if (chartDimension === 'issue_types') {
      return collectCounts(
        filteredRows.flatMap((row) =>
          row.issues.flatMap((issue) => {
            const checkName = issue['check_name']
            return typeof checkName === 'string' ? [formatIssueLabel(checkName, locale)] : []
          }),
        ),
      )
    }
    if (chartDimension === 'failed_validators') {
      return collectCounts(filteredRows.flatMap((row) => row.failed_validators))
    }
    if (chartDimension === 'tasks') {
      return collectCounts(filteredRows.map((row) => row.task || t('untitledTask')))
    }
    return collectCounts(filteredRows.map((row) => scoreBandLabel(row.quality_score)))
  }, [chartDimension, filteredRows, locale, t])

  const chartTitle = useMemo(() => {
    if (chartDimension === 'alignment_status') return t('alignmentStatus')
    if (chartDimension === 'issue_types') return t('issueDistribution')
    if (chartDimension === 'failed_validators') return t('validatorDistribution')
    if (chartDimension === 'tasks') return t('taskDistribution')
    return t('scoreDistribution')
  }, [chartDimension, t])

  const allVisibleSelected =
    filteredRows.length > 0
    && filteredRows.every((row) => selectedEpisodeIds.includes(row.episode_index))

  function toggleEpisodeSelection(episodeIndex: number) {
    setSelectedEpisodeIds((current) =>
      current.includes(episodeIndex)
        ? current.filter((value) => value !== episodeIndex)
        : [...current, episodeIndex],
    )
  }

  function selectFilteredRows() {
    setSelectedEpisodeIds((current) => {
      const next = new Set(current)
      filteredRows.forEach((row) => next.add(row.episode_index))
      return Array.from(next).sort((left, right) => left - right)
    })
  }

  function toggleSelectAllVisible() {
    if (allVisibleSelected) {
      const visibleIds = new Set(filteredRows.map((row) => row.episode_index))
      setSelectedEpisodeIds((current) => current.filter((id) => !visibleIds.has(id)))
      return
    }
    selectFilteredRows()
  }

  function clearInspectedEpisode(episodeIndex: number) {
    setInspectedEpisodeId((current) => (current === episodeIndex ? null : current))
  }

  function clearSelection() {
    setSelectedEpisodeIds([])
  }

  function exportCsv() {
    if (!exportRows.length) return
    const headers = Object.keys(exportRows[0])
    const csv = [
      headers.join(','),
      ...exportRows.map((row) =>
        headers.map((header) => escapeCsvValue(row[header as keyof typeof row])).join(','),
      ),
    ].join('\n')
    const baseName = datasetInfo?.label || selectedDataset || 'pipeline-overview'
    downloadBlob(csv, `${baseName}-pipeline-overview.csv`, 'text/csv;charset=utf-8;')
  }

  function exportJson() {
    if (!exportRows.length) return
    const baseName = datasetInfo?.label || selectedDataset || 'pipeline-overview'
    downloadBlob(
      JSON.stringify(exportRows, null, 2),
      `${baseName}-pipeline-overview.json`,
      'application/json;charset=utf-8;',
    )
  }

  return (
    <div className="page-enter quality-view pipeline-page">
      {selectedDataset && datasetInfo ? (
        <div className="workflow-view__info-bar">
          <span>{datasetInfo.label}</span>
          <span>{datasetInfo.stats.total_episodes} {t('episodes')}</span>
          <span>{datasetInfo.stats.fps} fps</span>
          <span>{datasetInfo.stats.robot_type}</span>
        </div>
      ) : (
        <GlassPanel className="quality-view__empty">{t('noWorkflowDataset')}</GlassPanel>
      )}

      <div className="pipeline-three-column">
        <div className="pipeline-three-column__left">
          <div className="quality-kpis">
            <MetricCard label={t('totalChecked')} value={alignmentOverview?.summary.total_checked ?? '--'} />
            <MetricCard
              label={t('perfectRatio')}
              value={alignmentOverview ? `${alignmentOverview.summary.perfect_ratio.toFixed(1)}%` : '--'}
              accent="sage"
            />
            <MetricCard label={t('selectedRows')} value={selectedEpisodeIds.length} accent="teal" />
            <MetricCard label={t('manualReviewed')} value={alignmentOverview?.summary.annotated_count ?? '--'} accent="coral" />
          </div>

          <GlassPanel className="quality-results-card">
            <div className="quality-results-card__head">
              <div>
                <h3>{t('overviewDashboard')}</h3>
                <p>{t('overviewDashboardDesc')}</p>
              </div>
              <div className="quality-results-card__filters">
                <span className="quality-sidebar__path">{t('chartDimension')}</span>
                <select
                  className="dataset-selector__select"
                  value={chartDimension}
                  onChange={(event) => setChartDimension(event.target.value as ChartDimension)}
                >
                  <option value="alignment_status">{t('alignmentStatus')}</option>
                  <option value="issue_types">{t('issueDistribution')}</option>
                  <option value="score_bands">{t('scoreDistribution')}</option>
                  <option value="failed_validators">{t('validatorDistribution')}</option>
                  <option value="tasks">{t('taskDistribution')}</option>
                </select>
              </div>
            </div>
            <DistributionChart title={chartTitle} items={chartItems.slice(0, 12)} />
          </GlassPanel>
        </div>

        <div className="pipeline-three-column__center">
          <GlassPanel className="quality-results-card">
            <div className="quality-results-card__head">
              <div>
                <h3>{t('dataOverviewTable')}</h3>
                <p>
                  {filteredRows.length} / {rows.length} rows
                </p>
              </div>
              <div className="quality-results-card__filters">
                <select
                  className="dataset-selector__select"
                  value={qualityFilter}
                  onChange={(event) => setQualityFilter(event.target.value as 'all' | 'passed' | 'failed')}
                >
                  <option value="all">{t('allValidated')}</option>
                  <option value="passed">{t('passedEpisodes')}</option>
                  <option value="failed">{t('failedEpisodes')}</option>
                </select>
                <select
                  className="dataset-selector__select"
                  value={alignmentFilter}
                  onChange={(event) =>
                    setAlignmentFilter(
                      event.target.value as 'all' | 'not_started' | 'annotated' | 'propagated',
                    )
                  }
                >
                  <option value="all">{t('allAlignmentStates')}</option>
                  <option value="not_started">{t('alignmentNotStarted')}</option>
                  <option value="annotated">{t('alignmentAnnotated')}</option>
                  <option value="propagated">{t('alignmentPropagated')}</option>
                </select>
                <ActionButton variant="secondary" onClick={selectFilteredRows} disabled={!filteredRows.length}>
                  {t('selectFiltered')}
                </ActionButton>
                <ActionButton variant="secondary" onClick={clearSelection} disabled={!selectedEpisodeIds.length}>
                  {t('clearSelection')}
                </ActionButton>
                <ActionButton variant="secondary" onClick={exportCsv} disabled={!exportRows.length}>
                  {t('exportCsv')}
                </ActionButton>
                <ActionButton variant="secondary" onClick={exportJson} disabled={!exportRows.length}>
                  {t('exportJson')}
                </ActionButton>
              </div>
            </div>

            <div className="quality-table-wrap quality-results-table-wrap">
              <table className="quality-table">
                <thead>
                  <tr>
                    <th className="quality-table__checkbox-cell">
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        onChange={toggleSelectAllVisible}
                        aria-label={t('selectFiltered')}
                      />
                    </th>
                    <th>Episode</th>
                    <th>{t('taskDesc')}</th>
                    <th>{t('qualityValidation')}</th>
                    <th>{t('score')}</th>
                    <th>{t('validators')}</th>
                    <th>{t('textAlignment')}</th>
                    <th>{t('annotation')}</th>
                    <th>{t('runPropagation')}</th>
                    <th>{t('updatedAt')}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row) => {
                    const selected = selectedEpisodeIds.includes(row.episode_index)
                    return (
                      <tr
                        key={row.episode_index}
                        className={cn(
                          'quality-result-row',
                          'overview-result-row',
                          selected && 'quality-table__row--selected',
                          inspectedEpisodeId === row.episode_index && 'is-inspected',
                        )}
                        tabIndex={0}
                        onClick={() => setInspectedEpisodeId(row.episode_index)}
                        onPointerEnter={() => setInspectedEpisodeId(row.episode_index)}
                        onPointerLeave={() => clearInspectedEpisode(row.episode_index)}
                        onMouseEnter={() => setInspectedEpisodeId(row.episode_index)}
                        onMouseLeave={() => clearInspectedEpisode(row.episode_index)}
                        onFocus={() => setInspectedEpisodeId(row.episode_index)}
                        onBlur={(event) => {
                          const nextTarget = event.relatedTarget as Node | null
                          if (!nextTarget || !event.currentTarget.contains(nextTarget)) {
                            clearInspectedEpisode(row.episode_index)
                          }
                        }}
                      >
                        <td className="quality-table__checkbox-cell">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleEpisodeSelection(row.episode_index)}
                            aria-label={`${t('selectedRows')} ${row.episode_index}`}
                          />
                        </td>
                        <td>{row.episode_index}</td>
                        <td>{row.task || t('untitledTask')}</td>
                        <td className={cn(row.quality_passed ? 'is-pass' : 'is-fail')}>
                          {row.quality_passed ? t('passed') : t('failed')}
                        </td>
                        <td>{row.quality_score.toFixed(1)}</td>
                        <td>{row.failed_validators.join(', ') || '-'}</td>
                        <td>{t(alignmentStatusKey(row.alignment_status))}</td>
                        <td>{row.annotation_count}</td>
                        <td>{row.propagated_count}</td>
                        <td>{row.updated_at || '-'}</td>
                      </tr>
                    )
                  })}
                  {filteredRows.length === 0 && (
                    <tr>
                      <td colSpan={10} className="quality-table__empty">No results</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {inspectedRow && <OverviewRowDetailPopover row={inspectedRow} locale={locale} />}
          </GlassPanel>
        </div>

        <GlassPanel className="pipeline-three-column__sidebar quality-layout__sidebar">
          <div className="quality-sidebar__section">
            <h3>{t('exportSummary')}</h3>
            <p>{t('dataOverviewSidebarDesc')}</p>
          </div>

          <div className="quality-sidebar__section">
            <div className="quality-sidebar__path">{t('selectedRows')}: {selectedEpisodeIds.length}</div>
            <div className="quality-sidebar__path">{t('exportRows')}: {exportRows.length}</div>
            <div className="quality-sidebar__path">{t('passedEpisodes')}: {alignmentOverview?.summary.passed_count ?? 0}</div>
            <div className="quality-sidebar__path">{t('failedEpisodes')}: {alignmentOverview?.summary.failed_count ?? 0}</div>
            <div className="quality-sidebar__path">{t('runPropagation')}: {alignmentOverview?.summary.propagated_count ?? 0}</div>
            <div className="quality-sidebar__path">{t('clusters')}: {alignmentOverview?.summary.prototype_cluster_count ?? 0}</div>
          </div>

          <div className="quality-sidebar__section">
            <ActionButton variant="secondary" onClick={exportCsv} disabled={!exportRows.length} className="w-full justify-center">
              {t('exportCsv')}
            </ActionButton>
            <ActionButton variant="secondary" onClick={exportJson} disabled={!exportRows.length} className="w-full justify-center">
              {t('exportJson')}
            </ActionButton>
            <div className="quality-sidebar__path">
              {visibleSelectedRows.length > 0 ? t('exportSelectedHint') : t('exportFilteredHint')}
            </div>
          </div>
        </GlassPanel>
      </div>
    </div>
  )
}
