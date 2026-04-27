import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { ActionButton, GlassPanel } from '@/shared/ui'
import { useI18n } from '@/i18n'
import {
  useWorkflow,
  type AlignmentOverviewRow,
  type AlignmentOverviewSpan,
  type AnnotationWorkspacePayload,
  type JointTrajectoryEntry,
  type PropagationResults,
  type PrototypeCluster,
} from '@/domains/curation/store/useCurationStore'
import { cn } from '@/shared/lib/cn'

type ValidatorKey = 'metadata' | 'timing' | 'action' | 'visual' | 'depth' | 'ee_trajectory'
type DelayMetric = 'dtw_start_delay_s' | 'dtw_end_delay_s' | 'duration_delta_s'
type MissingMatrixState = 'pass' | 'fail' | 'supplemented' | null
type OverviewVideoClip = AnnotationWorkspacePayload['videos'][number]
type QualityOverviewPanel = 'timeline' | 'validators' | 'missing'
type EpisodeInspectHandlers = {
  onPreviewEpisode: (episodeIndex: number) => void
  onCommitEpisode: (episodeIndex: number) => void
  onLeaveEpisode: () => void
}

const INSPECT_PREVIEW_DELAY_MS = 260
const INSPECT_CLOSE_DELAY_MS = 220

const VALIDATOR_KEYS: ValidatorKey[] = [
  'metadata',
  'timing',
  'action',
  'visual',
  'depth',
  'ee_trajectory',
]

const MISSING_CHECKS = [
  'info.json',
  'episode identity',
  'parquet_data',
  'videos',
  'task_description',
  'robot_type',
  'fps',
  'features',
] as const

const DELAY_METRICS: Array<{ key: DelayMetric; zh: string; en: string }> = [
  { key: 'dtw_start_delay_s', zh: '起点', en: 'Start' },
  { key: 'dtw_end_delay_s', zh: '终点', en: 'End' },
  { key: 'duration_delta_s', zh: '时长差', en: 'Duration' },
]

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

function isFailingIssue(issue: Record<string, unknown>): boolean {
  return issue['passed'] !== true
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function formatJointValue(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(3) : '-'
}

function getClipStart(videoItem: OverviewVideoClip | null): number {
  return typeof videoItem?.from_timestamp === 'number' ? videoItem.from_timestamp : 0
}

function getClipEnd(videoItem: OverviewVideoClip | null): number | null {
  return typeof videoItem?.to_timestamp === 'number' ? videoItem.to_timestamp : null
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

function relativeTrajectoryTimes(timeValues: number[]): number[] {
  const baseTime = isFiniteNumber(timeValues[0]) ? timeValues[0] : 0
  return timeValues.map((timeValue) =>
    isFiniteNumber(timeValue) ? Math.max(timeValue - baseTime, 0) : 0,
  )
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

function coerceOverviewNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return Number(value.toFixed(4))
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? Number(parsed.toFixed(4)) : null
  }
  return null
}

function subtractOverviewNumber(left: number | null | undefined, right: number | null | undefined): number | null {
  if (typeof left !== 'number' || typeof right !== 'number') return null
  return Number((left - right).toFixed(4))
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function spanTaskText(span: AlignmentOverviewSpan): string {
  return stringOrNull(span.text) || stringOrNull(span.label) || stringOrNull(span.category) || ''
}

function semanticTaskTextFromSpans(spans: AlignmentOverviewSpan[]): string {
  for (const span of spans) {
    const text = spanTaskText(span)
    if (text) return text
  }
  return ''
}

function semanticTaskTextForRow(row: AlignmentOverviewRow): string {
  return (
    stringOrNull(row.semantic_task_text)
    || semanticTaskTextFromSpans(row.propagation_spans || [])
    || semanticTaskTextFromSpans(row.annotation_spans || [])
    || ''
  )
}

function taskInfoForRow(row: AlignmentOverviewRow): { text: string; supplemental: boolean } {
  const task = stringOrNull(row.task) || semanticTaskTextForRow(row)
  const supplemental = Boolean(
    task
    && (row.task_is_supplemental || row.task_source === 'semantic_supplement' || !stringOrNull(row.task)),
  )
  return { text: task, supplemental }
}

function hasSemanticTaskSupplement(row: AlignmentOverviewRow): boolean {
  return taskInfoForRow(row).supplemental
}

function normalizeOverviewSpan(
  span: Record<string, unknown>,
  sourceSpan?: AlignmentOverviewSpan | null,
): AlignmentOverviewSpan {
  const startTime = coerceOverviewNumber(span.startTime)
  const endTime = coerceOverviewNumber(span.endTime)
  const sourceStartTime = sourceSpan ? coerceOverviewNumber(sourceSpan.startTime) : coerceOverviewNumber(span.source_start_time)
  const sourceEndTime = sourceSpan ? coerceOverviewNumber(sourceSpan.endTime) : coerceOverviewNumber(span.source_end_time)
  const durationDelta =
    startTime !== null && endTime !== null && sourceStartTime !== null && sourceEndTime !== null
      ? Number(((endTime - startTime) - (sourceEndTime - sourceStartTime)).toFixed(4))
      : coerceOverviewNumber(span.duration_delta_s)
  return {
    id: stringOrNull(span.id),
    label: stringOrNull(span.label),
    text: stringOrNull(span.text),
    category: stringOrNull(span.category),
    startTime,
    endTime,
    source: stringOrNull(span.source),
    target_record_key: stringOrNull(span.target_record_key),
    prototype_score: coerceOverviewNumber(span.prototype_score),
    source_start_time: sourceStartTime,
    source_end_time: sourceEndTime,
    dtw_start_delay_s: coerceOverviewNumber(span.dtw_start_delay_s) ?? subtractOverviewNumber(startTime, sourceStartTime),
    dtw_end_delay_s: coerceOverviewNumber(span.dtw_end_delay_s) ?? subtractOverviewNumber(endTime, sourceEndTime),
    duration_delta_s: durationDelta,
  }
}

function sourceSpanForFallback(
  span: Record<string, unknown>,
  index: number,
  sourceSpans: AlignmentOverviewSpan[],
): AlignmentOverviewSpan | null {
  const spanId = stringOrNull(span.id)
  if (spanId) {
    const matched = sourceSpans.find((sourceSpan) => String(sourceSpan.id || '') === spanId)
    if (matched) return matched
  }
  return sourceSpans[index] || null
}

function inferAlignmentMethodFromSpans(spans: AlignmentOverviewSpan[]): string {
  if (spans.some((span) => span.source === 'dtw_propagated')) return 'dtw'
  if (spans.some((span) => span.source === 'duration_scaled')) return 'scale'
  return ''
}

function enrichRowTask(row: AlignmentOverviewRow): AlignmentOverviewRow {
  const semanticTask = semanticTaskTextForRow(row)
  const existingTask = stringOrNull(row.task)
  return {
    ...row,
    task: existingTask || semanticTask || '',
    semantic_task_text: semanticTask || row.semantic_task_text || '',
    task_is_supplemental: Boolean(row.task_is_supplemental || (!existingTask && semanticTask)),
    task_source: row.task_source || (existingTask ? 'dataset' : semanticTask ? 'semantic_supplement' : ''),
  }
}

function augmentRowsWithPropagationFallback(
  rows: AlignmentOverviewRow[],
  propagationResults: PropagationResults | null,
  sourceAnnotationSpans: AlignmentOverviewSpan[],
): AlignmentOverviewRow[] {
  if (!propagationResults) return rows.map(enrichRowTask)

  const propagatedByEpisode = new Map(
    propagationResults.propagated.map((item) => [item.episode_index, item]),
  )
  return rows.map((row) => {
    const item = propagatedByEpisode.get(row.episode_index)
    const existingAnnotationSpans = row.annotation_spans || []
    const annotationSpans =
      row.episode_index === propagationResults.source_episode_index && existingAnnotationSpans.length === 0
        ? sourceAnnotationSpans
        : existingAnnotationSpans
    const existingPropagationSpans = row.propagation_spans || []
    const propagationSpans =
      existingPropagationSpans.length === 0 && item?.spans?.length
        ? item.spans.map((span, index) =>
          normalizeOverviewSpan(
            span as Record<string, unknown>,
            sourceSpanForFallback(span as Record<string, unknown>, index, sourceAnnotationSpans),
          ),
        )
        : existingPropagationSpans
    const sourceEpisodeIndex = row.propagation_source_episode_index ?? item?.source_episode_index ?? propagationResults.source_episode_index
    const enriched: AlignmentOverviewRow = {
      ...row,
      annotation_spans: annotationSpans,
      propagation_spans: propagationSpans,
      propagation_source_episode_index: sourceEpisodeIndex,
      propagation_alignment_method:
        row.propagation_alignment_method || item?.alignment_method || inferAlignmentMethodFromSpans(propagationSpans),
      propagated_count: row.propagated_count || propagationSpans.length,
      prototype_score: row.prototype_score ?? item?.prototype_score ?? null,
    }
    return enrichRowTask(enriched)
  })
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

function formatCompactNumber(value: number | string): string {
  if (typeof value === 'string') return value
  if (!Number.isFinite(value)) return '--'
  if (Math.abs(value) >= 1000000) return `${Number((value / 1000000).toFixed(1))}m`
  if (Math.abs(value) >= 1000) return `${Number((value / 1000).toFixed(1))}k`
  return String(value)
}

function formatChartValue(value: number): string {
  if (!Number.isFinite(value)) return '--'
  if (Math.abs(value) >= 10) return value.toFixed(1)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

function getIssueForCheck(
  row: AlignmentOverviewRow,
  checkName: string,
): Record<string, unknown> | undefined {
  return (row.issues || []).find((issue) => issue['check_name'] === checkName)
}

function getIssuePassState(issue: Record<string, unknown> | undefined): boolean | null {
  if (!issue) return null
  if (issue['passed'] === true) return true
  if (issue['passed'] === false) return false
  return null
}

function getMissingMatrixState(row: AlignmentOverviewRow, checkName: string): MissingMatrixState {
  const state = getIssuePassState(getIssueForCheck(row, checkName))
  if (checkName === 'task_description' && state === false && hasSemanticTaskSupplement(row)) {
    return 'supplemented'
  }
  if (state === true) return 'pass'
  if (state === false) return 'fail'
  return null
}

function formatMissingMatrixState(state: MissingMatrixState, locale: 'zh' | 'en'): string {
  if (state === 'pass') return locale === 'zh' ? '通过' : 'passed'
  if (state === 'fail') return locale === 'zh' ? '缺失' : 'missing'
  if (state === 'supplemented') return locale === 'zh' ? '语义补充*' : 'semantic supplement*'
  return locale === 'zh' ? '未记录' : 'not recorded'
}

function rowSemanticSpans(row: AlignmentOverviewRow): AlignmentOverviewSpan[] {
  const propagationSpans = row.propagation_spans || []
  if (propagationSpans.length > 0) return propagationSpans
  return row.annotation_spans || []
}

function spanEnd(span: AlignmentOverviewSpan): number {
  if (typeof span.endTime === 'number' && Number.isFinite(span.endTime)) return span.endTime
  if (typeof span.startTime === 'number' && Number.isFinite(span.startTime)) return span.startTime
  return 0
}

function spanStart(span: AlignmentOverviewSpan): number {
  return typeof span.startTime === 'number' && Number.isFinite(span.startTime) ? span.startTime : 0
}

function maxSpanEnd(rows: AlignmentOverviewRow[]): number {
  return Math.max(
    1,
    ...rows.flatMap((row) => rowSemanticSpans(row).map((span) => spanEnd(span))),
  )
}

function semanticLabel(span: AlignmentOverviewSpan, locale: 'zh' | 'en'): string {
  const label = span.label || span.text || span.category
  return label ? String(label) : (locale === 'zh' ? '未命名' : 'Untitled')
}

function average(values: number[]): number | null {
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function delayValuesForRow(row: AlignmentOverviewRow, metric: DelayMetric): number[] {
  return (row.propagation_spans || [])
    .map((span) => span[metric])
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
}

function averageDelayForRow(row: AlignmentOverviewRow, metric: DelayMetric): number | null {
  return average(delayValuesForRow(row, metric))
}

function collectDelayValues(rows: AlignmentOverviewRow[], metric: DelayMetric): number[] {
  return rows.flatMap((row) => delayValuesForRow(row, metric))
}

function buildHistogram(values: number[], binCount = 8): Array<{ label: string; count: number }> {
  if (values.length === 0) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (min === max) {
    return [{ label: formatChartValue(min), count: values.length }]
  }
  const size = (max - min) / binCount
  const bins = Array.from({ length: binCount }, (_, index) => {
    const from = min + index * size
    const to = index === binCount - 1 ? max : from + size
    return {
      label: `${formatChartValue(from)}-${formatChartValue(to)}`,
      count: 0,
    }
  })
  values.forEach((value) => {
    const index = Math.min(Math.floor((value - min) / size), binCount - 1)
    bins[index].count += 1
  })
  return bins
}

function qualityColor(row: AlignmentOverviewRow): string {
  return row.quality_passed ? '#064e3b' : '#c81e1e'
}

function validatorColor(score: number | undefined, failed: boolean): string {
  if (typeof score !== 'number' || Number.isNaN(score)) return 'rgba(148, 163, 184, 0.34)'
  if (failed) return '#c81e1e'
  const alpha = Math.min(Math.max(score / 100, 0.25), 1)
  return `rgba(6, 78, 59, ${alpha})`
}

function issueMatrixColor(state: MissingMatrixState): string {
  if (state === 'pass') return '#064e3b'
  if (state === 'fail') return '#c81e1e'
  if (state === 'supplemented') return '#1d4ed8'
  return 'rgba(148, 163, 184, 0.38)'
}

function firstClusterEpisode(cluster: PrototypeCluster): number | null {
  const member = cluster.members.find((item) => typeof item.episode_index === 'number')
  return member?.episode_index ?? null
}

function PipelineChartPanel({
  title,
  subtitle,
  className,
  children,
}: {
  title: string
  subtitle?: string
  className?: string
  children: ReactNode
}) {
  return (
    <section className={cn('pipeline-chart-card', className)}>
      <div className="pipeline-chart-card__head">
        <div>
          <h4>{title}</h4>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  )
}

function ChartEmpty({ label }: { label: string }) {
  return <div className="pipeline-chart-empty">{label}</div>
}

function QualityTimelineChart({
  rows,
  emptyLabel,
  inspectHandlers,
}: {
  rows: AlignmentOverviewRow[]
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  const sortedRows = [...rows].sort((left, right) => left.episode_index - right.episode_index)
  if (sortedRows.length === 0) return <ChartEmpty label={emptyLabel} />

  const width = 680
  const height = 220
  const padX = 40
  const padY = 28
  const minEpisode = sortedRows[0].episode_index
  const maxEpisode = sortedRows[sortedRows.length - 1].episode_index
  const xFor = (episodeIndex: number) =>
    maxEpisode === minEpisode
      ? width / 2
      : padX + ((episodeIndex - minEpisode) / (maxEpisode - minEpisode)) * (width - padX * 2)
  const yFor = (score: number) =>
    height - padY - (Math.max(0, Math.min(score, 100)) / 100) * (height - padY * 2)
  const points = sortedRows.map((row) => ({
    row,
    x: xFor(row.episode_index),
    y: yFor(row.quality_score),
  }))

  return (
    <div className="quality-timeline-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Episode quality timeline">
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line
              x1={padX}
              y1={yFor(tick)}
              x2={width - padX}
              y2={yFor(tick)}
              className="pipeline-chart-gridline"
            />
            <text x={8} y={yFor(tick) + 4} className="pipeline-chart-axis-label">{tick}</text>
          </g>
        ))}
        <polyline
          points={points.map((point) => `${point.x},${point.y}`).join(' ')}
          className="quality-timeline-chart__line"
          fill="none"
        />
        {points.map((point) => (
          <g
            key={point.row.episode_index}
            data-overview-inspect-trigger="true"
            role="button"
            tabIndex={0}
            onMouseEnter={() => onPreviewEpisode(point.row.episode_index)}
            onMouseLeave={onLeaveEpisode}
            onFocus={() => onPreviewEpisode(point.row.episode_index)}
            onBlur={onLeaveEpisode}
            onClick={() => onCommitEpisode(point.row.episode_index)}
          >
            <title>{`Episode ${point.row.episode_index}: ${point.row.quality_score.toFixed(1)}`}</title>
            <circle
              cx={point.x}
              cy={point.y}
              r={point.row.quality_passed ? 4.2 : 5.2}
              fill={qualityColor(point.row)}
              className="quality-timeline-chart__point"
            />
          </g>
        ))}
        <text x={padX} y={height - 4} className="pipeline-chart-axis-label">
          {minEpisode}
        </text>
        <text x={width - padX} y={height - 4} textAnchor="end" className="pipeline-chart-axis-label">
          {maxEpisode}
        </text>
      </svg>
    </div>
  )
}

function ValidatorHeatmap({
  rows,
  locale,
  emptyLabel,
  inspectHandlers,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  const sortedRows = [...rows].sort((left, right) => left.episode_index - right.episode_index)
  if (sortedRows.length === 0) return <ChartEmpty label={emptyLabel} />
  const gridStyle: CSSProperties = {
    gridTemplateColumns: `128px repeat(${sortedRows.length}, 28px)`,
  }

  return (
    <div className="validator-heatmap pipeline-matrix-shell">
      <div className="pipeline-matrix-legend">
        <span><i className="is-pass" />{locale === 'zh' ? '通过/高分' : 'Passed / high'}</span>
        <span><i className="is-fail" />{locale === 'zh' ? '失败' : 'Failed'}</span>
        <span><i className="is-missing" />{locale === 'zh' ? '缺失' : 'Missing'}</span>
      </div>
      <div className="pipeline-matrix-scroll">
        <div className="validator-heatmap__grid pipeline-matrix-grid" style={gridStyle}>
        <div className="validator-heatmap__corner">Episode</div>
        {sortedRows.map((row) => (
          <div key={`episode-${row.episode_index}`} className="validator-heatmap__episode">
            {row.episode_index}
          </div>
        ))}
        {VALIDATOR_KEYS.map((validator) => (
          <Fragment key={validator}>
            <div key={`${validator}-label`} className="validator-heatmap__label">
              {formatValidatorLabel(validator, locale)}
            </div>
            {sortedRows.map((row) => {
              const score = row.validator_scores?.[validator]
              const failed = row.failed_validators.includes(validator)
              return (
                <button
                  key={`${validator}-${row.episode_index}`}
                  type="button"
                  data-overview-inspect-trigger="true"
                  className={cn('validator-heatmap__cell', failed && 'is-fail')}
                  style={{ backgroundColor: validatorColor(score, failed) }}
                  title={`Episode ${row.episode_index} · ${validator}: ${
                    typeof score === 'number' ? score.toFixed(1) : 'missing'
                  }`}
                  onMouseEnter={() => onPreviewEpisode(row.episode_index)}
                  onMouseLeave={onLeaveEpisode}
                  onFocus={() => onPreviewEpisode(row.episode_index)}
                  onBlur={onLeaveEpisode}
                  onClick={() => onCommitEpisode(row.episode_index)}
                />
              )
            })}
          </Fragment>
        ))}
        </div>
      </div>
    </div>
  )
}

function MissingMatrix({
  rows,
  locale,
  emptyLabel,
  inspectHandlers,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  const sortedRows = [...rows].sort((left, right) => left.episode_index - right.episode_index)
  if (sortedRows.length === 0) return <ChartEmpty label={emptyLabel} />
  const gridStyle: CSSProperties = {
    gridTemplateColumns: `104px repeat(${MISSING_CHECKS.length}, 116px)`,
  }

  return (
    <div className="missing-matrix pipeline-matrix-shell">
      <div className="pipeline-matrix-legend">
        <span><i className="is-pass" />{locale === 'zh' ? '存在/通过' : 'Present / passed'}</span>
        <span><i className="is-fail" />{locale === 'zh' ? '缺失/失败' : 'Missing / failed'}</span>
        <span><i className="is-supplemented" />{locale === 'zh' ? '语义补充*' : 'Semantic supplement*'}</span>
        <span><i className="is-missing" />{locale === 'zh' ? '未记录' : 'Not recorded'}</span>
      </div>
      <div className="pipeline-matrix-scroll pipeline-matrix-scroll--tall">
        <div className="missing-matrix__grid pipeline-matrix-grid" style={gridStyle}>
        <div className="missing-matrix__corner">Episode</div>
        {MISSING_CHECKS.map((check) => (
          <div key={check} className="missing-matrix__head">{formatCheckLabel(check, locale)}</div>
        ))}
        {sortedRows.map((row) => (
          <Fragment key={row.episode_index}>
            <div key={`${row.episode_index}-episode`} className="missing-matrix__episode">
              {row.episode_index}
            </div>
            {MISSING_CHECKS.map((check) => {
              const state = getMissingMatrixState(row, check)
              return (
                <button
                  key={`${row.episode_index}-${check}`}
                  type="button"
                  data-overview-inspect-trigger="true"
                  className={cn(
                    'missing-matrix__cell',
                    state === 'pass' && 'is-pass',
                    state === 'fail' && 'is-fail',
                    state === 'supplemented' && 'is-supplemented',
                  )}
                  style={{ backgroundColor: issueMatrixColor(state) }}
                  title={`Episode ${row.episode_index} · ${formatCheckLabel(check, locale)}: ${
                    formatMissingMatrixState(state, locale)
                  }`}
                  onMouseEnter={() => onPreviewEpisode(row.episode_index)}
                  onMouseLeave={onLeaveEpisode}
                  onFocus={() => onPreviewEpisode(row.episode_index)}
                  onBlur={onLeaveEpisode}
                  onClick={() => onCommitEpisode(row.episode_index)}
                />
              )
            })}
          </Fragment>
        ))}
        </div>
      </div>
    </div>
  )
}

function DtwDelayHistogram({
  rows,
  locale,
  emptyLabel,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
}) {
  const [metric, setMetric] = useState<DelayMetric>('dtw_start_delay_s')
  const values = collectDelayValues(rows, metric)
  const bins = buildHistogram(values)
  const maxCount = Math.max(...bins.map((bin) => bin.count), 1)

  return (
    <div className="dtw-histogram">
      <div className="pipeline-segmented-control" role="group" aria-label="DTW delay metric">
        {DELAY_METRICS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={cn(metric === item.key && 'is-active')}
            onClick={() => setMetric(item.key)}
          >
            {item[locale]}
          </button>
        ))}
      </div>
      {bins.length === 0 ? (
        <ChartEmpty label={emptyLabel} />
      ) : (
        <div className="dtw-histogram__bars">
          {bins.map((bin) => (
            <div key={bin.label} className="dtw-histogram__row">
              <span>{bin.label}s</span>
              <div className="dtw-histogram__track">
                <div className="dtw-histogram__fill" style={{ width: `${(bin.count / maxCount) * 100}%` }} />
              </div>
              <strong>{bin.count}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function SemanticSpanTimeline({
  rows,
  locale,
  emptyLabel,
  inspectHandlers,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  const rowsWithSpans = [...rows]
    .sort((left, right) => left.episode_index - right.episode_index)
    .filter((row) => rowSemanticSpans(row).length > 0)
  if (rowsWithSpans.length === 0) return <ChartEmpty label={emptyLabel} />
  const maxEnd = maxSpanEnd(rowsWithSpans)

  return (
    <div className="semantic-timeline">
      {rowsWithSpans.map((row) => (
        <div key={row.episode_index} className="semantic-timeline__row">
          <button
            type="button"
            data-overview-inspect-trigger="true"
            className="semantic-timeline__episode"
            onMouseEnter={() => onPreviewEpisode(row.episode_index)}
            onMouseLeave={onLeaveEpisode}
            onFocus={() => onPreviewEpisode(row.episode_index)}
            onBlur={onLeaveEpisode}
            onClick={() => onCommitEpisode(row.episode_index)}
          >
            {row.episode_index}
          </button>
          <div className="semantic-timeline__track">
            {rowSemanticSpans(row).map((span, index) => {
              const start = Math.max(0, spanStart(span))
              const end = Math.max(start + 0.05, spanEnd(span))
              const left = Math.min((start / maxEnd) * 100, 98)
              const width = Math.max(((end - start) / maxEnd) * 100, 3.5)
              return (
                <button
                  key={`${span.id || semanticLabel(span, locale)}-${index}`}
                  type="button"
                  data-overview-inspect-trigger="true"
                  className={cn(
                    'semantic-timeline__span',
                    span.source === 'dtw_propagated' && 'is-propagated',
                  )}
                  style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
                  title={`${semanticLabel(span, locale)} · ${formatTimeWindow(span, locale)}`}
                  onMouseEnter={() => onPreviewEpisode(row.episode_index)}
                  onMouseLeave={onLeaveEpisode}
                  onFocus={() => onPreviewEpisode(row.episode_index)}
                  onBlur={onLeaveEpisode}
                  onClick={() => onCommitEpisode(row.episode_index)}
                >
                  <span>{semanticLabel(span, locale)}</span>
                  <em>{formatTimeWindow(span, locale)}</em>
                </button>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

function SemanticLabelBars({
  rows,
  locale,
  emptyLabel,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
}) {
  const items = Array.from(
    rows
      .flatMap((row) => rowSemanticSpans(row))
      .reduce((counts, span) => {
        const label = semanticLabel(span, locale)
        counts.set(label, (counts.get(label) || 0) + 1)
        return counts
      }, new Map<string, number>()),
  )
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, 12)
  const maxCount = Math.max(...items.map((item) => item.count), 1)

  if (items.length === 0) return <ChartEmpty label={emptyLabel} />
  return (
    <div className="semantic-label-bars">
      {items.map((item) => (
        <div key={item.label} className="semantic-label-bars__row">
          <span>{item.label}</span>
          <div className="semantic-label-bars__track">
            <div style={{ width: `${(item.count / maxCount) * 100}%` }} />
          </div>
          <strong>{item.count}</strong>
        </div>
      ))}
    </div>
  )
}

function CoverageStackedBar({
  rows,
  locale,
  emptyLabel,
}: {
  rows: AlignmentOverviewRow[]
  locale: 'zh' | 'en'
  emptyLabel: string
}) {
  if (rows.length === 0) return <ChartEmpty label={emptyLabel} />
  const labels = locale === 'zh'
    ? { notStarted: '未开始', annotated: '人工标注', propagated: '自动传播' }
    : { notStarted: 'Not started', annotated: 'Manual', propagated: 'Propagated' }
  const counts = rows.reduce(
    (acc, row) => {
      if (row.propagated_count > 0 || row.alignment_status === 'propagated') acc.propagated += 1
      else if (row.annotation_count > 0 || row.alignment_status === 'annotated') acc.annotated += 1
      else acc.notStarted += 1
      return acc
    },
    { notStarted: 0, annotated: 0, propagated: 0 },
  )
  const total = Math.max(rows.length, 1)
  const segments = [
    { key: 'notStarted', label: labels.notStarted, value: counts.notStarted, className: 'is-empty' },
    { key: 'annotated', label: labels.annotated, value: counts.annotated, className: 'is-manual' },
    { key: 'propagated', label: labels.propagated, value: counts.propagated, className: 'is-propagated' },
  ] as const

  return (
    <div className="coverage-bar">
      <div className="coverage-bar__track">
        {segments.map((segment) => (
          <div
            key={segment.key}
            className={cn('coverage-bar__segment', segment.className)}
            style={{ width: `${(segment.value / total) * 100}%` }}
            title={`${segment.label}: ${segment.value}`}
          />
        ))}
      </div>
      <div className="coverage-bar__legend">
        {segments.map((segment) => (
          <div key={segment.key} className={cn('coverage-bar__legend-item', segment.className)}>
            <span />
            <strong>{segment.label}</strong>
            <em>{segment.value}</em>
          </div>
        ))}
      </div>
    </div>
  )
}

function PrototypeClusterChart({
  clusters,
  emptyLabel,
  inspectHandlers,
}: {
  clusters: PrototypeCluster[]
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  if (clusters.length === 0) return <ChartEmpty label={emptyLabel} />
  const maxMembers = Math.max(...clusters.map((cluster) => cluster.member_count), 1)

  return (
    <div className="prototype-cluster-chart">
      {clusters.map((cluster) => {
        const episodeIndex = firstClusterEpisode(cluster)
        return (
          <button
            key={cluster.cluster_index}
            type="button"
            data-overview-inspect-trigger="true"
            className="prototype-cluster-chart__row"
            disabled={episodeIndex === null}
            onMouseEnter={() => {
              if (episodeIndex !== null) onPreviewEpisode(episodeIndex)
            }}
            onMouseLeave={onLeaveEpisode}
            onFocus={() => {
              if (episodeIndex !== null) onPreviewEpisode(episodeIndex)
            }}
            onBlur={onLeaveEpisode}
            onClick={() => {
              if (episodeIndex !== null) onCommitEpisode(episodeIndex)
            }}
          >
            <span className="prototype-cluster-chart__label">C{cluster.cluster_index}</span>
            <span className="prototype-cluster-chart__track">
              <span style={{ width: `${(cluster.member_count / maxMembers) * 100}%` }} />
            </span>
            <span className="prototype-cluster-chart__meta">
              <strong>{cluster.member_count}</strong>
              <em>{cluster.anchor_record_key || cluster.prototype_record_key}</em>
              {typeof cluster.average_distance === 'number' && (
                <em>avg {formatChartValue(cluster.average_distance)}</em>
              )}
              {typeof cluster.anchor_distance_to_barycenter === 'number' && (
                <em>bary {formatChartValue(cluster.anchor_distance_to_barycenter)}</em>
              )}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function QualityDtwScatter({
  rows,
  emptyLabel,
  inspectHandlers,
}: {
  rows: AlignmentOverviewRow[]
  emptyLabel: string
  inspectHandlers: EpisodeInspectHandlers
}) {
  const { onPreviewEpisode, onCommitEpisode, onLeaveEpisode } = inspectHandlers
  const points = rows
    .map((row) => ({ row, delay: averageDelayForRow(row, 'dtw_start_delay_s') }))
    .filter((point): point is { row: AlignmentOverviewRow; delay: number } => point.delay !== null)
  if (points.length === 0) return <ChartEmpty label={emptyLabel} />

  const width = 520
  const height = 220
  const padX = 38
  const padY = 28
  const minDelay = Math.min(...points.map((point) => point.delay), 0)
  const maxDelay = Math.max(...points.map((point) => point.delay), 0)
  const delayRange = maxDelay - minDelay || 1
  const xFor = (score: number) => padX + (Math.max(0, Math.min(score, 100)) / 100) * (width - padX * 2)
  const yFor = (delay: number) =>
    height - padY - ((delay - minDelay) / delayRange) * (height - padY * 2)

  return (
    <div className="quality-dtw-scatter">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Quality score versus DTW delay">
        <line x1={padX} y1={yFor(0)} x2={width - padX} y2={yFor(0)} className="quality-dtw-scatter__zero" />
        {[0, 50, 100].map((tick) => (
          <g key={tick}>
            <line
              x1={xFor(tick)}
              y1={padY}
              x2={xFor(tick)}
              y2={height - padY}
              className="pipeline-chart-gridline"
            />
            <text x={xFor(tick)} y={height - 4} textAnchor="middle" className="pipeline-chart-axis-label">
              {tick}
            </text>
          </g>
        ))}
        <text x={8} y={yFor(maxDelay) + 4} className="pipeline-chart-axis-label">
          {formatChartValue(maxDelay)}s
        </text>
        <text x={8} y={yFor(minDelay) + 4} className="pipeline-chart-axis-label">
          {formatChartValue(minDelay)}s
        </text>
        {points.map((point) => (
          <g
            key={point.row.episode_index}
            data-overview-inspect-trigger="true"
            role="button"
            tabIndex={0}
            onMouseEnter={() => onPreviewEpisode(point.row.episode_index)}
            onMouseLeave={onLeaveEpisode}
            onFocus={() => onPreviewEpisode(point.row.episode_index)}
            onBlur={onLeaveEpisode}
            onClick={() => onCommitEpisode(point.row.episode_index)}
          >
            <title>{`Episode ${point.row.episode_index}: score ${point.row.quality_score.toFixed(1)}, delay ${formatChartValue(point.delay)}s`}</title>
            <circle
              cx={xFor(point.row.quality_score)}
              cy={yFor(point.delay)}
              r={5}
              fill={qualityColor(point.row)}
              className="quality-dtw-scatter__point"
            />
          </g>
        ))}
      </svg>
    </div>
  )
}

function primaryPropagationSpan(row: AlignmentOverviewRow): AlignmentOverviewSpan | null {
  return row.propagation_spans?.[0] || null
}

function TaskDescriptionCell({
  row,
  locale,
  emptyLabel,
}: {
  row: AlignmentOverviewRow
  locale: 'zh' | 'en'
  emptyLabel: string
}) {
  const taskInfo = taskInfoForRow(row)
  if (!taskInfo.text) {
    return <span className="overview-task-cell overview-task-cell--empty">{emptyLabel}</span>
  }
  return (
    <span className={cn('overview-task-cell', taskInfo.supplemental && 'is-supplemented')}>
      <strong>
        {taskInfo.text}
        {taskInfo.supplemental && <sup>*</sup>}
      </strong>
      <em>
        {taskInfo.supplemental
          ? (locale === 'zh' ? '语义对齐补充' : 'Semantic supplement')
          : (locale === 'zh' ? '原始任务字段' : 'Dataset task field')}
      </em>
    </span>
  )
}

function DtwDelayCell({ row, locale }: { row: AlignmentOverviewRow; locale: 'zh' | 'en' }) {
  const startDelay = averageDelayForRow(row, 'dtw_start_delay_s')
  const endDelay = averageDelayForRow(row, 'dtw_end_delay_s')
  const durationDelta = averageDelayForRow(row, 'duration_delta_s')
  if (startDelay === null && endDelay === null && durationDelta === null) {
    return <span className="overview-data-cell overview-data-cell--empty">-</span>
  }
  return (
    <span className="overview-data-cell overview-dtw-cell">
      <strong>{formatSignedSeconds(startDelay, locale)}</strong>
      <em>
        {formatAlignmentMethod(row.propagation_alignment_method, locale)}
        {endDelay !== null && ` · ${locale === 'zh' ? '终点' : 'end'} ${formatSignedSeconds(endDelay, locale)}`}
        {durationDelta !== null && ` · Δ ${formatSignedSeconds(durationDelta, locale)}`}
      </em>
    </span>
  )
}

function SemanticTextCell({ row, locale }: { row: AlignmentOverviewRow; locale: 'zh' | 'en' }) {
  const spans = rowSemanticSpans(row)
  const firstSpan = spans[0]
  const text = semanticTaskTextForRow(row)
  if (!text) return <span className="overview-data-cell overview-data-cell--empty">-</span>
  return (
    <span className="overview-data-cell overview-semantic-cell">
      <strong>{text}</strong>
      <em>
        {firstSpan ? `${formatSpanSource(firstSpan.source, locale)} · ${formatTimeWindow(firstSpan, locale)}` : ''}
        {spans.length > 1 && ` · +${spans.length - 1}`}
      </em>
    </span>
  )
}

function buildExportRows(
  rows: AlignmentOverviewRow[],
  locale: 'zh' | 'en',
  t: (key: 'passed' | 'failed' | 'untitledTask' | 'alignmentPropagated' | 'alignmentAnnotated' | 'alignmentNotStarted') => string,
) {
  return rows.map((row) => {
    const taskInfo = taskInfoForRow(row)
    const propagationSpan = primaryPropagationSpan(row)
    return {
      episode_index: row.episode_index,
      record_key: row.record_key,
      task: taskInfo.text,
      task_source: row.task_source || '',
      task_is_supplemental: taskInfo.supplemental,
      semantic_task_text: semanticTaskTextForRow(row),
      quality_status: row.quality_passed ? t('passed') : t('failed'),
      quality_score: Number(row.quality_score.toFixed(1)),
      failed_validators: row.failed_validators.join(', '),
      issue_types: row.issues
        .filter((issue) => isFailingIssue(issue))
        .map((issue) => {
          const checkName = issue['check_name']
          return typeof checkName === 'string' ? formatIssueLabel(checkName, locale) : ''
        })
        .filter(Boolean)
        .join(', '),
      alignment_status: t(alignmentStatusKey(row.alignment_status)),
      alignment_method: row.propagation_alignment_method || '',
      propagation_source_episode_index: row.propagation_source_episode_index ?? '',
      annotation_count: row.annotation_count,
      propagated_count: row.propagated_count,
      target_start_s: propagationSpan?.startTime ?? '',
      target_end_s: propagationSpan?.endTime ?? '',
      source_start_s: propagationSpan?.source_start_time ?? '',
      source_end_s: propagationSpan?.source_end_time ?? '',
      dtw_start_delay_s: averageDelayForRow(row, 'dtw_start_delay_s') ?? '',
      dtw_end_delay_s: averageDelayForRow(row, 'dtw_end_delay_s') ?? '',
      duration_delta_s: averageDelayForRow(row, 'duration_delta_s') ?? '',
      prototype_score:
        typeof row.prototype_score === 'number' ? Number(row.prototype_score.toFixed(4)) : '',
      updated_at: row.updated_at || '',
    }
  })
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

interface OverviewJointPreviewEntry {
  key: string
  label: string
  actionValues: Array<number | null>
  stateValues: Array<number | null>
  xValues: number[]
}

function buildJointPreviewEntries(
  jointTrajectory: AnnotationWorkspacePayload['joint_trajectory'] | null | undefined,
): OverviewJointPreviewEntry[] {
  const timeValues = jointTrajectory?.time_values || []
  const xValues = relativeTrajectoryTimes(timeValues)
  return (jointTrajectory?.joint_trajectories || [])
    .map((item: JointTrajectoryEntry, index) => {
      const label = item.joint_name || item.state_name || item.action_name || `joint_${index + 1}`
      return {
        key: `${label}-${index}`,
        label,
        actionValues: item.action_values || [],
        stateValues: item.state_values || [],
        xValues,
      }
    })
    .filter((item) =>
      item.xValues.length
      && (
        item.actionValues.some((value) => value !== null && value !== undefined)
        || item.stateValues.some((value) => value !== null && value !== undefined)
      ),
    )
}

function jointSnapshot(entry: OverviewJointPreviewEntry | null, index: number) {
  if (!entry) {
    return { actionValue: null, stateValue: null, deltaValue: null }
  }
  const boundedIndex = Math.min(
    Math.max(index, 0),
    Math.max(entry.actionValues.length, entry.stateValues.length) - 1,
  )
  const actionValue = entry.actionValues[boundedIndex] ?? null
  const stateValue = entry.stateValues[boundedIndex] ?? null
  const deltaValue =
    isFiniteNumber(actionValue) && isFiniteNumber(stateValue)
      ? Number(actionValue) - Number(stateValue)
      : null
  return { actionValue, stateValue, deltaValue }
}

function OverviewEpisodeMediaPreview({
  workspace,
  loading,
  error,
  locale,
}: {
  workspace: AnnotationWorkspacePayload | null
  loading: boolean
  error: string
  locale: 'zh' | 'en'
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [selectedVideoPath, setSelectedVideoPath] = useState('')
  const [playbackState, setPlaybackState] = useState({ time: 0, index: 0, playing: false })
  const [selectedJointKey, setSelectedJointKey] = useState('')
  const copy = locale === 'zh'
    ? {
      title: '实时视频 / 关节角度',
      loading: '正在读取 episode 视频与关节轨迹...',
      failed: '读取 episode 预览失败',
      noWorkspace: '暂无 episode 预览数据',
      noVideo: '当前 episode 没有可播放视频',
      noJoints: '当前 episode 没有关节角度序列',
      stream: '视频流',
      frame: '帧',
      time: '时间',
      sampled: '采样点',
      currentJoint: '当前关节',
      action: 'Action',
      state: 'State',
      delta: '差值',
      playing: '播放中',
      paused: '已暂停',
    }
    : {
      title: 'Live Video / Joint Angles',
      loading: 'Loading episode video and joint trajectory...',
      failed: 'Failed to load episode preview',
      noWorkspace: 'No episode preview data yet',
      noVideo: 'No playable video for this episode',
      noJoints: 'No joint angle series for this episode',
      stream: 'Video stream',
      frame: 'Frame',
      time: 'Time',
      sampled: 'Samples',
      currentJoint: 'Current joint',
      action: 'Action',
      state: 'State',
      delta: 'Delta',
      playing: 'Playing',
      paused: 'Paused',
    }

  const videos = useMemo(() => workspace?.videos || [], [workspace])
  const effectiveVideo = useMemo(() => {
    if (!videos.length) return null
    return videos.find((video) => video.path === selectedVideoPath) || videos[0]
  }, [selectedVideoPath, videos])
  const entries = useMemo(
    () => buildJointPreviewEntries(workspace?.joint_trajectory),
    [workspace?.joint_trajectory],
  )
  const activeEntry =
    entries.find((entry) => entry.key === selectedJointKey)
    || entries[0]
    || null
  const activeSnapshot = jointSnapshot(activeEntry, playbackState.index)
  const frameValues = workspace?.joint_trajectory.frame_values || []
  const currentFrame = frameValues[playbackState.index] ?? playbackState.index

  useEffect(() => {
    setSelectedVideoPath(videos[0]?.path || '')
  }, [workspace?.episode_index, videos])

  useEffect(() => {
    setPlaybackState({ time: 0, index: 0, playing: false })
    setSelectedJointKey('')
  }, [workspace?.episode_index, selectedVideoPath])

  useEffect(() => {
    if (!entries.length) {
      setSelectedJointKey('')
      return
    }
    setSelectedJointKey((current) =>
      entries.some((entry) => entry.key === current) ? current : entries[0].key,
    )
  }, [entries])

  useEffect(() => {
    const playerEl = videoRef.current
    if (!playerEl || !effectiveVideo) return undefined
    const player = playerEl

    let rafId = 0
    const timeValues = workspace?.joint_trajectory.time_values || []
    const trajectoryBaseTime = isFiniteNumber(timeValues[0]) ? timeValues[0] : 0

    function stopPolling(): void {
      if (!rafId) return
      window.cancelAnimationFrame(rafId)
      rafId = 0
    }

    function updateFromVideo(): void {
      const clipStart = getClipStart(effectiveVideo)
      const clipEnd = getClipEnd(effectiveVideo)
      if (isFiniteNumber(clipEnd) && player.currentTime >= clipEnd) {
        player.pause()
      }
      const relativeTime = Math.max(player.currentTime - clipStart, 0)
      const lookupTime = relativeTime + trajectoryBaseTime
      const index = timeValues.length ? findClosestPlaybackIndex(timeValues, lookupTime) : 0
      setPlaybackState({
        time: relativeTime,
        index,
        playing: !player.paused && !player.ended,
      })
    }

    function poll(): void {
      updateFromVideo()
      if (!player.paused && !player.ended) {
        rafId = window.requestAnimationFrame(poll)
      } else {
        rafId = 0
      }
    }

    function handleLoadedMetadata(): void {
      const clipStart = getClipStart(effectiveVideo)
      const boundedStart = Number.isFinite(player.duration)
        ? Math.min(clipStart, player.duration)
        : clipStart
      if (Math.abs(player.currentTime - boundedStart) > 0.1) {
        player.currentTime = boundedStart
      }
      updateFromVideo()
    }

    function handlePlay(): void {
      stopPolling()
      rafId = window.requestAnimationFrame(poll)
    }

    function handlePause(): void {
      stopPolling()
      updateFromVideo()
    }

    player.addEventListener('loadedmetadata', handleLoadedMetadata)
    player.addEventListener('play', handlePlay)
    player.addEventListener('pause', handlePause)
    player.addEventListener('ended', handlePause)
    player.addEventListener('seeking', updateFromVideo)
    player.addEventListener('timeupdate', updateFromVideo)

    if (player.readyState >= 1) {
      handleLoadedMetadata()
    }

    return () => {
      stopPolling()
      player.removeEventListener('loadedmetadata', handleLoadedMetadata)
      player.removeEventListener('play', handlePlay)
      player.removeEventListener('pause', handlePause)
      player.removeEventListener('ended', handlePause)
      player.removeEventListener('seeking', updateFromVideo)
      player.removeEventListener('timeupdate', updateFromVideo)
    }
  }, [effectiveVideo, workspace?.joint_trajectory.time_values])

  if (loading) {
    return (
      <section className="overview-media-preview">
        <div className="overview-detail-section__title">{copy.title}</div>
        <div className="overview-detail-empty">{copy.loading}</div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="overview-media-preview">
        <div className="overview-detail-section__title">{copy.title}</div>
        <div className="overview-detail-empty is-fail">{copy.failed}: {error}</div>
      </section>
    )
  }

  if (!workspace) {
    return (
      <section className="overview-media-preview">
        <div className="overview-detail-section__title">{copy.title}</div>
        <div className="overview-detail-empty">{copy.noWorkspace}</div>
      </section>
    )
  }

  return (
    <section className="overview-media-preview">
      <div className="overview-media-preview__head">
        <div className="overview-detail-section__title">{copy.title}</div>
        <div className="overview-media-preview__status">
          <span>{playbackState.playing ? copy.playing : copy.paused}</span>
          <span>{copy.time} {playbackState.time.toFixed(2)}s</span>
          <span>{copy.frame} {currentFrame}</span>
          <span>{copy.sampled} {workspace.joint_trajectory.sampled_points}/{workspace.joint_trajectory.total_points}</span>
        </div>
      </div>

      <div className="overview-media-preview__grid">
        <div className="overview-media-preview__video-panel">
          {effectiveVideo ? (
            <>
              <video
                key={effectiveVideo.url}
                ref={videoRef}
                src={effectiveVideo.url}
                controls
                playsInline
                preload="metadata"
              />
              {videos.length > 1 && (
                <div className="overview-media-preview__streams" aria-label={copy.stream}>
                  {videos.map((video) => (
                    <button
                      key={video.path}
                      type="button"
                      className={cn(video.path === effectiveVideo.path && 'is-selected')}
                      onClick={() => setSelectedVideoPath(video.path)}
                    >
                      {video.stream || video.path}
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="overview-media-preview__empty">{copy.noVideo}</div>
          )}
        </div>

        <div className="overview-media-preview__joint-panel">
          {activeEntry ? (
            <>
              <div className="overview-media-preview__focus">
                <span>{copy.currentJoint}</span>
                <strong>{activeEntry.label}</strong>
                <em>{copy.action}: {formatJointValue(activeSnapshot.actionValue)}</em>
                <em>{copy.state}: {formatJointValue(activeSnapshot.stateValue)}</em>
                <em>{copy.delta}: {formatJointValue(activeSnapshot.deltaValue)}</em>
              </div>
              <div className="overview-media-preview__joint-list">
                {entries.map((entry) => {
                  const snapshot = jointSnapshot(entry, playbackState.index)
                  return (
                    <button
                      key={entry.key}
                      type="button"
                      className={cn(
                        'overview-media-preview__joint-row',
                        activeEntry.key === entry.key && 'is-selected',
                      )}
                      onClick={() => setSelectedJointKey(entry.key)}
                    >
                      <span>{entry.label}</span>
                      <strong>{formatJointValue(snapshot.actionValue)}</strong>
                      <em>{formatJointValue(snapshot.stateValue)}</em>
                    </button>
                  )
                })}
              </div>
            </>
          ) : (
            <div className="overview-media-preview__empty">{copy.noJoints}</div>
          )}
        </div>
      </div>
    </section>
  )
}

function OverviewRowDetailPopover({
  row,
  locale,
  workspace,
  workspaceLoading,
  workspaceError,
  onClose,
  onInspectorEnter,
  onInspectorLeave,
}: {
  row: AlignmentOverviewRow
  locale: 'zh' | 'en'
  workspace: AnnotationWorkspacePayload | null
  workspaceLoading: boolean
  workspaceError: string
  onClose: () => void
  onInspectorEnter: () => void
  onInspectorLeave: () => void
}) {
  const copy = locale === 'zh'
    ? {
      title: '数据纵览',
      close: '关闭数据纵览',
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
      taskDescription: '任务描述',
      semanticSupplement: '语义对齐补充',
      datasetTask: '原始任务字段',
      status: '状态',
      annotationCount: '标注数',
      propagatedCount: '传播数',
      source: '来源',
    }
    : {
      title: 'Data Overview',
      close: 'Close data overview',
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
      taskDescription: 'Task description',
      semanticSupplement: 'Semantic supplement',
      datasetTask: 'Dataset task field',
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
  const taskInfo = taskInfoForRow(row)

  return (
    <div
      className="quality-detail-inspector overview-row-detail-popover"
      role="status"
      onPointerEnter={onInspectorEnter}
      onPointerLeave={onInspectorLeave}
    >
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
        <button
          type="button"
          className="overview-row-detail-popover__close"
          onClick={onClose}
          aria-label={copy.close}
          title={copy.close}
        >
          ×
        </button>
      </div>

      <div className="overview-detail-sections">
        <OverviewEpisodeMediaPreview
          workspace={workspace}
          loading={workspaceLoading}
          error={workspaceError}
          locale={locale}
        />

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
              <strong>{copy.taskDescription}</strong>
              <span className={cn(taskInfo.supplemental && 'is-supplemented')}>
                {taskInfo.text ? (
                  <>
                    {taskInfo.text}
                    {taskInfo.supplemental ? '*' : ''}
                    {' · '}
                    {taskInfo.supplemental ? copy.semanticSupplement : copy.datasetTask}
                  </>
                ) : (
                  locale === 'zh' ? '未填写任务' : 'Untitled task'
                )}
              </span>
            </div>
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
    propagationResults,
    prototypeResults,
    loadAlignmentOverview,
    loadPropagationResults,
    loadPrototypeResults,
    fetchAnnotationWorkspace,
    selectDataset,
  } = useWorkflow()
  const [qualityFilter, setQualityFilter] = useState<'all' | 'passed' | 'failed'>('all')
  const [alignmentFilter, setAlignmentFilter] = useState<
    'all' | 'not_started' | 'annotated' | 'propagated'
  >('all')
  const [activeQualityPanel, setActiveQualityPanel] = useState<QualityOverviewPanel>('timeline')
  const [selectedEpisodeIds, setSelectedEpisodeIds] = useState<number[]>([])
  const [inspectedEpisodeId, setInspectedEpisodeId] = useState<number | null>(null)
  const [inspectedWorkspace, setInspectedWorkspace] = useState<AnnotationWorkspacePayload | null>(null)
  const [inspectedWorkspaceLoading, setInspectedWorkspaceLoading] = useState(false)
  const [inspectedWorkspaceError, setInspectedWorkspaceError] = useState('')
  const [propagationSourceSpans, setPropagationSourceSpans] = useState<AlignmentOverviewSpan[]>([])
  const inspectOpenTimerRef = useRef<ReturnType<typeof window.setTimeout> | null>(null)
  const inspectCloseTimerRef = useRef<ReturnType<typeof window.setTimeout> | null>(null)

  const clearInspectOpenTimer = useCallback(() => {
    if (inspectOpenTimerRef.current) {
      window.clearTimeout(inspectOpenTimerRef.current)
      inspectOpenTimerRef.current = null
    }
  }, [])

  const clearInspectCloseTimer = useCallback(() => {
    if (inspectCloseTimerRef.current) {
      window.clearTimeout(inspectCloseTimerRef.current)
      inspectCloseTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => {
      clearInspectOpenTimer()
      clearInspectCloseTimer()
    }
  }, [clearInspectOpenTimer, clearInspectCloseTimer])

  const closeInspector = useCallback(() => {
    clearInspectOpenTimer()
    clearInspectCloseTimer()
    setInspectedEpisodeId(null)
  }, [clearInspectOpenTimer, clearInspectCloseTimer])

  const cancelInspectorClose = useCallback(() => {
    clearInspectCloseTimer()
  }, [clearInspectCloseTimer])

  const previewInspectEpisode = useCallback((episodeIndex: number) => {
    clearInspectOpenTimer()
    clearInspectCloseTimer()
    inspectOpenTimerRef.current = window.setTimeout(() => {
      setInspectedEpisodeId(episodeIndex)
      inspectOpenTimerRef.current = null
    }, INSPECT_PREVIEW_DELAY_MS)
  }, [clearInspectOpenTimer, clearInspectCloseTimer])

  const commitInspectEpisode = useCallback((episodeIndex: number) => {
    clearInspectOpenTimer()
    clearInspectCloseTimer()
    setInspectedEpisodeId(episodeIndex)
  }, [clearInspectOpenTimer, clearInspectCloseTimer])

  const scheduleInspectorClose = useCallback(() => {
    clearInspectOpenTimer()
    clearInspectCloseTimer()
    inspectCloseTimerRef.current = window.setTimeout(() => {
      setInspectedEpisodeId(null)
      inspectCloseTimerRef.current = null
    }, INSPECT_CLOSE_DELAY_MS)
  }, [clearInspectOpenTimer, clearInspectCloseTimer])

  const inspectHandlers = useMemo<EpisodeInspectHandlers>(() => ({
    onPreviewEpisode: previewInspectEpisode,
    onCommitEpisode: commitInspectEpisode,
    onLeaveEpisode: scheduleInspectorClose,
  }), [previewInspectEpisode, commitInspectEpisode, scheduleInspectorClose])

  useEffect(() => {
    if (inspectedEpisodeId === null) return undefined

    const handlePointerMove = (event: PointerEvent) => {
      const target = event.target
      if (!(target instanceof Element)) return
      if (
        target.closest('.overview-row-detail-popover')
        || target.closest('[data-overview-inspect-trigger="true"]')
      ) {
        cancelInspectorClose()
        return
      }
      scheduleInspectorClose()
    }

    document.addEventListener('pointermove', handlePointerMove)
    return () => {
      document.removeEventListener('pointermove', handlePointerMove)
    }
  }, [inspectedEpisodeId, cancelInspectorClose, scheduleInspectorClose])

  useEffect(() => {
    if (selectedDataset && !datasetInfo) {
      void selectDataset(selectedDataset)
    }
  }, [selectedDataset, datasetInfo, selectDataset])

  useEffect(() => {
    if (selectedDataset) {
      void loadAlignmentOverview()
      void loadPropagationResults()
      void loadPrototypeResults()
    }
  }, [selectedDataset, loadAlignmentOverview, loadPropagationResults, loadPrototypeResults])

  useEffect(() => {
    const sourceEpisodeIndex = propagationResults?.source_episode_index
    if (!selectedDataset || sourceEpisodeIndex === null || sourceEpisodeIndex === undefined) {
      setPropagationSourceSpans([])
      return
    }
    let cancelled = false
    void fetch(
      `/api/curation/annotations?dataset=${encodeURIComponent(selectedDataset)}&episode_index=${sourceEpisodeIndex}`,
    )
      .then((response) => (response.ok ? response.json() : null))
      .then((payload: { annotations?: Array<Record<string, unknown>> } | null) => {
        if (cancelled) return
        const annotations = Array.isArray(payload?.annotations) ? payload.annotations : []
        setPropagationSourceSpans(annotations.map((span) => normalizeOverviewSpan(span)))
      })
      .catch(() => {
        if (!cancelled) setPropagationSourceSpans([])
      })
    return () => {
      cancelled = true
    }
  }, [selectedDataset, propagationResults?.source_episode_index])

  useEffect(() => {
    if (inspectedEpisodeId === null) {
      setInspectedWorkspace(null)
      setInspectedWorkspaceLoading(false)
      setInspectedWorkspaceError('')
      return undefined
    }

    let cancelled = false
    setInspectedWorkspace(null)
    setInspectedWorkspaceLoading(true)
    setInspectedWorkspaceError('')

    void fetchAnnotationWorkspace(inspectedEpisodeId)
      .then((payload) => {
        if (cancelled) return
        setInspectedWorkspace(payload)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setInspectedWorkspaceError(error instanceof Error ? error.message : 'Failed to load episode workspace')
      })
      .finally(() => {
        if (!cancelled) setInspectedWorkspaceLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [fetchAnnotationWorkspace, inspectedEpisodeId])

  const rawRows = alignmentOverview?.rows || []
  const rows = useMemo(
    () => augmentRowsWithPropagationFallback(rawRows, propagationResults, propagationSourceSpans),
    [rawRows, propagationResults, propagationSourceSpans],
  )
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

  const allVisibleSelected =
    filteredRows.length > 0
    && filteredRows.every((row) => selectedEpisodeIds.includes(row.episode_index))
  const clusterCount = prototypeResults?.cluster_count ?? alignmentOverview?.summary.prototype_cluster_count ?? 0
  const overviewCopy = locale === 'zh'
    ? {
      title: 'Pipeline 分析驾驶舱',
      subtitle: '同屏查看质量、语义、DTW 和原型聚类结果',
      showingRows: '当前结果',
      total: '总数',
      failed: '失败',
      annotated: '已标注',
      propagated: '已传播',
      clusters: '聚类数',
      toolbar: '筛选与导出',
      qualitySection: '质量',
      semanticSection: '语义 / DTW',
      coverageSection: '覆盖 / 原型',
      stackedCharts: '堆叠分析卡片',
      timeline: 'Episode 质量时间线',
      timelineDesc: '连续低分区间会直接浮现',
      validatorHeatmap: 'Validator × Episode 热力图',
      validatorDesc: '红色为失败，深绿为高分通过',
      missingMatrix: '缺失项矩阵',
      missingDesc: '元数据、视频、任务描述等存在性',
      dtwHistogram: 'DTW 延迟分布',
      dtwDesc: '起点、终点和时长差可切换',
      semanticTimeline: '语义片段时间轴',
      semanticTimelineDesc: '每个 episode 的语义 span 区间',
      labelBars: '语义标签分布',
      labelBarsDesc: '优先统计 label，其次 text/category',
      coverage: '人工标注 vs 自动传播',
      coverageDesc: '自动传播优先于人工标注计数',
      prototype: '原型聚类图',
      prototypeDesc: '成员数、anchor 与距离摘要',
      scatter: '质量分数 vs DTW 延迟',
      scatterDesc: '定位低质量且延迟异常的 episode',
      empty: '暂无可绘制数据',
      tableTitle: '结果明细',
      tableDesc: '悬浮或点击 episode 可查看质量、DTW 与语义详情',
      semanticTextColumn: '语义文本',
      dtwColumn: 'DTW 延迟',
      supplementLegend: '* 表示由语义对齐文本补充的任务描述',
      qualityPanelTabs: [
        { key: 'timeline' as const, label: '时间线' },
        { key: 'validators' as const, label: '验证器' },
        { key: 'missing' as const, label: '缺失项' },
      ],
    }
    : {
      title: 'Pipeline Analytics Cockpit',
      subtitle: 'Quality, semantics, DTW, and prototype clusters in one dense view',
      showingRows: 'Current rows',
      total: 'Total',
      failed: 'Failed',
      annotated: 'Annotated',
      propagated: 'Propagated',
      clusters: 'Clusters',
      toolbar: 'Filters and export',
      qualitySection: 'Quality',
      semanticSection: 'Semantic / DTW',
      coverageSection: 'Coverage / Prototype',
      stackedCharts: 'Stacked analytics cards',
      timeline: 'Episode Quality Timeline',
      timelineDesc: 'Consecutive low-score ranges stand out',
      validatorHeatmap: 'Validator × Episode Heatmap',
      validatorDesc: 'Red means failed, darker green means higher passed score',
      missingMatrix: 'Missing Item Matrix',
      missingDesc: 'Metadata, videos, task description, and schema presence',
      dtwHistogram: 'DTW Delay Distribution',
      dtwDesc: 'Switch start, end, and duration delta',
      semanticTimeline: 'Semantic Span Timeline',
      semanticTimelineDesc: 'Semantic span windows per episode',
      labelBars: 'Semantic Label Distribution',
      labelBarsDesc: 'Counts label first, then text/category',
      coverage: 'Manual Annotation vs Propagation',
      coverageDesc: 'Propagation takes priority over manual annotation',
      prototype: 'Prototype Cluster Chart',
      prototypeDesc: 'Member count, anchor, and distance summary',
      scatter: 'Quality Score vs DTW Delay',
      scatterDesc: 'Find low-quality episodes with abnormal delay',
      empty: 'No drawable data yet',
      tableTitle: 'Result Details',
      tableDesc: 'Hover or click an episode to inspect quality, DTW, and semantics',
      semanticTextColumn: 'Semantic Text',
      dtwColumn: 'DTW Delay',
      supplementLegend: '* marks task descriptions supplemented from semantic alignment text',
      qualityPanelTabs: [
        { key: 'timeline' as const, label: 'Timeline' },
        { key: 'validators' as const, label: 'Validators' },
        { key: 'missing' as const, label: 'Missing' },
      ],
    }

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
    <div className="page-enter quality-view pipeline-page pipeline-compact-shell pipeline-data-overview">
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

      <div className="pipeline-page-title">
        <div>
          <h2>{overviewCopy.title}</h2>
          <p>{overviewCopy.subtitle}</p>
        </div>
        <span>{overviewCopy.showingRows}: {filteredRows.length} / {rows.length}</span>
      </div>

      <div className="pipeline-metric-strip">
        <div className="pipeline-mini-metric">
          <span>{overviewCopy.total}</span>
          <strong>{formatCompactNumber(alignmentOverview?.summary.total_checked ?? rows.length)}</strong>
        </div>
        <div className="pipeline-mini-metric is-fail">
          <span>{overviewCopy.failed}</span>
          <strong>{formatCompactNumber(alignmentOverview?.summary.failed_count ?? 0)}</strong>
        </div>
        <div className="pipeline-mini-metric">
          <span>{overviewCopy.annotated}</span>
          <strong>{formatCompactNumber(alignmentOverview?.summary.annotated_count ?? 0)}</strong>
        </div>
        <div className="pipeline-mini-metric">
          <span>{overviewCopy.propagated}</span>
          <strong>{formatCompactNumber(alignmentOverview?.summary.propagated_count ?? 0)}</strong>
        </div>
        <div className="pipeline-mini-metric">
          <span>{overviewCopy.clusters}</span>
          <strong>{formatCompactNumber(clusterCount)}</strong>
        </div>
      </div>

      <div className="pipeline-toolbar" aria-label={overviewCopy.toolbar}>
        <label>
          <span>{t('qualityValidation')}</span>
          <select
            className="dataset-selector__select"
            value={qualityFilter}
            onChange={(event) => setQualityFilter(event.target.value as 'all' | 'passed' | 'failed')}
          >
            <option value="all">{t('allValidated')}</option>
            <option value="passed">{t('passedEpisodes')}</option>
            <option value="failed">{t('failedEpisodes')}</option>
          </select>
        </label>
        <label>
          <span>{t('textAlignment')}</span>
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
        </label>
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
        <div className="pipeline-toolbar__hint">
          {visibleSelectedRows.length > 0 ? t('exportSelectedHint') : t('exportFilteredHint')}
        </div>
      </div>

      <section className="pipeline-stacked-charts" aria-label={overviewCopy.stackedCharts}>
        <div className="pipeline-stack-card">
          <div className="pipeline-stack-card__head">
            <div>
              <h3>{overviewCopy.qualitySection}</h3>
              <p>
                {activeQualityPanel === 'timeline'
                  ? overviewCopy.timelineDesc
                  : activeQualityPanel === 'validators'
                    ? overviewCopy.validatorDesc
                    : overviewCopy.missingDesc}
              </p>
            </div>
            <div className="pipeline-segmented-control" role="tablist" aria-label={overviewCopy.qualitySection}>
              {overviewCopy.qualityPanelTabs.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  role="tab"
                  aria-selected={activeQualityPanel === item.key}
                  className={cn(activeQualityPanel === item.key && 'is-active')}
                  onClick={() => setActiveQualityPanel(item.key)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <div className="pipeline-stack-card__body">
            {activeQualityPanel === 'timeline' && (
              <PipelineChartPanel
                title={overviewCopy.timeline}
                subtitle={overviewCopy.timelineDesc}
                className="pipeline-chart-card--stacked-active"
              >
                <QualityTimelineChart
                  rows={filteredRows}
                  emptyLabel={overviewCopy.empty}
                  inspectHandlers={inspectHandlers}
                />
              </PipelineChartPanel>
            )}
            {activeQualityPanel === 'validators' && (
              <PipelineChartPanel
                title={overviewCopy.validatorHeatmap}
                subtitle={overviewCopy.validatorDesc}
                className="pipeline-chart-card--stacked-active pipeline-chart-card--matrix"
              >
                <ValidatorHeatmap
                  rows={filteredRows}
                  locale={locale}
                  emptyLabel={overviewCopy.empty}
                  inspectHandlers={inspectHandlers}
                />
              </PipelineChartPanel>
            )}
            {activeQualityPanel === 'missing' && (
              <PipelineChartPanel
                title={overviewCopy.missingMatrix}
                subtitle={overviewCopy.missingDesc}
                className="pipeline-chart-card--stacked-active pipeline-chart-card--matrix"
              >
                <MissingMatrix
                  rows={filteredRows}
                  locale={locale}
                  emptyLabel={overviewCopy.empty}
                  inspectHandlers={inspectHandlers}
                />
              </PipelineChartPanel>
            )}
          </div>
        </div>

        <div className="pipeline-stack-card">
          <div className="pipeline-stack-card__head">
            <div>
              <h3>{overviewCopy.semanticSection}</h3>
              <p>{overviewCopy.semanticTimelineDesc}</p>
            </div>
          </div>
          <div className="pipeline-stack-card__grid pipeline-stack-card__grid--semantic">
            <PipelineChartPanel
              title={overviewCopy.dtwHistogram}
              subtitle={overviewCopy.dtwDesc}
              className="pipeline-chart-card--flat"
            >
              <DtwDelayHistogram rows={filteredRows} locale={locale} emptyLabel={overviewCopy.empty} />
            </PipelineChartPanel>
            <PipelineChartPanel
              title={overviewCopy.semanticTimeline}
              subtitle={overviewCopy.semanticTimelineDesc}
              className="pipeline-chart-card--flat pipeline-chart-card--wide"
            >
              <SemanticSpanTimeline
                rows={filteredRows}
                locale={locale}
                emptyLabel={overviewCopy.empty}
                inspectHandlers={inspectHandlers}
              />
            </PipelineChartPanel>
            <PipelineChartPanel
              title={overviewCopy.labelBars}
              subtitle={overviewCopy.labelBarsDesc}
              className="pipeline-chart-card--flat"
            >
              <SemanticLabelBars rows={filteredRows} locale={locale} emptyLabel={overviewCopy.empty} />
            </PipelineChartPanel>
          </div>
        </div>

        <div className="pipeline-stack-card">
          <div className="pipeline-stack-card__head">
            <div>
              <h3>{overviewCopy.coverageSection}</h3>
              <p>{overviewCopy.coverageDesc}</p>
            </div>
          </div>
          <div className="pipeline-stack-card__grid">
            <PipelineChartPanel
              title={overviewCopy.coverage}
              subtitle={overviewCopy.coverageDesc}
              className="pipeline-chart-card--flat"
            >
              <CoverageStackedBar rows={filteredRows} locale={locale} emptyLabel={overviewCopy.empty} />
            </PipelineChartPanel>
            <PipelineChartPanel
              title={overviewCopy.prototype}
              subtitle={overviewCopy.prototypeDesc}
              className="pipeline-chart-card--flat"
            >
              <PrototypeClusterChart
                clusters={prototypeResults?.clusters || []}
                emptyLabel={overviewCopy.empty}
                inspectHandlers={inspectHandlers}
              />
            </PipelineChartPanel>
            <PipelineChartPanel
              title={overviewCopy.scatter}
              subtitle={overviewCopy.scatterDesc}
              className="pipeline-chart-card--flat"
            >
              <QualityDtwScatter
                rows={filteredRows}
                emptyLabel={overviewCopy.empty}
                inspectHandlers={inspectHandlers}
              />
            </PipelineChartPanel>
          </div>
        </div>
      </section>

      <GlassPanel className="quality-results-card pipeline-overview-table-card">
        <div className="quality-results-card__head">
          <div>
            <h3>{overviewCopy.tableTitle}</h3>
            <p>
              {overviewCopy.tableDesc} · {filteredRows.length} / {rows.length} rows
            </p>
          </div>
          <div className="quality-results-card__filters">
            <span className="quality-sidebar__path">{overviewCopy.supplementLegend}</span>
            <span className="quality-sidebar__path">{t('selectedRows')}: {selectedEpisodeIds.length}</span>
            <span className="quality-sidebar__path">{t('exportRows')}: {exportRows.length}</span>
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
                <th>{overviewCopy.semanticTextColumn}</th>
                <th>{overviewCopy.dtwColumn}</th>
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
                    data-overview-inspect-trigger="true"
                    className={cn(
                      'quality-result-row',
                      'overview-result-row',
                      selected && 'quality-table__row--selected',
                      inspectedEpisodeId === row.episode_index && 'is-inspected',
                    )}
                    tabIndex={0}
                    onClick={() => commitInspectEpisode(row.episode_index)}
                    onPointerEnter={() => previewInspectEpisode(row.episode_index)}
                    onPointerLeave={scheduleInspectorClose}
                    onFocus={() => previewInspectEpisode(row.episode_index)}
                    onBlur={scheduleInspectorClose}
                  >
                    <td className="quality-table__checkbox-cell">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleEpisodeSelection(row.episode_index)}
                        onClick={(event) => event.stopPropagation()}
                        aria-label={`${t('selectedRows')} ${row.episode_index}`}
                      />
                    </td>
                    <td>{row.episode_index}</td>
                    <td>
                      <TaskDescriptionCell row={row} locale={locale} emptyLabel={t('untitledTask')} />
                    </td>
                    <td className={cn(row.quality_passed ? 'is-pass' : 'is-fail')}>
                      {row.quality_passed ? t('passed') : t('failed')}
                    </td>
                    <td>{row.quality_score.toFixed(1)}</td>
                    <td>{row.failed_validators.join(', ') || '-'}</td>
                    <td>{t(alignmentStatusKey(row.alignment_status))}</td>
                    <td><SemanticTextCell row={row} locale={locale} /></td>
                    <td><DtwDelayCell row={row} locale={locale} /></td>
                    <td>{row.annotation_count}</td>
                    <td>{row.propagated_count}</td>
                    <td>{row.updated_at || '-'}</td>
                  </tr>
                )
              })}
              {filteredRows.length === 0 && (
                <tr>
                  <td colSpan={12} className="quality-table__empty">No results</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </GlassPanel>
      {inspectedRow && (
        <OverviewRowDetailPopover
          row={inspectedRow}
          locale={locale}
          workspace={inspectedWorkspace}
          workspaceLoading={inspectedWorkspaceLoading}
          workspaceError={inspectedWorkspaceError}
          onClose={closeInspector}
          onInspectorEnter={cancelInspectorClose}
          onInspectorLeave={scheduleInspectorClose}
        />
      )}

    </div>
  )
}
