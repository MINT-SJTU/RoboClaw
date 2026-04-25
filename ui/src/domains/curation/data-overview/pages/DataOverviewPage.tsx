import { useEffect, useMemo, useState } from 'react'
import { ActionButton, GlassPanel, MetricCard } from '@/shared/ui'
import { useI18n } from '@/i18n'
import { useWorkflow, type AlignmentOverviewRow } from '@/domains/curation/store/useCurationStore'
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
  }
  const label = labels[checkName]
  return label ? label[locale] : checkName
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
                      <tr key={row.episode_index} className={cn(selected && 'quality-table__row--selected')}>
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
