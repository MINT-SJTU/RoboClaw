import { useEffect, useState } from 'react'
import AnnotationPanel from '@/domains/curation/components/AnnotationPanel'
import PrototypePanel from '@/domains/curation/components/PrototypePanel'
import { ActionButton, GlassPanel } from '@/shared/ui'
import { useI18n } from '@/i18n'
import type { TranslationKey } from '@/i18n/store'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'

function numericSummaryValue(summary: Record<string, unknown> | null | undefined, key: string): number | null {
  const value = summary?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringSummaryValue(summary: Record<string, unknown> | null | undefined, key: string): string | null {
  const value = summary?.[key]
  return typeof value === 'string' && value.trim() ? value : null
}

function clampProgress(value: number | null): number {
  if (value === null) return 0
  return Math.min(Math.max(value, 0), 100)
}

function formatProgress(value: number): string {
  return Number.isInteger(value) ? `${value}%` : `${value.toFixed(1)}%`
}

function phaseLabel(phase: string | null, t: (key: TranslationKey) => string): string {
  if (phase === 'queued') return t('queued')
  if (phase === 'building_canonical') return t('buildingCanonical')
  if (phase === 'building_dtw_graph') return t('buildingDtwGraph')
  if (phase === 'k_medoids') return t('kMedoids')
  if (phase === 'dba_refinement') return t('dbaRefinement')
  if (phase === 'semantic_propagation') return t('semanticPropagation')
  return phase || t('running')
}

export default function TextAlignmentView() {
  const { t } = useI18n()
  const {
    selectedDataset,
    datasetInfo,
    qualityResults,
    prototypeResults,
    propagationResults,
    applyTextAnnotationsToTrainingTasks,
    publishTextAnnotationsParquet,
    loadDatasets,
    datasets,
    datasetsLoading,
    selectDataset,
    loadPrototypeResults,
    loadPropagationResults,
    stopPolling,
    alignmentSourceMode,
    setAlignmentSourceMode,
    alignmentQualityFilter,
    setAlignmentQualityFilter,
    workflowState,
  } = useWorkflow()
  const [publishing, setPublishing] = useState(false)
  const [applyingTasks, setApplyingTasks] = useState(false)
  const [publishMessage, setPublishMessage] = useState('')
  const [publishError, setPublishError] = useState('')

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  useEffect(() => {
    if (!selectedDataset && !datasetsLoading && datasets.length === 0) {
      void loadDatasets()
    }
  }, [selectedDataset, datasetsLoading, datasets.length, loadDatasets])

  useEffect(() => {
    if (!selectedDataset && !datasetsLoading && datasets.length === 1) {
      void selectDataset(datasets[0].id)
    }
  }, [selectedDataset, datasetsLoading, datasets, selectDataset])

  useEffect(() => {
    if (selectedDataset && !datasetInfo) {
      void selectDataset(selectedDataset)
    }
  }, [selectedDataset, datasetInfo, selectDataset])

  useEffect(() => {
    if (!selectedDataset) return
    void loadPrototypeResults()
    void loadPropagationResults()
  }, [selectedDataset, loadPrototypeResults, loadPropagationResults])

  async function handlePublish(): Promise<void> {
    setPublishing(true)
    setPublishMessage('')
    setPublishError('')
    try {
      const result = await publishTextAnnotationsParquet()
      setPublishMessage(`${t('textAnnotationsParquet')}: ${result.path}`)
    } catch (error) {
      setPublishError(error instanceof Error ? error.message : 'Publish failed')
    } finally {
      setPublishing(false)
    }
  }

  async function handleApplyTasks(): Promise<void> {
    setApplyingTasks(true)
    setPublishMessage('')
    setPublishError('')
    try {
      const result = await applyTextAnnotationsToTrainingTasks()
      setPublishMessage(
        `${t('trainingTasksApplied')}: ${result.updated_episode_count} ${t('episodes')} · ${result.manifest_path}`,
      )
    } catch (error) {
      setPublishError(error instanceof Error ? error.message : 'Apply failed')
    } finally {
      setApplyingTasks(false)
    }
  }

  const qualityReady =
    workflowState?.stages.quality_validation.status === 'completed'
    || workflowState?.stages.quality_validation.status === 'paused'
    || Boolean(qualityResults?.episodes.length)
  const validatedEpisodes = qualityResults?.episodes || []
  const filteredCount = validatedEpisodes.filter((episode) => {
    if (alignmentQualityFilter === 'all') return true
    return alignmentQualityFilter === 'passed' ? episode.passed : !episode.passed
  }).length
  const rawEpisodeCount = datasetInfo?.stats.total_episodes ?? 0
  const selectedCount = alignmentSourceMode === 'raw'
    ? rawEpisodeCount
    : qualityReady ? filteredCount : 0
  const prototypeStage = workflowState?.stages.prototype_discovery
  const annotationStage = workflowState?.stages.annotation
  const prototypeSummary = prototypeStage?.summary
  const annotationSummary = annotationStage?.summary
  const prototypeProgress = clampProgress(numericSummaryValue(prototypeSummary, 'progress_percent'))
  const annotationProgress = clampProgress(numericSummaryValue(annotationSummary, 'progress_percent'))
  const prototypePhase = stringSummaryValue(prototypeSummary, 'phase')
  const annotationPhase = stringSummaryValue(annotationSummary, 'phase')
  const prototypePairsCompleted = numericSummaryValue(prototypeSummary, 'distance_pairs_completed')
  const prototypePairsTotal = numericSummaryValue(prototypeSummary, 'distance_pair_count')
  const annotationCompleted = numericSummaryValue(annotationSummary, 'completed')
  const annotationTotal = numericSummaryValue(annotationSummary, 'total')
  const activeProgress =
    prototypeStage?.status === 'running'
      ? {
          title: t('prototypeDiscovery'),
          phase: phaseLabel(prototypePhase, t),
          progress: prototypeProgress,
          detail: prototypePairsTotal !== null
            ? `${prototypePairsCompleted ?? 0}/${prototypePairsTotal}`
            : `${numericSummaryValue(prototypeSummary, 'entry_count') ?? 0}/${selectedCount}`,
          tone: 'prototype',
        }
      : annotationStage?.status === 'running'
        ? {
            title: t('runPropagation'),
            phase: phaseLabel(annotationPhase, t),
            progress: annotationProgress,
            detail: annotationTotal !== null
              ? `${annotationCompleted ?? 0}/${annotationTotal}`
              : `${propagationResults?.target_count ?? 0}`,
            tone: 'annotation',
          }
        : annotationStage?.status === 'completed' && (propagationResults?.target_count ?? 0) > 0
            ? {
                title: t('runPropagation'),
                phase: t('completed'),
                progress: 100,
                detail: `${propagationResults?.target_count ?? 0} ${t('episodes')}`,
                tone: 'complete',
              }
            : prototypeStage?.status === 'completed'
              ? {
                  title: t('prototypeDiscovery'),
                  phase: t('completed'),
                  progress: 100,
                  detail: `${prototypeResults?.cluster_count ?? 0} ${t('clusters')}`,
                  tone: 'complete',
                }
              : null

  return (
    <div className="page-enter quality-view pipeline-page pipeline-compact-shell pipeline-compact-text-page">
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

      <div className="text-alignment-workbench pipeline-compact-text">
        <GlassPanel className="text-alignment-control-card pipeline-toolbar-card">
          <div className="text-alignment-control-card__row">
            <div className="text-alignment-control-card__meta">
              <div className="text-alignment-control-card__title">{t('textAlignmentSource')}</div>
              <div className="text-alignment-control-card__stats">
                <span>{t('rawDataEpisodes')}: {rawEpisodeCount}</span>
                <span>{t('validatedEpisodes')}: {qualityResults?.total ?? 0}</span>
                <span>{t('selectedEpisodes')}: {selectedCount}</span>
                <span>{t('clusters')}: {prototypeResults?.cluster_count ?? 0}</span>
                <span>{t('runPropagation')}: {propagationResults?.target_count ?? 0}</span>
              </div>
            </div>
            <div className="text-alignment-control-card__actions">
              <select
                className="dataset-selector__select"
                disabled={!selectedDataset}
                value={alignmentSourceMode}
                onChange={(event) =>
                  setAlignmentSourceMode(event.target.value as 'quality' | 'raw')
                }
              >
                <option value="quality">{t('fromQualityValidation')}</option>
                <option value="raw">{t('fromRawData')}</option>
              </select>
              <select
                className="dataset-selector__select"
                disabled={alignmentSourceMode === 'raw' || !qualityReady}
                value={alignmentQualityFilter}
                onChange={(event) =>
                  setAlignmentQualityFilter(event.target.value as 'passed' | 'failed' | 'all')
                }
              >
                <option value="passed">{t('passedEpisodes')}</option>
                <option value="failed">{t('failedEpisodes')}</option>
                <option value="all">{t('allValidated')}</option>
              </select>
              <ActionButton
                type="button"
                variant="secondary"
                disabled={!selectedDataset || publishing || applyingTasks}
                onClick={() => void handlePublish()}
                className="justify-center"
              >
                {publishing ? t('publishing') : t('publishTextParquet')}
              </ActionButton>
              <ActionButton
                type="button"
                variant="warning"
                disabled={!selectedDataset || publishing || applyingTasks}
                onClick={() => void handleApplyTasks()}
                className="justify-center"
              >
                {applyingTasks ? t('applying') : t('applyTextToTrainingTasks')}
              </ActionButton>
            </div>
          </div>
          {alignmentSourceMode === 'quality' && !qualityReady ? (
            <div className="status-panel pipeline-inline-status">{t('textAlignmentNeedsQuality')}</div>
          ) : null}
          {alignmentSourceMode === 'raw' ? (
            <div className="status-panel pipeline-inline-status">{t('textAlignmentRawHint')}</div>
          ) : null}
          {activeProgress ? (
            <div className={`text-alignment-progress text-alignment-progress--${activeProgress.tone}`}>
              <div className="text-alignment-progress__head">
                <span>{activeProgress.title}</span>
                <span>
                  {activeProgress.phase} · {formatProgress(activeProgress.progress)}
                  {activeProgress.detail ? ` · ${activeProgress.detail}` : ''}
                </span>
              </div>
              <div
                className="text-alignment-progress__track"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(activeProgress.progress)}
                aria-label={activeProgress.title}
              >
                <div
                  className="text-alignment-progress__bar"
                  style={{ width: `${activeProgress.progress}%` }}
                />
              </div>
            </div>
          ) : null}
          <div className="text-alignment-control-card__footer">
            <div className="quality-sidebar__path">
              {t('textAnnotationsParquet')}: {propagationResults?.published_parquet_path || '-'}
            </div>
            {publishMessage ? <div className="quality-sidebar__path">{publishMessage}</div> : null}
            {publishError ? <div className="quality-sidebar__error">{publishError}</div> : null}
          </div>
          <div className="text-alignment-control-card__prototype">
            <PrototypePanel compact />
          </div>
        </GlassPanel>

        <AnnotationPanel />
      </div>
    </div>
  )
}
