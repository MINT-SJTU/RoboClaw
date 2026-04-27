import { useEffect, useState } from 'react'
import AnnotationPanel from '@/domains/curation/components/AnnotationPanel'
import PrototypePanel from '@/domains/curation/components/PrototypePanel'
import { ActionButton, GlassPanel } from '@/shared/ui'
import { useI18n } from '@/i18n'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'

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
    selectDataset,
    stopPolling,
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
    if (selectedDataset && !datasetInfo) {
      void selectDataset(selectedDataset)
    }
  }, [selectedDataset, datasetInfo, selectDataset])

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
              <div className="text-alignment-control-card__title">{t('qualityValidation')}</div>
              <div className="text-alignment-control-card__stats">
                <span>{t('validatedEpisodes')}: {qualityResults?.total ?? 0}</span>
                <span>{t('selectedEpisodes')}: {qualityReady ? filteredCount : 0}</span>
                <span>{t('clusters')}: {prototypeResults?.cluster_count ?? 0}</span>
                <span>{t('runPropagation')}: {propagationResults?.target_count ?? 0}</span>
              </div>
            </div>
            <div className="text-alignment-control-card__actions">
              <select
                className="dataset-selector__select"
                disabled={!qualityReady}
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
          {!qualityReady ? (
            <div className="status-panel pipeline-inline-status">{t('textAlignmentNeedsQuality')}</div>
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
