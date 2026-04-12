import { useEffect, useState } from 'react'
import AnnotationPanel from '../components/curation/AnnotationPanel'
import PrototypePanel from '../components/curation/PrototypePanel'
import { ActionButton, GlassPanel, MetricCard } from '../components/ux'
import { useI18n } from '../controllers/i18n'
import { useWorkflow } from '../controllers/curation'

export default function TextAlignmentView() {
  const { t } = useI18n()
  const {
    selectedDataset,
    datasetInfo,
    qualityResults,
    prototypeResults,
    propagationResults,
    publishTextAnnotationsParquet,
    selectDataset,
    stopPolling,
  } = useWorkflow()
  const [publishing, setPublishing] = useState(false)
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

  return (
    <div className="quality-view">
      <div className="quality-view__hero">
        <div>
          <h2 className="quality-view__title">{t('textAlignmentTitle')}</h2>
          <p className="quality-view__desc">{t('textAlignmentDesc')}</p>
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

      <div className="text-alignment-layout">
        <div className="text-alignment-layout__main">
          <div className="quality-kpis">
            <MetricCard
              label={t('passedEpisodes')}
              value={qualityResults?.passed ?? '--'}
              accent="sage"
            />
            <MetricCard
              label={t('clusters')}
              value={prototypeResults?.cluster_count ?? '--'}
              accent="amber"
            />
            <MetricCard
              label={t('annotation')}
              value={prototypeResults?.anchor_record_keys.length ?? '--'}
              accent="teal"
            />
            <MetricCard
              label={t('runPropagation')}
              value={propagationResults?.target_count ?? '--'}
              accent="coral"
            />
          </div>

          <GlassPanel className="text-alignment-section">
            <div className="text-alignment-section__head">
              <div>
                <h3>{t('prototypeDiscovery')}</h3>
                <p>{t('prototypeDesc')}</p>
              </div>
            </div>
            <PrototypePanel />
          </GlassPanel>

          <AnnotationPanel />
        </div>

        <GlassPanel className="quality-layout__sidebar">
          <div className="quality-sidebar__section">
            <h3>{t('textAlignment')}</h3>
            <p>{t('annotationDesc')}</p>
          </div>

          <div className="quality-sidebar__section">
            <div className="quality-sidebar__path">
              {t('passedEpisodes')}: {qualityResults?.passed ?? 0}
            </div>
            <div className="quality-sidebar__path">
              {t('clusters')}: {prototypeResults?.cluster_count ?? 0}
            </div>
            <div className="quality-sidebar__path">
              {t('textAnnotationsParquet')}: {propagationResults?.published_parquet_path || '-'}
            </div>
          </div>

          <div className="quality-sidebar__section">
            <ActionButton
              type="button"
              variant="secondary"
              disabled={!selectedDataset || publishing}
              onClick={() => void handlePublish()}
              className="w-full justify-center"
            >
              {publishing ? t('publishing') : t('publishTextParquet')}
            </ActionButton>
            {publishMessage && (
              <div className="quality-sidebar__path">{publishMessage}</div>
            )}
            {publishError && (
              <div className="quality-sidebar__error">{publishError}</div>
            )}
          </div>
        </GlassPanel>
      </div>
    </div>
  )
}
