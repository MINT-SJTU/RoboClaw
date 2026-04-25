import { useI18n } from '@/i18n'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'

export default function PrototypePanel() {
  const { t } = useI18n()
  const {
    runPrototypeDiscovery,
    prototypeRunning,
    prototypeResults,
    workflowState,
    alignmentQualityFilter,
  } = useWorkflow()

  const pStage = workflowState?.stages.prototype_discovery
  const qStage = workflowState?.stages.quality_validation
  const isRunning = prototypeRunning || pStage?.status === 'running'
  const qualityDone = qStage?.status === 'completed'

  return (
    <div className="prototype-panel">
      <div className="prototype-panel__topbar">
        <button
          type="button"
          className="prototype-panel__run-btn"
          onClick={() => runPrototypeDiscovery()}
          disabled={isRunning || !qualityDone}
        >
          {isRunning ? t('running') : t('runPrototype')}
        </button>

        {!qualityDone && (
          <p className="prototype-panel__hint">{t('qualityNotDone')}</p>
        )}
      </div>

      {prototypeResults && (
        <div className="prototype-panel__results">
          <div className="prototype-panel__summary">
            <div className="prototype-panel__stat">
              <span className="prototype-panel__stat-label">{t('clusters')}</span>
              <span className="prototype-panel__stat-value">{prototypeResults.cluster_count}</span>
            </div>
            <div className="prototype-panel__stat">
              <span className="prototype-panel__stat-label">{t('candidateEpisodes')}</span>
              <span className="prototype-panel__stat-value">{prototypeResults.candidate_count}</span>
            </div>
            <div className="prototype-panel__summary-item">
              <span className="prototype-panel__stat-label">{t('qualityFilter')}</span>
              <span className="prototype-panel__summary-value">
                {t(
                  alignmentQualityFilter === 'all'
                    ? 'allValidated'
                    : alignmentQualityFilter === 'failed'
                      ? 'failedEpisodes'
                      : 'passedEpisodes',
                )}
              </span>
            </div>
          </div>

          <div className="prototype-panel__clusters">
            {prototypeResults.clusters.map((cluster, idx) => (
              <div key={idx} className="prototype-panel__cluster-card">
                <div className="prototype-panel__cluster-header">
                  Cluster {idx + 1}
                  <span className="prototype-panel__cluster-count">
                    {cluster.member_count} {t('episodes')}
                  </span>
                </div>
                <div className="prototype-panel__cluster-detail">
                  Anchor: {cluster.anchor_record_key}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
