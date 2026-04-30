import { useI18n } from '@/i18n'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'

const VALIDATOR_OPTIONS = [
  { key: 'metadata', labelKey: 'metadata' as const },
  { key: 'timing', labelKey: 'timing' as const },
  { key: 'action', labelKey: 'action' as const },
  { key: 'visual', labelKey: 'visual' as const },
  { key: 'depth', labelKey: 'depth' as const },
]

export default function QualityPanel() {
  const { t } = useI18n()
  const {
    selectedValidators,
    toggleValidator,
    runQualityValidation,
    qualityRunning,
    qualityResults,
    workflowState,
  } = useWorkflow()

  const qStage = workflowState?.stages.quality_validation
  const isRunning = qualityRunning || qStage?.status === 'running'

  return (
    <div className="quality-panel">
      <div className="quality-panel__validators">
        <label className="quality-panel__section-label">{t('validators')}</label>
        <div className="quality-panel__checkboxes">
          {VALIDATOR_OPTIONS.map((opt) => (
            <label key={opt.key} className="quality-panel__checkbox">
              <input
                type="checkbox"
                checked={selectedValidators.includes(opt.key)}
                onChange={() => toggleValidator(opt.key)}
                disabled={isRunning}
              />
              <span>{t(opt.labelKey)}</span>
            </label>
          ))}
        </div>
      </div>

      <button
        type="button"
        className="quality-panel__run-btn"
        onClick={runQualityValidation}
        disabled={isRunning || selectedValidators.length === 0}
      >
        {isRunning ? t('running') : t('runQuality')}
      </button>

      {qualityResults && <QualityResultsTable results={qualityResults} />}
    </div>
  )
}

function QualityResultsTable({ results }: { results: { total: number; passed: number; failed: number; overall_score: number; episodes?: Array<{ episode_index: number; passed: boolean; score: number }> } }) {
  const { t } = useI18n()

  return (
    <div className="quality-panel__results">
      <div className="quality-panel__summary">
        <div className="quality-panel__stat">
          <span className="quality-panel__stat-label">{t('totalEpisodes')}</span>
          <span className="quality-panel__stat-value">{results.total}</span>
        </div>
        <div className="quality-panel__stat">
          <span className="quality-panel__stat-label">{t('passedEpisodes')}</span>
          <span className="quality-panel__stat-value" style={{ color: '#22c55e' }}>{results.passed}</span>
        </div>
        <div className="quality-panel__stat">
          <span className="quality-panel__stat-label">{t('failedEpisodes')}</span>
          <span className="quality-panel__stat-value" style={{ color: '#ef4444' }}>{results.failed}</span>
        </div>
        <div className="quality-panel__stat">
          <span className="quality-panel__stat-label">{t('score')}</span>
          <span className="quality-panel__stat-value">{results.overall_score.toFixed(1)}</span>
        </div>
      </div>

      {results.episodes && results.episodes.length > 0 && (
        <div className="quality-panel__table-wrap">
          <table className="quality-panel__table">
            <thead>
              <tr>
                <th>Episode</th>
                <th>{t('score')}</th>
                <th>{t('passed')}</th>
              </tr>
            </thead>
            <tbody>
              {results.episodes.map((ep) => (
                <tr key={ep.episode_index}>
                  <td>{ep.episode_index}</td>
                  <td>{ep.score.toFixed(1)}</td>
                  <td style={{ color: ep.passed ? '#22c55e' : '#ef4444' }}>
                    {ep.passed ? '\u2713' : '\u2717'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
