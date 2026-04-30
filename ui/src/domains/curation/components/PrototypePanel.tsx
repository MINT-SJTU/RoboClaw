import { useI18n } from '@/i18n'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'
import type {
  PrototypeGroupSummary,
  PrototypeSelectionDiagnostics,
} from '@/domains/curation/store/useCurationStore'

function formatScore(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '0.000'
  return value.toFixed(3)
}

function formatMemberCounts(counts: number[]): string {
  if (!counts.length) return ''
  const visible = counts.slice(0, 6).join('/')
  return counts.length > 6 ? `${visible}/...` : visible
}

function selectDiagnosticRows(diagnostics: PrototypeSelectionDiagnostics) {
  const selectedK = diagnostics.selected_k
  const bestK = diagnostics.best_k
  return diagnostics.evaluated
    .filter((row) => (
      row.selected
      || row.k === bestK
      || Math.abs(row.k - selectedK) <= 2
      || (!row.eligible && row.k > selectedK && row.k <= selectedK + 3)
    ))
    .slice(0, 8)
}

function resolvePrimaryGroup(groups: PrototypeGroupSummary[] | undefined) {
  if (!groups?.length) return null
  return groups.reduce((best, group) => (
    group.entry_count > best.entry_count ? group : best
  ), groups[0])
}

export default function PrototypePanel({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n()
  const {
    runPrototypeDiscovery,
    prototypeRunning,
    prototypeResults,
    workflowState,
    alignmentSourceMode,
  } = useWorkflow()

  const pStage = workflowState?.stages.prototype_discovery
  const qStage = workflowState?.stages.quality_validation
  const isRunning = prototypeRunning || pStage?.status === 'running'
  const qualityDone = qStage?.status === 'completed'
  const requiresQuality = alignmentSourceMode === 'quality'
  const runningSummary = isRunning && pStage?.summary ? pStage.summary : null
  const displayCandidateCount =
    typeof runningSummary?.candidate_count === 'number'
      ? runningSummary.candidate_count
      : prototypeResults?.candidate_count
  const displayEntryCount =
    typeof runningSummary?.entry_count === 'number'
      ? runningSummary.entry_count
      : prototypeResults?.entry_count
  const displayClusterCount =
    typeof runningSummary?.cluster_count === 'number'
      ? runningSummary.cluster_count
      : prototypeResults?.cluster_count
  const displayQualityMode =
    typeof runningSummary?.quality_filter_mode === 'string'
      ? runningSummary.quality_filter_mode
      : prototypeResults?.quality_filter_mode
  const displayPhase = typeof runningSummary?.phase === 'string' ? runningSummary.phase : null
  const displayProgress =
    typeof runningSummary?.progress_percent === 'number'
      ? runningSummary.progress_percent
      : null
  const primaryGroup = resolvePrimaryGroup(prototypeResults?.groups)
  const diagnostics = primaryGroup?.selection_diagnostics || null
  const diagnosticRows = diagnostics ? selectDiagnosticRows(diagnostics) : []
  const distanceBackend =
    typeof runningSummary?.distance_backend === 'string'
      ? runningSummary.distance_backend
      : prototypeResults?.distance_backend
  const distancePairCount =
    typeof runningSummary?.distance_pair_count === 'number'
      ? runningSummary.distance_pair_count
      : prototypeResults?.distance_pair_count

  return (
    <div className={compact ? 'prototype-panel prototype-panel--compact' : 'prototype-panel'}>
      <div className="prototype-panel__topbar">
        <button
          type="button"
          className="prototype-panel__run-btn"
          onClick={() => runPrototypeDiscovery()}
          disabled={isRunning || (requiresQuality && !qualityDone)}
        >
          {isRunning ? t('running') : t('runPrototype')}
        </button>

        {requiresQuality && !qualityDone && (
          <p className="prototype-panel__hint">{t('qualityNotDone')}</p>
        )}
      </div>

      {(prototypeResults || runningSummary) && (
        <div className="prototype-panel__results">
          <div className="prototype-panel__summary">
            <div className="prototype-panel__stat">
              <span className="prototype-panel__stat-label">{t('clusters')}</span>
              <span className="prototype-panel__stat-value">{displayClusterCount ?? 0}</span>
            </div>
            <div className="prototype-panel__stat">
              <span className="prototype-panel__stat-label">{t('candidateEpisodes')}</span>
              <span className="prototype-panel__stat-value">{displayCandidateCount ?? 0}</span>
            </div>
            {isRunning && (
              <div className="prototype-panel__stat">
                <span className="prototype-panel__stat-label">{displayPhase || t('running')}</span>
                <span className="prototype-panel__stat-value">
                  {displayProgress === null ? displayEntryCount ?? 0 : `${displayProgress}%`}
                </span>
              </div>
            )}
            <div className="prototype-panel__summary-item">
              <span className="prototype-panel__stat-label">{t('qualityFilter')}</span>
              <span className="prototype-panel__summary-value">
                {t(
                  displayQualityMode === 'all'
                    ? 'allValidated'
                    : displayQualityMode === 'raw'
                      ? 'rawDataEpisodes'
                      : displayQualityMode === 'failed'
                        ? 'failedEpisodes'
                        : 'passedEpisodes',
                )}
              </span>
            </div>
            {distanceBackend && (
              <div className="prototype-panel__summary-item">
                <span className="prototype-panel__stat-label">{t('dtwBackend')}</span>
                <span className="prototype-panel__summary-value">
                  {distanceBackend.toUpperCase()}
                  {typeof distancePairCount === 'number' ? ` · ${distancePairCount}` : ''}
                </span>
              </div>
            )}
          </div>

          {prototypeResults && (
            <>
              {diagnostics && (
                <div className="prototype-panel__diagnostics">
                  <div className="prototype-panel__diagnostics-head">
                    <span>{t('autoKSelection')}</span>
                    <span>
                      k={diagnostics.selected_k} · {t('bestK')} {diagnostics.best_k} · {t('score')} {formatScore(diagnostics.selected_score)}
                    </span>
                  </div>
                  <div className="prototype-panel__diagnostics-meta">
                    <span>{t('evaluatedK')}: {diagnostics.evaluated_count}</span>
                    <span>{t('tolerance')}: {formatScore(diagnostics.tolerance)}</span>
                    <span>{t('singletonFiltered')}: {diagnostics.rejected_singleton_heavy_count}</span>
                  </div>
                  <div className="prototype-panel__diagnostics-table" role="table">
                    {diagnosticRows.map((row) => (
                      <div
                        key={row.k}
                        className={
                          row.selected
                            ? 'prototype-panel__diagnostic-row prototype-panel__diagnostic-row--selected'
                            : 'prototype-panel__diagnostic-row'
                        }
                        role="row"
                      >
                        <span>k={row.k}</span>
                        <span>{formatScore(row.score)}</span>
                        <span>{formatMemberCounts(row.member_counts)}</span>
                        <span>{row.eligible ? t('eligible') : t('filtered')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

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
            </>
          )}
        </div>
      )}
    </div>
  )
}
