import { useEffect, useState } from 'react'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { useI18n } from '@/i18n'

export default function DatasetsPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const deleteDataset = useDatasetsStore((state) => state.deleteDataset)
  const hubLoading = useHubTransferStore((state) => state.hubLoading)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushDataset = useHubTransferStore((state) => state.pushDataset)
  const pullDataset = useHubTransferStore((state) => state.pullDataset)
  const { t } = useI18n()

  // Hub state
  const [pullDatasetRepo, setPullDatasetRepo] = useState('')

  useEffect(() => {
    void loadDatasets()
  }, [loadDatasets])

  const promptPush = (value: string) => {
    const repoId = prompt(t('enterRepoId'))
    if (!repoId) return
    void pushDataset(value, repoId)
  }

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('datasetsNav')}</h2>
      </div>

      <div className="flex-1 p-6">
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('datasets')}</h3>
            <button
              onClick={() => { void loadDatasets() }}
              className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors"
            >
              {t('refresh')}
            </button>
          </div>

          {datasets.length === 0 && (
            <div className="text-tx3 text-center py-8 text-sm">{t('noDatasets')}</div>
          )}
          <div className="space-y-1.5">
            {datasets.map((d) => (
              <div
                key={d.id}
                className="bg-bg border border-bd/30 rounded-lg px-3 py-2.5 flex items-center gap-2 text-sm"
              >
                <span className="flex-1 font-semibold text-tx truncate">{d.label}</span>
                <span className="text-tx3 text-2xs font-mono whitespace-nowrap">
                  {`${d.stats.total_episodes} ep`}
                  {` · ${d.stats.total_frames} fr`}
                </span>
                <button
                  disabled={!!hubLoading || !d.capabilities.can_push}
                  onClick={() => promptPush(d.id)}
                  className="px-2 py-0.5 text-ac/60 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                >
                  {t('pushToHub')}
                </button>
                <button
                  onClick={() => {
                    if (confirm(`${t('deleteConfirm')} "${d.label}"?`)) {
                      void deleteDataset(d.id)
                    }
                  }}
                  className="px-2 py-0.5 text-rd/60 rounded text-xs hover:text-rd hover:bg-rd/10 transition-colors"
                >
                  {t('del')}
                </button>
              </div>
            ))}
          </div>

          {/* Pull dataset from Hub */}
          <div className="mt-4 pt-4 border-t border-bd/40">
            <h4 className="text-xs font-bold text-tx3 uppercase mb-2">{t('pullFromHub')}</h4>
            <div className="flex gap-2">
              <input
                placeholder={t('repoIdPlaceholder')}
                value={pullDatasetRepo}
                onChange={(e) => setPullDatasetRepo(e.target.value)}
                className="flex-1 bg-bg border border-bd text-tx px-3 py-1.5 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
              <button
                disabled={!pullDatasetRepo || !!hubLoading}
                onClick={() => {
                  void pullDataset(pullDatasetRepo)
                  setPullDatasetRepo('')
                }}
                className="px-3 py-1.5 bg-ac/10 text-ac rounded-lg text-sm font-medium
                  hover:bg-ac/20 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                {hubLoading === 'pullDataset' ? t('downloading') : t('download')}
              </button>
            </div>
          </div>

          {/* Hub progress bar */}
          {hubProgress && !hubProgress.done && hubLoading?.startsWith('pull') && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-2xs text-tx3 mb-1">
                <span>{hubProgress.operation}</span>
                <span>{hubProgress.progress_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-bd/30 rounded-full h-1.5">
                <div
                  className="bg-ac h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(hubProgress.progress_percent, 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Data quality placeholder */}
          <div className="mt-6 pt-4 border-t border-bd/40">
            <div className="bg-bg border border-bd/20 border-dashed rounded-lg p-6 text-center text-sm text-tx3">
              {t('dataQualityPlaceholder')}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
