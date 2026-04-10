import { useEffect, useState } from 'react'
import { useDashboard } from '../controllers/dashboard'
import { useI18n } from '../controllers/i18n'

export default function DataView() {
  const store = useDashboard()
  const { datasets, loading, session, policies } = store
  const { t } = useI18n()

  const [trainDataset, setTrainDataset] = useState('')
  const [trainSteps, setTrainSteps] = useState(100000)
  const [trainDevice, setTrainDevice] = useState('cuda')

  useEffect(() => {
    store.loadDatasets()
    store.fetchTrainPolicies()
  }, [])

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('dataCenter')}</h2>
      </div>

      <div className="flex-1 p-6 grid grid-cols-2 gap-6 items-start max-[900px]:grid-cols-1">
        {/* Left: Datasets */}
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('datasets')}</h3>
            <button
              onClick={store.loadDatasets}
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
                key={d.name}
                className="bg-bg border border-bd/30 rounded-lg px-3 py-2.5 flex items-center gap-2 text-sm"
              >
                <span className="flex-1 font-semibold text-tx truncate">{d.name}</span>
                <span className="text-tx3 text-2xs font-mono whitespace-nowrap">
                  {d.total_episodes != null ? `${d.total_episodes} ep` : ''}
                  {d.total_frames != null ? ` · ${d.total_frames} fr` : ''}
                </span>
                <button
                  onClick={() => {
                    if (confirm(`${t('deleteConfirm')} "${d.name}"?`)) store.deleteDataset(d.name)
                  }}
                  className="px-2 py-0.5 text-rd/60 rounded text-xs hover:text-rd hover:bg-rd/10 transition-colors"
                >
                  {t('del')}
                </button>
              </div>
            ))}
          </div>

          {/* Data quality placeholder */}
          <div className="mt-6 pt-4 border-t border-bd/40">
            <div className="bg-bg border border-bd/20 border-dashed rounded-lg p-6 text-center text-sm text-tx3">
              {t('dataQualityPlaceholder')}
            </div>
          </div>
        </section>

        {/* Right: Training + Policies */}
        <div className="space-y-6">
          <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-yl">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('training')}</h3>
            <select
              value={trainDataset}
              onChange={(e) => setTrainDataset(e.target.value)}
              className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm mb-3
                focus:outline-none focus:border-ac"
            >
              <option value="">{t('selectDataset')}</option>
              {datasets.map(d => (
                <option key={d.name} value={d.name}>{d.name}</option>
              ))}
            </select>
            <div className="flex gap-3 mb-3">
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
                {t('steps')}
                <input type="number" value={trainSteps} onChange={(e) => setTrainSteps(Number(e.target.value) || 100000)}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm font-mono focus:outline-none focus:border-ac" />
              </label>
              <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono w-[90px]">
                {t('device')}
                <select value={trainDevice} onChange={(e) => setTrainDevice(e.target.value)}
                  className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac">
                  <option value="cuda">cuda</option>
                  <option value="cpu">cpu</option>
                </select>
              </label>
            </div>
            <button
              disabled={(session.state !== 'idle' && session.state !== 'error') || !trainDataset || !!loading}
              onClick={() => store.doTrainStart({ dataset_name: trainDataset, steps: trainSteps, device: trainDevice })}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-ac hover:bg-ac2 shadow-glow-ac
                transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {loading === 'train' ? t('startingTraining') : t('startTraining')}
            </button>
            {store.trainJobMessage && (
              <div className="mt-3 text-xs text-tx2 font-mono bg-bg rounded-lg p-2.5 break-all">
                {store.trainJobMessage}
              </div>
            )}
          </section>

          {/* Policies */}
          {policies.length > 0 && (
            <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-gn">
              <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('policies') || 'Policies'}</h3>
              <div className="space-y-1.5">
                {policies.map((p: any, i: number) => (
                  <div key={i} className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-sm font-mono text-tx2 truncate">
                    {typeof p === 'string' ? p : p.name || JSON.stringify(p)}
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
