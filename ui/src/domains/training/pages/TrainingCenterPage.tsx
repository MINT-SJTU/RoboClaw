import { useEffect, useState } from 'react'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { LossCurvePanel } from '@/domains/training/components/LossCurvePanel'
import { TrainingProgressPanel } from '@/domains/training/components/TrainingProgressPanel'
import { useI18n } from '@/i18n'

const POLICY_TYPES = [
  'act',
  'diffusion',
  'groot',
  'multi_task_dit',
  'pi0',
  'pi0_fast',
  'pi05',
  'reward_classifier',
  'sac',
  'sarm',
  'smolvla',
  'tdmpc',
  'vqbet',
  'wall_x',
  'xvla',
]

export default function TrainingCenterPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const session = useSessionStore((state) => state.session)
  const policies = useTrainingStore((state) => state.policies)
  const loadPolicies = useTrainingStore((state) => state.loadPolicies)
  const restoreCurrentTrainJob = useTrainingStore((state) => state.restoreCurrentTrainJob)
  const doTrainStart = useTrainingStore((state) => state.doTrainStart)
  const doTrainStop = useTrainingStore((state) => state.doTrainStop)
  const currentTrainJobId = useTrainingStore((state) => state.currentTrainJobId)
  const trainingLoading = useTrainingStore((state) => state.trainingLoading)
  const trainingStopLoading = useTrainingStore((state) => state.trainingStopLoading)
  const hubLoading = useHubTransferStore((state) => state.hubLoading)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushPolicy = useHubTransferStore((state) => state.pushPolicy)
  const pullPolicy = useHubTransferStore((state) => state.pullPolicy)
  const { t } = useI18n()
  const runtimeDatasets = datasets.filter((dataset) => dataset.capabilities.can_train && dataset.runtime)

  const [trainDataset, setTrainDataset] = useState('')
  const [policyType, setPolicyType] = useState('act')
  const [trainSteps, setTrainSteps] = useState(100000)
  const [trainDevice, setTrainDevice] = useState('cuda')
  const [pullPolicyRepo, setPullPolicyRepo] = useState('')

  useEffect(() => {
    void loadDatasets()
    void loadPolicies()
    void restoreCurrentTrainJob()
  }, [loadDatasets, loadPolicies, restoreCurrentTrainJob])

  const promptPushPolicy = (value: string) => {
    const repoId = prompt(t('enterRepoId'))
    if (!repoId) return
    void pushPolicy(value, repoId)
  }

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('trainingCenter')}</h2>
      </div>

      <div className="flex-1 p-6 grid grid-cols-2 gap-6 items-start max-[1100px]:grid-cols-1">
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-yl">
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-4">{t('training')}</h3>
          <select
            value={trainDataset}
            onChange={(e) => setTrainDataset(e.target.value)}
            className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm mb-3
              focus:outline-none focus:border-ac"
          >
            <option value="">{t('selectDataset')}</option>
            {runtimeDatasets.map(d => (
              <option key={d.id} value={d.runtime!.name}>{d.label}</option>
            ))}
          </select>
          <div className="flex gap-3 mb-3 max-[700px]:flex-col">
            <label className="flex flex-col gap-1 text-2xs text-tx3 font-mono flex-1">
              {t('policyType')}
              <select
                value={policyType}
                onChange={(e) => setPolicyType(e.target.value)}
                className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
              >
                {POLICY_TYPES.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </label>
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
          <div className="flex gap-3 max-[520px]:flex-col">
            <button
              disabled={(session.state !== 'idle' && session.state !== 'error') || !trainDataset || !!trainingLoading}
              onClick={() => {
                void doTrainStart({
                  dataset_name: trainDataset,
                  policy_type: policyType,
                  steps: trainSteps,
                  device: trainDevice,
                })
              }}
              className="flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-ac hover:bg-ac2 shadow-glow-ac
                transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {trainingLoading ? t('startingTraining') : t('startTraining')}
            </button>
            <button
              disabled={!currentTrainJobId || !!trainingStopLoading}
              onClick={() => { void doTrainStop() }}
              className="px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-rd hover:bg-rd/90
                transition-all active:scale-[0.97] disabled:opacity-25 disabled:cursor-not-allowed"
            >
              {trainingStopLoading ? t('stoppingTraining') : t('stopTraining')}
            </button>
          </div>
        </section>

        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-gn">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('policies') || 'Policies'}</h3>
            <button
              onClick={() => { void loadPolicies() }}
              className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors"
            >
              {t('refresh')}
            </button>
          </div>

          {policies.length === 0 && (
            <div className="text-tx3 text-center py-4 text-sm">{t('noPolicies')}</div>
          )}
          <div className="space-y-1.5">
            {policies.map((p: any, i: number) => (
              <div key={i} className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-sm flex items-center gap-2">
                <span className="flex-1 font-mono text-tx2 truncate">
                  {typeof p === 'string' ? p : p.name || JSON.stringify(p)}
                </span>
                <button
                  disabled={!!hubLoading}
                  onClick={() => promptPushPolicy(typeof p === 'string' ? p : p.name)}
                  className="px-2 py-0.5 text-ac/60 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                >
                  {t('pushToHub')}
                </button>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-bd/40">
            <h4 className="text-xs font-bold text-tx3 uppercase mb-2">{t('downloadPolicy')}</h4>
            <div className="flex gap-2">
              <input
                placeholder={t('repoIdPlaceholder')}
                value={pullPolicyRepo}
                onChange={(e) => setPullPolicyRepo(e.target.value)}
                className="flex-1 bg-bg border border-bd text-tx px-3 py-1.5 rounded-lg text-sm
                  focus:outline-none focus:border-ac"
              />
              <button
                disabled={!pullPolicyRepo || !!hubLoading}
                onClick={() => {
                  void pullPolicy(pullPolicyRepo)
                  setPullPolicyRepo('')
                }}
                className="px-3 py-1.5 bg-ac/10 text-ac rounded-lg text-sm font-medium
                  hover:bg-ac/20 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                {hubLoading === 'pullPolicy' ? t('downloading') : t('download')}
              </button>
            </div>
          </div>

          {hubProgress && !hubProgress.done && hubLoading === 'pullPolicy' && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-2xs text-tx3 mb-1">
                <span>{hubProgress.operation}</span>
                <span>{hubProgress.progress_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-bd/30 rounded-full h-1.5">
                <div
                  className="bg-gn h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(hubProgress.progress_percent, 100)}%` }}
                />
              </div>
            </div>
          )}
        </section>

        <LossCurvePanel />
        <TrainingProgressPanel />
      </div>
    </div>
  )
}
