import { useEffect, useState } from 'react'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { useI18n } from '@/i18n'

function formatBytes(value?: number) {
  const bytes = value ?? 0
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let current = bytes / 1024
  let index = 0
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024
    index += 1
  }
  return `${current.toFixed(current >= 10 ? 1 : 2)} ${units[index]}`
}

export default function DatasetsPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const storageUsage = useDatasetsStore((state) => state.storageUsage)
  const loadingAction = useDatasetsStore((state) => state.loadingAction)
  const statusMessage = useDatasetsStore((state) => state.statusMessage)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const loadStorageUsage = useDatasetsStore((state) => state.loadStorageUsage)
  const ingestDataset = useDatasetsStore((state) => state.ingestDataset)
  const requestPublish = useDatasetsStore((state) => state.requestPublish)
  const redeemAccess = useDatasetsStore((state) => state.redeemAccess)
  const deleteDataset = useDatasetsStore((state) => state.deleteDataset)
  const hubLoading = useHubTransferStore((state) => state.hubLoading)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushDataset = useHubTransferStore((state) => state.pushDataset)
  const pullDataset = useHubTransferStore((state) => state.pullDataset)
  const { t } = useI18n()

  // Hub state
  const [pullDatasetRepo, setPullDatasetRepo] = useState('')
  const [username, setUsername] = useState(() => localStorage.getItem('roboclaw.dataset.username') || 'pearl')
  const [datasetId, setDatasetId] = useState('')
  const [sourceUri, setSourceUri] = useState('')
  const [sourceKind, setSourceKind] = useState('remote_dataset')

  useEffect(() => {
    void loadDatasets()
  }, [loadDatasets])

  useEffect(() => {
    localStorage.setItem('roboclaw.dataset.username', username)
    void loadStorageUsage(username)
  }, [loadStorageUsage, username])

  const promptPush = (value: string) => {
    const repoId = prompt(t('enterRepoId'))
    if (!repoId) return
    void pushDataset(value, repoId)
  }

  const resolvedSourceKind = sourceUri.includes('huggingface.co') || sourceUri.startsWith('hf://')
    ? 'remote_dataset'
    : sourceKind

  return (
    <div className="page-enter flex flex-col h-full overflow-y-auto">
      <div className="border-b border-bd/50 px-6 py-4 bg-sf">
        <h2 className="text-xl font-bold tracking-tight">{t('datasetsNav')}</h2>
      </div>

      <div className="flex-1 p-6">
        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac mb-5">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-[180px] flex-1">
              <label className="block text-xs font-semibold text-tx3 mb-1">当前用户</label>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="w-full bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
              />
            </div>
            <button
              onClick={() => { void loadStorageUsage(username) }}
              className="px-3 py-2 bg-ac/10 text-ac rounded-lg text-sm font-medium hover:bg-ac/20 transition-colors"
            >
              刷新用量
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">我的数据集</div>
              <div className="text-lg font-bold text-tx mt-1">{storageUsage?.datasetCount ?? 0}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">已用空间</div>
              <div className="text-lg font-bold text-tx mt-1">{formatBytes(storageUsage?.usedBytes)}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">private</div>
              <div className="text-lg font-bold text-tx mt-1">{formatBytes(storageUsage?.privateBytes)}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">public</div>
              <div className="text-lg font-bold text-tx mt-1">{formatBytes(storageUsage?.publicBytes)}</div>
            </div>
          </div>

          {statusMessage && (
            <div className="mt-3 text-sm text-ac bg-ac/10 border border-ac/20 rounded-lg px-3 py-2">
              {statusMessage}
            </div>
          )}
        </section>

        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac mb-5">
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-3">接入数据</h3>
          <div className="grid grid-cols-1 md:grid-cols-[160px_1fr_1fr_auto] gap-2">
            <select
              value={sourceKind}
              onChange={(event) => setSourceKind(event.target.value)}
              className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
            >
              <option value="remote_dataset">HuggingFace / 远程公开数据</option>
              <option value="mounted_path">服务器挂载目录</option>
              <option value="local_archive">服务器本地压缩包</option>
              <option value="cloud_object">云对象地址</option>
            </select>
            <input
              placeholder="数据集 ID，例如 gr00t-libero"
              value={datasetId}
              onChange={(event) => setDatasetId(event.target.value)}
              className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
            />
            <input
              placeholder="来源地址，例如 hf://nvidia/GR00T-N1.7-LIBERO"
              value={sourceUri}
              onChange={(event) => setSourceUri(event.target.value)}
              className="bg-bg border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
            />
            <button
              disabled={!datasetId || !sourceUri || loadingAction === 'ingest'}
              onClick={() => {
                void ingestDataset({
                  dataset_id: datasetId,
                  username,
                  source_kind: resolvedSourceKind,
                  source_uri: sourceUri,
                }).then(() => {
                  setDatasetId('')
                  setSourceUri('')
                  void loadStorageUsage(username)
                })
              }}
              className="px-3 py-2 bg-ac text-white rounded-lg text-sm font-semibold hover:bg-ac/90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loadingAction === 'ingest' ? '接入中' : '接入为 private'}
            </button>
          </div>
          {resolvedSourceKind !== sourceKind && (
            <p className="text-xs text-ac mt-2">
              已识别为 HuggingFace 数据集，将按“远程公开数据”接入。
            </p>
          )}
          <p className="text-xs text-tx3 mt-2">
            private 数据可直接用于自己的训练；只有主动申请公开并通过质检后，才会进入共享数据池并产生积分。
          </p>
        </section>

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
                className="bg-bg border border-bd/30 rounded-lg px-3 py-2.5 flex flex-wrap items-center gap-2 text-sm"
              >
                <span className="flex-1 font-semibold text-tx truncate">{d.label}</span>
                <span className="text-tx3 text-2xs font-mono whitespace-nowrap">
                  {`${d.stats.total_episodes} ep`}
                  {` · ${d.stats.total_frames} fr`}
                  {` · ${formatBytes(d.stats.total_bytes)}`}
                </span>
                <button
                  disabled={loadingAction === `publish:${d.id}`}
                  onClick={() => {
                    void requestPublish(d.id, username).then(() => {
                      void loadStorageUsage(username)
                    })
                  }}
                  className="px-2 py-0.5 text-gn/70 rounded text-xs hover:text-gn hover:bg-gn/10 transition-colors disabled:opacity-25"
                >
                  {loadingAction === `publish:${d.id}` ? '质检中' : '申请公开并质检'}
                </button>
                <button
                  disabled={loadingAction === `redeem:${d.id}`}
                  onClick={() => {
                    void redeemAccess(d.id, username).then(() => {
                      void loadStorageUsage(username)
                    })
                  }}
                  className="px-2 py-0.5 text-ac/70 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                >
                  {loadingAction === `redeem:${d.id}` ? '兑换中' : '积分兑换'}
                </button>
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

          <div className="mt-6 pt-4 border-t border-bd/40 text-xs text-tx3 leading-5">
            质检不是上传即强制执行：private 数据不自动质检；点击“申请公开并质检”后才会触发公开审核。
            “积分兑换”只对 public 数据成功，private 数据会被后端拒绝。
          </div>
        </section>
      </div>
    </div>
  )
}
