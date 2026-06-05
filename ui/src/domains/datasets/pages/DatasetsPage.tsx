import { useEffect, useMemo, useState } from 'react'
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

function formatDateTime(value: string | undefined, emptyLabel: string) {
  if (!value) return emptyLabel
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function formatRetentionDays(value: number | string | undefined, fallbackLabel: string) {
  const days = Number(value)
  if (!Number.isFinite(days) || days <= 0) return fallbackLabel
  return `${days} 天`
}

export default function DatasetsPage() {
  const datasets = useDatasetsStore((state) => state.datasets)
  const storageUsage = useDatasetsStore((state) => state.storageUsage)
  const storageStatus = useDatasetsStore((state) => state.storageStatus)
  const loadingAction = useDatasetsStore((state) => state.loadingAction)
  const statusMessage = useDatasetsStore((state) => state.statusMessage)
  const loadDatasets = useDatasetsStore((state) => state.loadDatasets)
  const loadStorageUsage = useDatasetsStore((state) => state.loadStorageUsage)
  const loadStorageStatus = useDatasetsStore((state) => state.loadStorageStatus)
  const uploadDatasetFile = useDatasetsStore((state) => state.uploadDatasetFile)
  const ingestDataset = useDatasetsStore((state) => state.ingestDataset)
  const requestPublish = useDatasetsStore((state) => state.requestPublish)
  const redeemAccess = useDatasetsStore((state) => state.redeemAccess)
  const deleteDataset = useDatasetsStore((state) => state.deleteDataset)
  const hubProgress = useHubTransferStore((state) => state.hubProgress)
  const pushDataset = useHubTransferStore((state) => state.pushDataset)
  const { t } = useI18n()

  const [username, setUsername] = useState(() => localStorage.getItem('roboclaw.dataset.username') || '')
  const [datasetId, setDatasetId] = useState('')
  const [sourceUri, setSourceUri] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [sourceMode, setSourceMode] = useState<'robot_upload' | 'public_ref'>('robot_upload')
  const [publicSourceKind, setPublicSourceKind] = useState('remote_dataset')
  const [pushRepoId, setPushRepoId] = useState<Record<string, string>>({})
  const activeDatasetUsername = username.trim()
  const hasUsername = Boolean(activeDatasetUsername)

  useEffect(() => {
    if (!activeDatasetUsername) return
    void loadDatasets(activeDatasetUsername)
    void loadStorageStatus()
  }, [activeDatasetUsername, loadDatasets, loadStorageStatus])

  useEffect(() => {
    if (!activeDatasetUsername) return
    localStorage.setItem('roboclaw.dataset.username', activeDatasetUsername)
    void loadStorageUsage(activeDatasetUsername)
  }, [activeDatasetUsername, loadStorageUsage])

  const resolvedSourceKind = sourceUri.includes('huggingface.co') || sourceUri.startsWith('hf://')
    ? 'remote_dataset'
    : publicSourceKind
  const storageRowsById = useMemo(() => {
    const rows = new Map<string, NonNullable<typeof storageUsage>['datasets'][number]>()
    for (const row of storageUsage?.datasets ?? []) {
      rows.set(row.id, row)
    }
    return rows
  }, [storageUsage])

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
              {!hasUsername && (
                <p className="text-xs text-rd mt-1">请输入用户名后继续操作</p>
              )}
            </div>
            <button
              disabled={!hasUsername}
              onClick={() => { void loadStorageUsage(activeDatasetUsername) }}
              className="px-3 py-2 bg-ac/10 text-ac rounded-lg text-sm font-medium hover:bg-ac/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {t('refreshUsage')}
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">账号档位</div>
              <div className="text-lg font-bold text-tx mt-1">{storageUsage?.role ?? 'free'}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">已用空间</div>
              <div className="text-lg font-bold text-tx mt-1">{formatBytes(storageUsage?.usedBytes)}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">可用空间</div>
              <div className="text-lg font-bold text-tx mt-1">{formatBytes(storageUsage?.availableBytes)}</div>
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg p-3">
              <div className="text-2xs text-tx3">我的数据集</div>
              <div className="text-lg font-bold text-tx mt-1">{storageUsage?.datasetCount ?? 0}</div>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
            <div className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-tx3">
              配额：free {formatBytes(storageStatus?.policy?.quotas?.free)}
              {' · '}contributor {formatBytes(storageStatus?.policy?.quotas?.contributor)}
              {' · '}team {formatBytes(storageStatus?.policy?.quotas?.team)}
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-tx3">
              私有数据保留：{formatRetentionDays(storageStatus?.policy?.privateRetentionDays ?? 30, t('platformPolicy'))}
            </div>
            <div className="bg-bg border border-bd/30 rounded-lg px-3 py-2 text-tx3">
              推荐格式：tar / zip / hdf5 / parquet
            </div>
          </div>

          {statusMessage && (
            <div
              role="status"
              aria-live="polite"
              className="mt-3 text-sm text-ac bg-ac/10 border border-ac/20 rounded-lg px-3 py-2"
            >
              {statusMessage}
            </div>
          )}
        </section>

        {datasets.length === 0 && !hasUsername && (
          <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac mb-5">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">欢迎使用数据资产管理</h3>
            <ol className="mt-3 space-y-2 text-sm text-tx2">
              <li>1. 在上方输入你的用户名</li>
              <li>2. 上传机器人数据或登记公开数据集</li>
              <li>3. 申请共享后可获得贡献积分</li>
            </ol>
          </section>
        )}

        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac mb-5">
          <h3 className="text-sm font-bold text-tx uppercase tracking-wide mb-3">我的数据资产</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
            {[
              ['robot_upload', '上传我的数据', '真机 / 仿真 / 混合数据，默认仅自己可见'],
              ['public_ref', '登记外部公开数据', '只保存公开来源，训练时云端按需拉取'],
            ].map(([mode, title, description]) => (
              <button
                key={mode}
                onClick={() => setSourceMode(mode as typeof sourceMode)}
                className={`text-left border rounded-lg px-3 py-2 transition-colors ${
                  sourceMode === mode
                    ? 'border-ac bg-ac/10 text-ac'
                    : 'border-bd/40 bg-bg text-tx hover:border-ac/50'
                }`}
              >
                <div className="text-sm font-semibold">{title}</div>
                <div className="text-2xs text-tx3 mt-0.5">{description}</div>
              </button>
            ))}
          </div>

          {sourceMode === 'robot_upload' && (
            <div className="bg-bg border border-bd/30 rounded-lg p-4">
              <div className="text-sm font-semibold text-tx">上传我的数据</div>
              <p className="text-xs text-tx3 mt-1">
                上传真机、仿真或混合数据到 Evo Studio 平台数据池的 private 区。上传后默认只属于当前用户，可直接用于自己的训练；只有主动申请共享并通过质检后，才会进入 public 数据池并产生贡献积分。
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-3">
                <div className="bg-sf border border-bd/30 rounded-lg p-3">
                  <div className="text-2xs text-tx3">上传服务</div>
                  <div className={`text-sm font-bold mt-1 ${storageStatus?.configured ? 'text-gn' : 'text-rd'}`}>
                    {storageStatus?.configured ? '可用' : '待管理员配置'}
                  </div>
                </div>
                <div className="bg-sf border border-bd/30 rounded-lg p-3">
                  <div className="text-2xs text-tx3">默认权限</div>
                  <div className="text-sm font-bold text-tx mt-1">仅自己可见</div>
                </div>
                <div className="bg-sf border border-bd/30 rounded-lg p-3">
                  <div className="text-2xs text-tx3">共享方式</div>
                  <div className="text-sm font-bold text-tx mt-1">申请后质检</div>
                </div>
              </div>
              <div className="mt-3 bg-sf border border-bd/30 rounded-lg px-3 py-2 text-xs text-tx3 leading-5">
                推荐上传打包后的数据文件，例如 `.tar` / `.tar.gz` / `.zip` / `.hdf5` / `.parquet`。机器人数据常包含大量图片、状态和动作文件，打包后上传更快，也更适合后续训练读取。未完成登记的临时文件会定期清理；已加入“我的数据”的 private 数据受账号配额和保留期限制，通过质检成为 public 后才进入长期共享池。
              </div>
              {!storageStatus?.configured && (
                <div className="text-xs text-rd bg-rd/10 border border-rd/20 rounded-lg px-3 py-2 mt-3">
                  Evo Studio 平台数据池还没有连接，暂时不能上传大文件。管理员需在服务器配置 ROBOCLAW_STORAGE_* 环境变量后重启服务。
                </div>
              )}
              {storageStatus?.configured && !storageStatus.clientAvailable && (
                <div className="text-xs text-rd mt-3">
                  Python S3 客户端未安装，请安装可选依赖：pip install -e ".[s3]"
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-[1fr_1.3fr_auto] gap-2 mt-3">
                <input
                  placeholder="数据集 ID，例如 so101-session-001"
                  value={datasetId}
                  onChange={(event) => setDatasetId(event.target.value)}
                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                />
                <input
                  type="file"
                  accept=".tar,.tar.gz,.tgz,.zip,.hdf5,.h5,.parquet,.jsonl"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                />
                <button
                  disabled={
                    !datasetId
                    || !selectedFile
                    || !hasUsername
                    || !storageStatus?.configured
                    || !storageStatus?.clientAvailable
                    || loadingAction === 'upload-file'
                  }
                  onClick={() => {
                    if (!selectedFile) return
                    void uploadDatasetFile({
                      dataset_id: datasetId,
                      username: activeDatasetUsername,
                      file: selectedFile,
                    }).then(() => {
                      setDatasetId('')
                      setSelectedFile(null)
                    })
                  }}
                  className="px-3 py-2 bg-ac text-white rounded-lg text-sm font-semibold disabled:bg-bd/40 disabled:text-tx3 disabled:cursor-not-allowed"
                >
                  {loadingAction === 'upload-file' ? '上传中' : '上传并加入我的数据'}
                </button>
              </div>
              <div className="text-xs text-tx3 mt-2">
                上传成功后会自动出现在下方“我的数据”列表里；那一行会显示“申请公开并质检”按钮。
              </div>
              <button
                onClick={() => { void loadStorageStatus() }}
                className="mt-3 px-3 py-2 bg-ac/10 text-ac rounded-lg text-sm font-semibold hover:bg-ac/20"
              >
                刷新数据池状态
              </button>
            </div>
          )}

          {sourceMode === 'public_ref' && (
            <div className="bg-bg border border-bd/30 rounded-lg p-4">
              <div className="text-sm font-semibold text-tx">登记外部公开数据</div>
              <p className="text-xs text-tx3 mt-1 mb-3">
                这里只保存 HuggingFace、ModelScope、Kaggle 等公开数据来源。平台不会把大数据集下载到你的电脑，也不会因为登记链接发积分；真正训练时由云端按需拉取或缓存。
              </p>
              <div className="grid grid-cols-1 md:grid-cols-[180px_1fr_1fr_auto] gap-2">
                <select
                  value={publicSourceKind}
                  onChange={(event) => setPublicSourceKind(event.target.value)}
                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                >
                  <option value="remote_dataset">HuggingFace</option>
                  <option value="modelscope_dataset">ModelScope</option>
                  <option value="kaggle_dataset">Kaggle</option>
                  <option value="dagshub_dvc">DagsHub / DVC</option>
                  <option value="public_http">公开 HTTP 链接</option>
                </select>
                <input
                  placeholder="平台内显示的 ID，例如 gr00t-libero"
                  value={datasetId}
                  onChange={(event) => setDatasetId(event.target.value)}
                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                />
                <input
                  placeholder="公开链接，例如 https://huggingface.co/datasets/..."
                  value={sourceUri}
                  onChange={(event) => setSourceUri(event.target.value)}
                  className="bg-sf border border-bd text-tx px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-ac"
                />
                <button
                  disabled={!datasetId || !sourceUri || !hasUsername || loadingAction === 'ingest'}
                  onClick={() => {
                    void ingestDataset({
                      dataset_id: datasetId,
                      username: activeDatasetUsername,
                      source_kind: resolvedSourceKind,
                      source_uri: sourceUri,
                      storage_mode: 'external_reference',
                    }).then(() => {
                      setDatasetId('')
                      setSourceUri('')
                      void loadStorageUsage(activeDatasetUsername)
                    })
                  }}
                  className="px-3 py-2 bg-ac text-white rounded-lg text-sm font-semibold hover:bg-ac/90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {loadingAction === 'ingest' ? '保存中' : '保存来源'}
                </button>
              </div>
            </div>
          )}

          <p className="text-xs text-tx3 mt-2">
            管理员批量迁移、已在云端的数据登记、服务器目录导入等内部操作不放在用户页面；后续通过内部脚本、后台任务或单独管理页处理。
          </p>
        </section>

        <section className="bg-sf rounded-xl p-5 shadow-card shadow-inset-ac">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-tx uppercase tracking-wide">{t('datasets')}</h3>
            <button
              disabled={!hasUsername}
              onClick={() => { void loadDatasets(activeDatasetUsername) }}
              className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {t('refresh')}
            </button>
          </div>

          {datasets.length === 0 && (
            <div className="text-tx3 text-center py-8 text-sm">{t('noDatasets')}</div>
          )}
          <div className="space-y-1.5">
            {datasets.map((d) => {
              const storageRow = storageRowsById.get(d.id)
              return (
                <div
                  key={d.id}
                  className="bg-bg border border-bd/30 rounded-lg px-3 py-2.5 flex flex-wrap items-center gap-2 text-sm"
                >
                  <div className="flex-1 min-w-[240px]">
                    <div className="font-semibold text-tx truncate">{d.label}</div>
                    <div className="text-2xs text-tx3 mt-0.5 truncate">
                      {storageRow
                        ? `${storageRow.visibility} · ${storageRow.storageMode || 'managed'} · ${storageRow.sourceKind || 'unknown'} · ${storageRow.sourceUri || '无来源'}`
                        : '无访问权限，请确认用户名或申请授权'}
                    </div>
                    <div className="text-2xs text-tx3 mt-0.5">
                      接入时间：{formatDateTime(storageRow?.ingestedAt || storageRow?.createdAt, t('notRecorded'))}
                      {storageRow?.publicationStatus ? ` · 公开状态：${storageRow.publicationStatus}` : ''}
                      {storageRow ? ` · 可训练：${storageRow.canTrain ? '是' : '否'}` : ''}
                      {storageRow?.visibility === 'private' && storageRow.privateExpiresAt
                        ? ` · 私有保留至：${formatDateTime(storageRow.privateExpiresAt, t('notRecorded'))}`
                        : ''}
                    </div>
                  </div>
                  <span className="text-tx3 text-2xs font-mono whitespace-nowrap">
                    {`${d.stats.total_episodes} ep`}
                    {` · ${d.stats.total_frames} fr`}
                    {` · ${formatBytes(d.stats.total_bytes)}`}
                  </span>
                  <button
                    disabled={!hasUsername || loadingAction === `publish:${d.id}`}
                    onClick={() => {
                      void requestPublish(d.id, activeDatasetUsername).then(() => {
                        void loadStorageUsage(activeDatasetUsername)
                      })
                    }}
                    className="px-2 py-0.5 text-gn/70 rounded text-xs hover:text-gn hover:bg-gn/10 transition-colors disabled:opacity-25"
                  >
                    {loadingAction === `publish:${d.id}` ? '质检中' : '申请公开并质检'}
                  </button>
                  <button
                    disabled={!hasUsername || loadingAction === `redeem:${d.id}`}
                    onClick={() => {
                      void redeemAccess(d.id, activeDatasetUsername).then(() => {
                        void loadStorageUsage(activeDatasetUsername)
                      })
                    }}
                    className="px-2 py-0.5 text-ac/70 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                  >
                    {loadingAction === `redeem:${d.id}` ? '兑换中' : '积分兑换'}
                  </button>
                  <div className="flex items-center gap-1">
                    <input
                      value={pushRepoId[d.id] || ''}
                      onChange={(event) => {
                        const value = event.target.value
                        setPushRepoId((current) => ({ ...current, [d.id]: value }))
                      }}
                      placeholder={t('enterRepoId')}
                      className="w-36 bg-sf border border-bd text-tx px-2 py-1 rounded text-xs focus:outline-none focus:border-ac disabled:opacity-40"
                      disabled={!hasUsername || !d.capabilities.can_push}
                    />
                    <button
                      disabled={!hasUsername || !d.capabilities.can_push || !pushRepoId[d.id]?.trim()}
                      onClick={() => {
                        const repoId = pushRepoId[d.id]?.trim()
                        if (!repoId) return
                        void pushDataset(d.id, repoId)
                      }}
                      className="px-2 py-0.5 text-ac/60 rounded text-xs hover:text-ac hover:bg-ac/10 transition-colors disabled:opacity-25"
                    >
                      {t('pushToHub')}
                    </button>
                  </div>
                  <button
                    disabled={!hasUsername}
                    onClick={() => {
                      if (confirm(`${t('deleteConfirm')} "${d.label}"?`)) {
                        void deleteDataset(d.id, activeDatasetUsername)
                      }
                    }}
                    className="px-2 py-0.5 text-rd/60 rounded text-xs hover:text-rd hover:bg-rd/10 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
                  >
                    {t('del')}
                  </button>
                </div>
              )
            })}
          </div>

          {/* Hub progress bar */}
          {hubProgress && !hubProgress.done && (
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
