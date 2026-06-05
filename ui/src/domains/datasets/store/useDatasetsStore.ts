import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'
import type { DatasetRef, DatasetStorageStatus, DatasetStorageUsage } from '@/domains/datasets/types'

const DATASETS = '/api/datasets'

interface DatasetsStore {
  datasets: DatasetRef[]
  storageUsage: DatasetStorageUsage | null
  storageStatus: DatasetStorageStatus | null
  loadingAction: string | null
  statusMessage: string
  loadDatasets: (username?: string) => Promise<void>
  loadStorageUsage: (username: string) => Promise<void>
  loadStorageStatus: () => Promise<void>
  uploadDatasetFile: (payload: {
    dataset_id: string
    username: string
    file: File
  }) => Promise<void>
  ingestDataset: (payload: {
    dataset_id: string
    username: string
    source_kind: string
    source_uri: string
    source_auth_ref?: string
    storage_mode?: string
    force?: boolean
  }) => Promise<void>
  requestPublish: (datasetId: string, username: string) => Promise<void>
  redeemAccess: (datasetId: string, username: string) => Promise<void>
  deleteDataset: (datasetId: string, username?: string) => Promise<void>
}

function activeUsername(username?: string): string {
  const saved = typeof localStorage === 'undefined'
    ? ''
    : localStorage.getItem('roboclaw.dataset.username') || ''
  return (username || saved || '').trim()
}

async function loadDatasetRefs(username?: string): Promise<DatasetRef[]> {
  const owner = activeUsername(username)
  const response = await api(owner ? `${DATASETS}?username=${encodeURIComponent(owner)}` : DATASETS)
  return Array.isArray(response) ? response : response.datasets || []
}

export const useDatasetsStore = create<DatasetsStore>((set) => ({
  datasets: [],
  storageUsage: null,
  storageStatus: null,
  loadingAction: null,
  statusMessage: '',

  loadDatasets: async (username) => {
    set({ datasets: await loadDatasetRefs(username) })
  },

  loadStorageUsage: async (username) => {
    if (!username.trim()) {
      set({ storageUsage: null })
      return
    }
    const storageUsage = await api(`${DATASETS}/storage-usage?username=${encodeURIComponent(username.trim())}`)
    set({ storageUsage })
  },

  loadStorageStatus: async () => {
    const storageStatus = await api(`${DATASETS}/storage/status`)
    set({ storageStatus })
  },

  uploadDatasetFile: async (payload) => {
    const contentType = payload.file.type || 'application/octet-stream'
    set({ loadingAction: 'upload-file', statusMessage: '正在生成上传入口...' })
    try {
      const response = await postJson(`${DATASETS}/upload-url`, {
        dataset_id: payload.dataset_id,
        username: payload.username,
        filename: payload.file.name,
        content_type: contentType,
        max_size_bytes: payload.file.size,
      })
      const objectUri = response?.upload?.objectUri || ''
      const upload = response?.upload
      if (!upload?.url || !upload?.fields || !objectUri) {
        throw new Error('上传入口返回不完整')
      }
      set({ loadingAction: 'upload-file', statusMessage: '正在上传到 Evo Studio 平台数据池...' })
      const form = new FormData()
      for (const [key, value] of Object.entries(upload.fields as Record<string, string>)) {
        form.append(key, value)
      }
      form.append('file', payload.file)
      const uploadResponse = await fetch(upload.url, { method: upload.method || 'POST', body: form })
      if (!uploadResponse.ok) {
        const text = await uploadResponse.text().catch(() => '')
        throw new Error(text || `对象存储上传失败：HTTP ${uploadResponse.status}`)
      }

      set({ loadingAction: 'upload-file', statusMessage: '上传完成，正在登记为我的 private 数据...' })
      await postJson(`${DATASETS}/complete-upload`, {
        dataset_id: payload.dataset_id,
        username: payload.username,
        source_kind: 'local_upload',
        source_uri: objectUri,
        source_auth_ref: 'evo-studio-data-pool',
        auto_quality: false,
      })
      const datasets = await loadDatasetRefs(payload.username)
      const storageUsage = await api(`${DATASETS}/storage-usage?username=${encodeURIComponent(payload.username.trim())}`)
      set({
        datasets,
        storageUsage,
        loadingAction: null,
        statusMessage: '上传完成，已加入“我的数据”。现在可以在数据列表里点击“申请公开并质检”。',
      })
    } catch (error) {
      set({
        loadingAction: null,
        statusMessage: error instanceof Error ? `上传失败：${error.message}` : '上传失败',
      })
      throw error
    }
  },

  ingestDataset: async (payload) => {
    const startedAt = new Date().toLocaleTimeString()
    set({
      loadingAction: 'ingest',
      statusMessage: `公开来源保存请求已提交（${startedAt}）。这里只登记链接，不会下载大数据集。`,
    })
    try {
      await postJson(`${DATASETS}/ingest`, {
        ...payload,
        source_auth_ref: payload.source_auth_ref || 'public',
        force: payload.force ?? true,
      })
      const datasets = await loadDatasetRefs(payload.username)
      set({ datasets, loadingAction: null, statusMessage: '公开来源已保存为 private 记录，可在“我的数据”里查看来源。训练时再由云端按需拉取。' })
    } catch (error) {
      set({
        loadingAction: null,
        statusMessage: error instanceof Error ? `接入失败：${error.message}` : '接入失败',
      })
      throw error
    }
  },

  requestPublish: async (datasetId, username) => {
    set({ loadingAction: `publish:${datasetId}`, statusMessage: '已提交公开申请，正在触发质检...' })
    try {
      await postJson(`${DATASETS}/${encodeURIComponent(datasetId)}/publish-request`, { username })
      const datasets = await loadDatasetRefs(username)
      set({ datasets, loadingAction: null, statusMessage: '公开申请已提交，质检任务已启动。' })
    } catch (error) {
      set({
        loadingAction: null,
        statusMessage: error instanceof Error ? `公开申请失败：${error.message}` : '公开申请失败',
      })
      throw error
    }
  },

  redeemAccess: async (datasetId, username) => {
    set({ loadingAction: `redeem:${datasetId}`, statusMessage: '正在兑换数据集使用权...' })
    try {
      await postJson(`${DATASETS}/${encodeURIComponent(datasetId)}/redeem-access`, { username })
      set({ loadingAction: null, statusMessage: '兑换完成，已获得该 public 数据集使用权。' })
    } catch (error) {
      set({
        loadingAction: null,
        statusMessage: error instanceof Error ? `兑换失败：${error.message}` : '兑换失败',
      })
      throw error
    }
  },

  deleteDataset: async (datasetId, username) => {
    const owner = activeUsername(username)
    if (!owner) {
      throw new Error('请输入用户名后继续操作')
    }
    await api(`${DATASETS}/${encodeURIComponent(datasetId)}?username=${encodeURIComponent(owner)}`, { method: 'DELETE' })
    set({ datasets: await loadDatasetRefs(owner) })
  },
}))
