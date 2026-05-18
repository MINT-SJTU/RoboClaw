import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'
import type { DatasetRef, DatasetStorageUsage } from '@/domains/datasets/types'

const DATASETS = '/api/datasets'

interface DatasetsStore {
  datasets: DatasetRef[]
  storageUsage: DatasetStorageUsage | null
  loadingAction: string | null
  statusMessage: string
  loadDatasets: () => Promise<void>
  loadStorageUsage: (username: string) => Promise<void>
  ingestDataset: (payload: {
    dataset_id: string
    username: string
    source_kind: string
    source_uri: string
    source_auth_ref?: string
    force?: boolean
  }) => Promise<void>
  requestPublish: (datasetId: string, username: string) => Promise<void>
  redeemAccess: (datasetId: string, username: string) => Promise<void>
  deleteDataset: (datasetId: string) => Promise<void>
}

async function loadDatasetRefs(): Promise<DatasetRef[]> {
  const response = await api(DATASETS)
  return Array.isArray(response) ? response : response.datasets || []
}

export const useDatasetsStore = create<DatasetsStore>((set) => ({
  datasets: [],
  storageUsage: null,
  loadingAction: null,
  statusMessage: '',

  loadDatasets: async () => {
    set({ datasets: await loadDatasetRefs() })
  },

  loadStorageUsage: async (username) => {
    if (!username.trim()) {
      set({ storageUsage: null })
      return
    }
    const storageUsage = await api(`${DATASETS}/storage-usage?username=${encodeURIComponent(username.trim())}`)
    set({ storageUsage })
  },

  ingestDataset: async (payload) => {
    set({ loadingAction: 'ingest', statusMessage: '数据接入中...' })
    try {
      await postJson(`${DATASETS}/ingest`, {
        ...payload,
        source_auth_ref: payload.source_auth_ref || 'public',
        force: payload.force ?? true,
      })
      const datasets = await loadDatasetRefs()
      set({ datasets, loadingAction: null, statusMessage: '数据已接入为 private。' })
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
      const datasets = await loadDatasetRefs()
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

  deleteDataset: async (datasetId) => {
    await api(`${DATASETS}/${encodeURIComponent(datasetId)}`, { method: 'DELETE' })
    set({ datasets: await loadDatasetRefs() })
  },
}))
