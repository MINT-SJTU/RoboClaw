import { create } from 'zustand'
import { api } from '@/shared/api/client'
import type { DatasetRef } from '@/domains/datasets/types'

const DATASETS = '/api/datasets'

interface DatasetsStore {
  datasets: DatasetRef[]
  loadDatasets: () => Promise<void>
  deleteDataset: (datasetId: string) => Promise<void>
}

async function loadDatasetRefs(): Promise<DatasetRef[]> {
  const response = await api(DATASETS)
  return Array.isArray(response) ? response : response.datasets || []
}

export const useDatasetsStore = create<DatasetsStore>((set) => ({
  datasets: [],

  loadDatasets: async () => {
    set({ datasets: await loadDatasetRefs() })
  },

  deleteDataset: async (datasetId) => {
    await api(`${DATASETS}/${encodeURIComponent(datasetId)}`, { method: 'DELETE' })
    set({ datasets: await loadDatasetRefs() })
  },
}))
