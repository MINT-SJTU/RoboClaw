import { create } from 'zustand'
import { postJson } from '@/shared/api/client'
import { useDatasetsStore } from '@/domains/datasets/store/useDatasetsStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'

const HUB = '/api/hub'

export interface HubProgress {
  operation: string
  progress_percent: number
  total_bytes: number
  done: boolean
}

interface HubTransferStore {
  hubLoading: string | null
  hubProgress: HubProgress | null
  pushDataset: (datasetId: string, repoId: string) => Promise<void>
  pullDataset: (repoId: string, datasetId?: string) => Promise<void>
  pushPolicy: (name: string, repoId: string) => Promise<void>
  pullPolicy: (repoId: string, name?: string) => Promise<void>
  handleDashboardEvent: (event: any) => void
}

export const useHubTransferStore = create<HubTransferStore>((set) => ({
  hubLoading: null,
  hubProgress: null,

  pushDataset: async (datasetId, repoId) => {
    set({ hubLoading: 'pushDataset' })
    try {
      await postJson(`${HUB}/datasets/push`, { dataset_id: datasetId, repo_id: repoId })
    } finally {
      set({ hubLoading: null })
    }
  },

  pullDataset: async (repoId, datasetId) => {
    set({ hubLoading: 'pullDataset', hubProgress: null })
    try {
      await postJson(`${HUB}/datasets/pull`, { repo_id: repoId, dataset_id: datasetId || '' })
      await useDatasetsStore.getState().loadDatasets()
    } finally {
      set({ hubLoading: null, hubProgress: null })
    }
  },

  pushPolicy: async (name, repoId) => {
    set({ hubLoading: 'pushPolicy' })
    try {
      await postJson(`${HUB}/policies/push`, { name, repo_id: repoId })
    } finally {
      set({ hubLoading: null })
    }
  },

  pullPolicy: async (repoId, name) => {
    set({ hubLoading: 'pullPolicy', hubProgress: null })
    try {
      await postJson(`${HUB}/policies/pull`, { repo_id: repoId, name: name || '' })
      await useTrainingStore.getState().loadPolicies()
    } finally {
      set({ hubLoading: null, hubProgress: null })
    }
  },

  handleDashboardEvent: (event) => {
    if (event.type !== 'dashboard.hub.progress') {
      return
    }

    set({
      hubProgress: {
        operation: event.operation ?? '',
        progress_percent: event.progress_percent ?? 0,
        total_bytes: event.total_bytes ?? 0,
        done: !!event.done,
      },
    })
  },
}))
