import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'

const TRAIN = '/api/train'
const POLICIES = '/api/policies'
const CURRENT_TRAIN_JOB_KEY = 'roboclaw.currentTrainJobId'

function loadStoredTrainJobId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(CURRENT_TRAIN_JOB_KEY) || ''
}

function storeTrainJobId(jobId: string) {
  if (typeof window === 'undefined') return
  if (jobId) {
    window.localStorage.setItem(CURRENT_TRAIN_JOB_KEY, jobId)
  } else {
    window.localStorage.removeItem(CURRENT_TRAIN_JOB_KEY)
  }
}

export interface Policy {
  name: string
  checkpoint: string
  dataset?: string
  steps?: number
}

export interface TrainingCurvePoint {
  step: string
  ep: number
  epoch: number
  loss: number
}

export interface TrainingCurve {
  job_id: string
  log_path: string
  exists: boolean
  points: TrainingCurvePoint[]
  last_epoch: number | null
  last_loss: number | null
  best_ep: number | null
  best_loss: number | null
  updated_at: number | null
}

interface TrainingStore {
  policies: Policy[]
  trainJobMessage: string
  currentTrainJobId: string
  trainCurve: TrainingCurve | null
  trainingLoading: boolean
  trainingStopLoading: boolean
  loadPolicies: () => Promise<void>
  restoreCurrentTrainJob: () => Promise<void>
  doTrainStart: (params: { dataset_name: string; steps?: number; device?: string; policy_type?: string }) => Promise<void>
  doTrainStop: () => Promise<void>
  fetchTrainStatus: (jobId: string) => Promise<void>
  fetchTrainCurve: (jobId: string) => Promise<void>
  clearTrainCurve: () => void
}

export const useTrainingStore = create<TrainingStore>((set) => ({
  policies: [],
  trainJobMessage: '',
  currentTrainJobId: loadStoredTrainJobId(),
  trainCurve: null,
  trainingLoading: false,
  trainingStopLoading: false,

  loadPolicies: async () => {
    const response = await api(`${POLICIES}`)
    set({ policies: Array.isArray(response) ? response : response.policies || [] })
  },

  restoreCurrentTrainJob: async () => {
    const storedJobId = loadStoredTrainJobId()
    if (storedJobId) {
      try {
        const status = await api(`${TRAIN}/status/${encodeURIComponent(storedJobId)}`)
        const message = dataMessage(status)
        if (message.includes('running: True')) {
          set({ currentTrainJobId: storedJobId, trainJobMessage: message })
          return
        }
      } catch {
        storeTrainJobId('')
      }
    }

    const current = await api(`${TRAIN}/current`)
    const jobId = typeof current.job_id === 'string' ? current.job_id : ''
    if (jobId && current.running) {
      storeTrainJobId(jobId)
      set({ currentTrainJobId: jobId, trainJobMessage: statusMessage(current) })
      return
    }

    storeTrainJobId('')
    set({ currentTrainJobId: '' })
  },

  doTrainStart: async (params) => {
    set({ trainingLoading: true })
    try {
      const data = await postJson(`${TRAIN}/start`, params)
      const jobId = typeof data.job_id === 'string' ? data.job_id : ''
      storeTrainJobId(jobId)
      set({ trainJobMessage: data.message || '', currentTrainJobId: jobId })
    } finally {
      set({ trainingLoading: false })
    }
  },

  doTrainStop: async () => {
    const jobId = useTrainingStore.getState().currentTrainJobId
    if (!jobId) {
      set({ trainJobMessage: 'No active training job id.' })
      return
    }
    set({ trainingStopLoading: true })
    try {
      const data = await postJson(`${TRAIN}/stop`, { job_id: jobId })
      storeTrainJobId('')
      set({ trainJobMessage: data.message || '', currentTrainJobId: '' })
    } finally {
      set({ trainingStopLoading: false })
    }
  },

  fetchTrainStatus: async (jobId) => {
    const data = await api(`${TRAIN}/status/${encodeURIComponent(jobId)}`)
    const message = data.message || ''
    if (!message.includes('running: True') && useTrainingStore.getState().currentTrainJobId === jobId) {
      storeTrainJobId('')
      set({ trainJobMessage: message, currentTrainJobId: '' })
      return
    }
    set({ trainJobMessage: message })
  },

  fetchTrainCurve: async (jobId) => {
    const data = await api(`${TRAIN}/curve/${encodeURIComponent(jobId)}`) as TrainingCurve
    set({ trainCurve: data })
  },

  clearTrainCurve: () => {
    set({ trainCurve: null })
  },
}))

function dataMessage(data: any) {
  return typeof data.message === 'string' ? data.message : statusMessage(data)
}

function statusMessage(data: any) {
  return Object.entries(data)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')
}
