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

function errorMessage(error: unknown) {
  if (error instanceof Error && error.message.trim()) return error.message.trim()
  return 'Unknown error'
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

export interface TrainingPresetCapability {
  id: string
  backend_preset: string
  label: string
  summary: string
  gpu_type: string
  gpu_count: number
  cpu_cores: number
  memory_gb: number
  node_count: number
}

export interface TrainingProviderCapability {
  id: string
  display_name: string
  kind: 'current_machine' | 'remote_backend'
  configured: boolean
  default_image_configured?: boolean
  presets: TrainingPresetCapability[]
  supports_image_override: boolean
  supports_resource_overrides: boolean
}

export interface TrainingCapabilities {
  locations: {
    current_machine: { configured: boolean }
    remote_backend: {
      configured: boolean
      mode: 'self_hosted' | 'managed' | 'unavailable'
      notice?: string
    }
  }
  providers: Record<string, TrainingProviderCapability>
}

export interface TrainingStatusData {
  job_id: string
  provider?: string
  status?: string
  running?: boolean
  terminal?: boolean
  message?: string
  remote_job_id?: string
  log_path?: string
  log_tail?: string
  output_dir?: string
  updated_at?: number | null
  provider_data?: Record<string, unknown>
}

export interface TrainingStartParams {
  dataset_name: string
  steps?: number
  device?: string
  policy_type?: string
  provider?: 'local' | 'aliyun' | 'autodl'
  preset?: string
  job_name?: string
  gpu_count?: number
  gpu_type?: string
  cpu_cores?: number
  memory_gb?: number
  node_count?: number
  image?: string
}

interface TrainingStore {
  policies: Policy[]
  trainJobMessage: string
  currentTrainJobId: string
  trainCurve: TrainingCurve | null
  trainingCapabilities: TrainingCapabilities | null
  trainJobStatus: TrainingStatusData | null
  trainingLoading: boolean
  trainingStopLoading: boolean
  loadPolicies: () => Promise<void>
  loadTrainingCapabilities: () => Promise<void>
  restoreCurrentTrainJob: () => Promise<void>
  doTrainStart: (params: TrainingStartParams) => Promise<void>
  doTrainStop: (jobId?: string) => Promise<void>
  fetchTrainStatus: (jobId: string) => Promise<void>
  fetchTrainCurve: (jobId: string) => Promise<void>
  clearTrainCurve: () => void
}

export const useTrainingStore = create<TrainingStore>((set) => ({
  policies: [],
  trainJobMessage: '',
  currentTrainJobId: loadStoredTrainJobId(),
  trainCurve: null,
  trainingCapabilities: null,
  trainJobStatus: null,
  trainingLoading: false,
  trainingStopLoading: false,

  loadPolicies: async () => {
    const response = await api(`${POLICIES}`)
    set({ policies: Array.isArray(response) ? response : response.policies || [] })
  },

  loadTrainingCapabilities: async () => {
    try {
      const response = await api(`${TRAIN}/capabilities`) as TrainingCapabilities
      set({ trainingCapabilities: response })
    } catch (error) {
      console.error('Failed to load training capabilities:', error)
      set({ trainingCapabilities: null })
    }
  },

  restoreCurrentTrainJob: async () => {
    const storedJobId = loadStoredTrainJobId()
    if (storedJobId) {
      try {
        const status = await api(`${TRAIN}/status/${encodeURIComponent(storedJobId)}`)
        const message = dataMessage(status)
        if (Boolean((status as TrainingStatusData).running)) {
          set({ currentTrainJobId: storedJobId, trainJobMessage: message, trainJobStatus: status as TrainingStatusData })
          return
        }
      } catch (error) {
        console.warn('Failed to restore stored training job; clearing:', error)
        storeTrainJobId('')
      }
    }

    const current = await api(`${TRAIN}/current`)
    const jobId = typeof current.job_id === 'string' ? current.job_id : ''
    if (jobId && current.running) {
      storeTrainJobId(jobId)
      set({ currentTrainJobId: jobId, trainJobMessage: statusMessage(current), trainJobStatus: current as TrainingStatusData })
      return
    }

    storeTrainJobId('')
    set({ currentTrainJobId: '', trainJobStatus: null })
  },

  doTrainStart: async (params) => {
    set({ trainingLoading: true })
    try {
      const data = await postJson(`${TRAIN}/start`, params)
      const jobId = extractJobId(data)
      storeTrainJobId(jobId)
      set({ trainJobMessage: data.message || '', currentTrainJobId: jobId, trainJobStatus: null })
    } catch (error) {
      storeTrainJobId('')
      set({
        trainJobMessage: `status: failed\nmessage: ${errorMessage(error)}`,
        currentTrainJobId: '',
        trainJobStatus: {
          job_id: '',
          status: 'failed',
          running: false,
          terminal: true,
          message: errorMessage(error),
        },
      })
    } finally {
      set({ trainingLoading: false })
    }
  },

  doTrainStop: async (jobIdArg) => {
    const state = useTrainingStore.getState()
    const jobId = jobIdArg || state.currentTrainJobId || state.trainJobStatus?.job_id || ''
    if (!jobId) {
      set({ trainJobMessage: 'No active training job id.', trainJobStatus: null })
      return
    }
    set({ trainingStopLoading: true })
    try {
      const data = await postJson(`${TRAIN}/stop`, { job_id: jobId })
      storeTrainJobId('')
      set({ trainJobMessage: data.message || '', currentTrainJobId: '', trainJobStatus: null })
    } catch (error) {
      set({
        trainJobMessage: `status: failed\nmessage: ${errorMessage(error)}`,
        trainJobStatus: {
          job_id: jobId,
          status: 'failed',
          running: false,
          terminal: true,
          message: errorMessage(error),
        },
      })
    } finally {
      set({ trainingStopLoading: false })
    }
  },

  fetchTrainStatus: async (jobId) => {
    const data = await api(`${TRAIN}/status/${encodeURIComponent(jobId)}`)
    const message = data.message || ''
    const status = data as TrainingStatusData
    if (!status.running && useTrainingStore.getState().currentTrainJobId === jobId) {
      storeTrainJobId('')
      set({ trainJobMessage: message, currentTrainJobId: '', trainJobStatus: status })
      return
    }
    set({ trainJobMessage: message, trainJobStatus: status })
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

function extractJobId(data: any) {
  if (typeof data?.job_id === 'string' && data.job_id.trim()) {
    return data.job_id.trim().split('\n', 1)[0].trim()
  }
  if (typeof data?.message !== 'string') return ''
  const match = data.message.match(/(?:^|\n)[^\n]*Job ID:\s*([^\n]+)/)
  return match?.[1]?.trim() ?? ''
}
