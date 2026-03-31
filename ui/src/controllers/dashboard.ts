import { create } from 'zustand'

interface ArmStatus {
  alias: string
  type: string
  role: string
  connected: boolean
  calibrated: boolean
}

interface CameraStatus {
  alias: string
  connected: boolean
  width: number
  height: number
}

interface HardwareStatus {
  ready: boolean
  missing: string[]
  arms: ArmStatus[]
  cameras: CameraStatus[]
  recording_active: boolean
}

interface RecordingState {
  session_id: string
  dataset_name: string
  dataset_root: string
  task: string
  state: string
  current_episode: number
  total_episodes: number
  total_frames: number
  elapsed_seconds: number
  error_message: string
}

interface CompletionSummary {
  dataset_name: string
  dataset_root: string
  episodes_completed: number
  total_frames: number
}

interface Fault {
  fault_type: string
  device_alias: string
  message: string
  timestamp: number
}

interface TroubleshootEntry {
  title: string
  description: string
  steps: string[]
  can_recheck: boolean
}

interface StartRecordingParams {
  task: string
  num_episodes: number
  episode_time_s: number
  reset_time_s: number
}

interface NetworkInfo {
  host: string
  port: number
  lan_ip: string
}

interface DashboardStore {
  hardwareStatus: HardwareStatus | null
  recording: RecordingState | null
  completionSummary: CompletionSummary | null
  activeFaults: Fault[]
  troubleshootMap: Record<string, TroubleshootEntry> | null
  networkInfo: NetworkInfo | null

  fetchHardwareStatus: () => Promise<void>
  startRecording: (params: StartRecordingParams) => Promise<void>
  stopRecording: () => Promise<void>
  fetchTroubleshootMap: () => Promise<void>
  fetchNetworkInfo: () => Promise<void>
  recheckFault: (faultType: string, deviceAlias: string) => Promise<void>
  generateSnapshot: () => Promise<any>
  dismissFault: (faultType: string, deviceAlias: string) => void
  clearCompletion: () => void
  handleDashboardEvent: (event: any) => void
}

export type {
  ArmStatus,
  CameraStatus,
  HardwareStatus,
  RecordingState,
  CompletionSummary,
  Fault,
  TroubleshootEntry,
  StartRecordingParams,
  NetworkInfo,
}

export const useDashboard = create<DashboardStore>((set, get) => ({
  hardwareStatus: null,
  recording: null,
  completionSummary: null,
  activeFaults: [],
  troubleshootMap: null,
  networkInfo: null,

  fetchHardwareStatus: async () => {
    const res = await fetch('/api/dashboard/hardware-status')
    if (!res.ok) return
    set({ hardwareStatus: await res.json() })
  },

  startRecording: async (params) => {
    const res = await fetch('/api/dashboard/recording/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    if (!res.ok) {
      const body = await res.json()
      throw new Error(body.detail || '启动录制失败')
    }
    const data = await res.json()
    set({
      recording: {
        ...data,
        state: 'starting',
        task: params.task,
        current_episode: 0,
        total_episodes: params.num_episodes,
        total_frames: 0,
        elapsed_seconds: 0,
        error_message: '',
        dataset_root: data.dataset_root || '',
      },
      completionSummary: null,
    })
  },

  stopRecording: async () => {
    await fetch('/api/dashboard/recording/stop', { method: 'POST' })
  },

  fetchTroubleshootMap: async () => {
    const res = await fetch('/api/dashboard/troubleshoot-map')
    if (!res.ok) return
    set({ troubleshootMap: await res.json() })
  },

  fetchNetworkInfo: async () => {
    const res = await fetch('/api/dashboard/network-info')
    if (!res.ok) return
    set({ networkInfo: await res.json() })
  },

  recheckFault: async (faultType, deviceAlias) => {
    const res = await fetch('/api/dashboard/troubleshoot/recheck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fault_type: faultType, device_alias: deviceAlias }),
    })
    if (!res.ok) return
    const data = await res.json()
    set({ activeFaults: data.faults || [] })
    get().fetchHardwareStatus()
  },

  generateSnapshot: async () => {
    const res = await fetch('/api/dashboard/troubleshoot/snapshot', { method: 'POST' })
    return res.json()
  },

  dismissFault: (faultType, deviceAlias) => {
    set((state) => ({
      activeFaults: state.activeFaults.filter(
        (f) => !(f.fault_type === faultType && f.device_alias === deviceAlias),
      ),
    }))
  },

  clearCompletion: () => set({ completionSummary: null }),

  handleDashboardEvent: (event) => {
    const type = event.type as string

    if (type === 'dashboard.recording.progress') {
      set({ recording: event as RecordingState })
      return
    }

    if (type === 'dashboard.recording.completed') {
      const status = event as RecordingState
      set({
        recording: null,
        completionSummary: {
          dataset_name: status.dataset_name,
          dataset_root: status.dataset_root,
          episodes_completed: status.current_episode,
          total_frames: status.total_frames,
        },
      })
      return
    }

    if (type === 'dashboard.recording.error') {
      set({ recording: { ...(event as RecordingState), state: 'error' } })
      return
    }

    if (type === 'dashboard.fault') {
      const fault: Fault = {
        fault_type: event.fault_type,
        device_alias: event.device_alias,
        message: event.message,
        timestamp: event.timestamp,
      }
      set((state) => ({
        activeFaults: [
          ...state.activeFaults.filter(
            (f) => !(f.fault_type === fault.fault_type && f.device_alias === fault.device_alias),
          ),
          fault,
        ],
      }))
      return
    }

    if (type === 'dashboard.fault.resolved') {
      set((state) => ({
        activeFaults: state.activeFaults.filter(
          (f) => !(f.fault_type === event.fault_type && f.device_alias === event.device_alias),
        ),
      }))
    }
  },
}))
