import { create } from 'zustand'
import { api, postJson } from './api'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SessionState = 'idle' | 'preparing' | 'calibrating' | 'teleoperating' | 'recording' | 'replaying' | 'inferring' | 'error'
export type EpisodePhase = '' | 'recording' | 'saving' | 'resetting'

export interface ArmStatus {
  alias: string
  type: string
  role: string
  connected: boolean
  calibrated: boolean
}

export interface CameraStatus {
  alias: string
  connected: boolean
  width: number
  height: number
}

export interface HardwareStatus {
  ready: boolean
  missing: string[]
  arms: ArmStatus[]
  cameras: CameraStatus[]
  session_busy: boolean
}

export interface SessionStatus {
  state: SessionState
  episode_phase: EpisodePhase
  saved_episodes: number
  current_episode: number
  target_episodes: number
  total_frames: number
  elapsed_seconds: number
  dataset: string | null
  rerun_web_port: number
  error: string
  // Calibration-specific (populated when state === 'calibrating')
  calibration_step: string
  calibration_arm: string
  calibration_positions: Record<string, { min: number; pos: number; max: number }> | null
  // Cross-process embodiment owner (e.g. "teleop", "recording")
  embodiment_owner: string
  // Human-readable preparation milestone (used by inference panel)
  prepare_stage: string
}

export interface Fault {
  fault_type: string
  device_alias: string
  message: string
  timestamp: number
}

export interface TroubleshootEntry {
  can_recheck: boolean
  step_count: number
}

export interface Dataset {
  name: string
  total_episodes?: number
  total_frames?: number
  fps?: number
}

export interface Policy {
  name: string
  checkpoint: string
  dataset?: string
  steps?: number
}

export interface NetworkInfo {
  host: string
  port: number
  lan_ip: string
}

interface CalibrationStatus {
  state: string
  arm_alias: string
  mins?: Record<string, number>
  maxes?: Record<string, number>
  homing_offsets?: Record<string, number>
  error?: string
}

export interface StartRecordingParams {
  task: string
  num_episodes: number
  fps?: number
  episode_time_s: number
  reset_time_s: number
  dataset_name?: string
  use_cameras?: boolean
  arms?: string
}

interface LogEntry {
  time: string
  message: string
  cls: 'info' | 'ok' | 'err'
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface DashboardStore {
  // State
  session: SessionStatus
  hardwareStatus: HardwareStatus | null
  datasets: Dataset[]
  activeFaults: Fault[]
  troubleshootMap: Record<string, TroubleshootEntry> | null
  networkInfo: NetworkInfo | null
  logs: LogEntry[]
  loading: string | null
  calibration: CalibrationStatus

  // Session actions
  doDismissError: () => Promise<void>
  doTeleopStart: (params?: { fps?: number; arms?: string }) => Promise<void>
  doTeleopStop: () => Promise<void>
  doRecordStart: (params: StartRecordingParams) => Promise<void>
  doRecordStop: () => Promise<void>
  doSaveEpisode: () => Promise<void>
  doDiscardEpisode: () => Promise<void>
  doSkipReset: () => Promise<void>
  fetchSessionStatus: () => Promise<void>

  // Hardware & datasets
  fetchHardwareStatus: () => Promise<void>
  loadDatasets: () => Promise<void>
  loadPolicies: () => Promise<void>
  deleteDataset: (name: string) => Promise<void>

  // Troubleshooting
  fetchTroubleshootMap: () => Promise<void>
  fetchNetworkInfo: () => Promise<void>
  recheckFault: (faultType: string, deviceAlias: string) => Promise<void>
  generateSnapshot: () => Promise<any>
  dismissFault: (faultType: string, deviceAlias: string) => void

  // Replay
  doReplayStart: (params: { dataset_name: string; episode?: number; fps?: number }) => Promise<void>
  doReplayStop: () => Promise<void>

  // Training
  doTrainStart: (params: { dataset_name: string; steps?: number; device?: string }) => Promise<void>
  fetchTrainStatus: (jobId: string) => Promise<void>
  fetchTrainDatasets: () => Promise<void>
  trainJobMessage: string
  policies: Policy[]

  // Inference
  doInferStart: (params: { checkpoint_path?: string; num_episodes?: number; episode_time_s?: number }) => Promise<void>
  doInferStop: () => Promise<void>

  // Hub
  hubLoading: string | null
  hubProgress: { operation: string; progress_percent: number; total_bytes: number; done: boolean } | null
  pushDataset: (name: string, repoId: string) => Promise<void>
  pullDataset: (repoId: string, name?: string) => Promise<void>
  pushPolicy: (name: string, repoId: string) => Promise<void>
  pullPolicy: (repoId: string, name?: string) => Promise<void>

  // Events & logging
  handleDashboardEvent: (event: any) => void
  addLog: (message: string, cls?: 'info' | 'ok' | 'err') => void
  clearLog: () => void
}

const SESSION = '/api/session'
const TELEOP = '/api/teleop'
const RECORD = '/api/record'
const HARDWARE = '/api/hardware'
const DATASETS = '/api/datasets'
const POLICIES = '/api/policies'
const TROUBLESHOOT = '/api/troubleshoot'
const SYSTEM = '/api/system'
const REPLAY = '/api/replay'
const TRAIN = '/api/train'
const INFER = '/api/infer'
const HUB = '/api/hub'

// ---------------------------------------------------------------------------
// Default session state
// ---------------------------------------------------------------------------

const defaultSession: SessionStatus = {
  state: 'idle',
  episode_phase: '',
  saved_episodes: 0,
  current_episode: 0,
  target_episodes: 0,
  total_frames: 0,
  elapsed_seconds: 0,
  dataset: null,
  rerun_web_port: 0,
  error: '',
  calibration_step: '',
  calibration_arm: '',
  calibration_positions: null,
  embodiment_owner: '',
  prepare_stage: '',
}

// ---------------------------------------------------------------------------
// Store implementation
// ---------------------------------------------------------------------------

const defaultCalibration: CalibrationStatus = { state: 'idle', arm_alias: '' }

export const useDashboard = create<DashboardStore>((set, get) => ({
  session: { ...defaultSession },
  hardwareStatus: null,
  datasets: [],
  activeFaults: [],
  troubleshootMap: null,
  networkInfo: null,
  logs: [],
  loading: null,
  calibration: { ...defaultCalibration },
  trainJobMessage: '',
  policies: [],
  hubLoading: null,
  hubProgress: null,

  addLog: (message, cls = 'info') => {
    const time = new Date().toLocaleTimeString()
    set((s) => ({ logs: [...s.logs.slice(-199), { time, message, cls }] }))
  },

  clearLog: () => set({ logs: [] }),

  // -- Session lifecycle --------------------------------------------------

  doDismissError: async () => {
    try {
      await postJson(`${SESSION}/dismiss-error`)
    } catch (e: unknown) {
      get().addLog(`Dismiss error failed: ${(e as Error).message}`, 'err')
    }
  },

  fetchSessionStatus: async () => {
    try {
      const data = await api(`${SESSION}/status`)
      set({ session: data })
    } catch { /* ignore */ }
  },

  doTeleopStart: async (params?) => {
    set({ loading: 'teleop' })
    get().addLog('Starting teleoperation...')
    try {
      await postJson(`${TELEOP}/start`, params || {})
      get().addLog('Teleoperation started', 'ok')
    } catch (e: unknown) {
      get().addLog(`Teleop start failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ loading: null })
    }
  },

  doTeleopStop: async () => {
    get().addLog('Stopping teleoperation...')
    try {
      await postJson(`${TELEOP}/stop`)
      get().addLog('Teleoperation stopped', 'info')
    } catch (e: unknown) {
      get().addLog(`Teleop stop failed: ${(e as Error).message}`, 'err')
    }
  },

  doRecordStart: async (params) => {
    set({ loading: 'record' })
    get().addLog(`Starting recording: ${params.task} (${params.num_episodes} episodes)`)
    try {
      const data = await postJson(`${RECORD}/start`, params)
      get().addLog(`Recording started: ${data.dataset_name}`, 'ok')
    } catch (e: unknown) {
      get().addLog(`Record start failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ loading: null })
    }
  },

  doRecordStop: async () => {
    get().addLog('Stopping recording...')
    try {
      await postJson(`${RECORD}/stop`)
      get().addLog('Recording stopped', 'info')
      get().loadDatasets()
    } catch (e: unknown) {
      get().addLog(`Record stop failed: ${(e as Error).message}`, 'err')
    }
  },

  // -- Episode control ----------------------------------------------------

  doSaveEpisode: async () => {
    get().addLog('Saving episode...')
    try {
      await postJson(`${RECORD}/episode/save`)
    } catch (e: unknown) {
      get().addLog(`Save episode failed: ${(e as Error).message}`, 'err')
    }
  },

  doDiscardEpisode: async () => {
    get().addLog('Discarding episode...')
    try {
      await postJson(`${RECORD}/episode/discard`)
      get().addLog('Discard signal sent', 'info')
    } catch (e: unknown) {
      get().addLog(`Discard episode failed: ${(e as Error).message}`, 'err')
    }
  },

  doSkipReset: async () => {
    get().addLog('Skipping reset wait...')
    try {
      await postJson(`${RECORD}/episode/skip-reset`)
      get().addLog('Skip signal sent', 'ok')
    } catch (e: unknown) {
      get().addLog(`Skip reset failed: ${(e as Error).message}`, 'err')
    }
  },

  // -- Hardware & datasets ------------------------------------------------

  fetchHardwareStatus: async () => {
    try {
      const res = await fetch(`${HARDWARE}/status`)
      if (!res.ok) return
      set({ hardwareStatus: await res.json() })
    } catch { /* ignore */ }
  },

  loadDatasets: async () => {
    try {
      const r = await api(`${DATASETS}`)
      set({ datasets: Array.isArray(r) ? r : r.datasets || [] })
    } catch (e: unknown) {
      get().addLog(`Load datasets failed: ${(e as Error).message}`, 'err')
    }
  },

  loadPolicies: async () => {
    try {
      const r = await api(`${POLICIES}`)
      set({ policies: Array.isArray(r) ? r : r.policies || [] })
    } catch (e: unknown) {
      get().addLog(`Load policies failed: ${(e as Error).message}`, 'err')
    }
  },

  deleteDataset: async (name) => {
    try {
      await api(`${DATASETS}/${encodeURIComponent(name)}`, { method: 'DELETE' })
      get().addLog(`Dataset deleted: ${name}`, 'info')
      get().loadDatasets()
    } catch (e: unknown) {
      get().addLog(`Delete failed: ${(e as Error).message}`, 'err')
    }
  },

  // -- Troubleshooting ----------------------------------------------------

  fetchTroubleshootMap: async () => {
    try {
      const res = await fetch(`${TROUBLESHOOT}/map`)
      if (!res.ok) return
      set({ troubleshootMap: await res.json() })
    } catch { /* ignore */ }
  },

  fetchNetworkInfo: async () => {
    try {
      const res = await fetch(`${SYSTEM}/network`)
      if (!res.ok) return
      set({ networkInfo: await res.json() })
    } catch { /* ignore */ }
  },

  recheckFault: async (faultType, deviceAlias) => {
    try {
      const res = await fetch(`${TROUBLESHOOT}/recheck`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fault_type: faultType, device_alias: deviceAlias }),
      })
      if (!res.ok) return
      const data = await res.json()
      set({ activeFaults: data.faults || [] })
      get().fetchHardwareStatus()
    } catch { /* ignore */ }
  },

  generateSnapshot: async () => {
    const res = await fetch(`${TROUBLESHOOT}/snapshot`, { method: 'POST' })
    return res.json()
  },

  dismissFault: (faultType, deviceAlias) => {
    set((s) => ({
      activeFaults: s.activeFaults.filter(
        (f) => !(f.fault_type === faultType && f.device_alias === deviceAlias),
      ),
    }))
  },

  // -- WebSocket event handler --------------------------------------------

  handleDashboardEvent: (event) => {
    const type = event.type as string

    if (type === 'dashboard.session.state_changed') {
      const prev = get().session
      const newSaved = event.saved_episodes ?? 0
      if (newSaved > prev.saved_episodes && prev.episode_phase !== '') {
        get().addLog(`Episode ${newSaved} saved`, 'ok')
      }
      set({
        session: {
          state: event.state || 'idle',
          episode_phase: event.episode_phase || '',
          saved_episodes: newSaved,
          current_episode: event.current_episode ?? 0,
          target_episodes: event.target_episodes ?? 0,
          total_frames: event.total_frames ?? 0,
          elapsed_seconds: event.elapsed_seconds ?? 0,
          dataset: event.dataset || null,
          rerun_web_port: event.rerun_web_port || 0,
          error: event.error || '',
          calibration_step: event.calibration_step || '',
          calibration_arm: event.calibration_arm || '',
          calibration_positions: event.calibration_positions || null,
          embodiment_owner: event.embodiment_owner || '',
          prepare_stage: event.prepare_stage || '',
        },
      })
      return
    }

    if (type === 'dashboard.fault.detected') {
      const fault: Fault = {
        fault_type: event.fault_type,
        device_alias: event.device_alias,
        message: event.message,
        timestamp: event.timestamp,
      }
      set((s) => ({
        activeFaults: [
          ...s.activeFaults.filter(
            (f) => !(f.fault_type === fault.fault_type && f.device_alias === fault.device_alias),
          ),
          fault,
        ],
      }))
      return
    }

    if (type === 'dashboard.fault.resolved') {
      set((s) => ({
        activeFaults: s.activeFaults.filter(
          (f) => !(f.fault_type === event.fault_type && f.device_alias === event.device_alias),
        ),
      }))
      return
    }

    if (type === 'dashboard.calibration.state_changed') {
      set({ calibration: { state: event.state || 'idle', arm_alias: event.arm_alias || '' } })
      return
    }

    if (type === 'dashboard.hub.progress') {
      set({
        hubProgress: {
          operation: event.operation ?? '',
          progress_percent: event.progress_percent ?? 0,
          total_bytes: event.total_bytes ?? 0,
          done: !!event.done,
        },
      })
    }
  },

  // -- Replay -------------------------------------------------------------

  doReplayStart: async (params) => {
    set({ loading: 'replay' })
    get().addLog(`Starting replay: ${params.dataset_name} ep${params.episode ?? 0}`)
    try {
      await postJson(`${REPLAY}/start`, params)
      get().addLog('Replay started', 'ok')
    } catch (e: unknown) {
      get().addLog(`Replay start failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ loading: null })
    }
  },

  doReplayStop: async () => {
    get().addLog('Stopping replay...')
    try {
      await postJson(`${REPLAY}/stop`)
      get().addLog('Replay stopped', 'info')
    } catch (e: unknown) {
      get().addLog(`Replay stop failed: ${(e as Error).message}`, 'err')
    }
  },

  // -- Training -----------------------------------------------------------

  doTrainStart: async (params) => {
    set({ loading: 'train' })
    get().addLog(`Starting training: ${params.dataset_name} (${params.steps ?? 100000} steps)`)
    try {
      const data = await postJson(`${TRAIN}/start`, params)
      set({ trainJobMessage: data.message || '' })
      get().addLog(data.message || 'Training started', 'ok')
    } catch (e: unknown) {
      get().addLog(`Train start failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ loading: null })
    }
  },

  fetchTrainStatus: async (jobId) => {
    try {
      const data = await api(`${TRAIN}/status/${encodeURIComponent(jobId)}`)
      set({ trainJobMessage: data.message || '' })
    } catch { /* ignore */ }
  },

  fetchTrainDatasets: async () => {
    try {
      await get().loadDatasets()
    } catch { /* ignore */ }
  },

  // -- Hub ----------------------------------------------------------------

  pushDataset: async (name, repoId) => {
    set({ hubLoading: 'pushDataset' })
    get().addLog(`Pushing dataset "${name}" → ${repoId}...`)
    try {
      const data = await postJson(`${HUB}/datasets/push`, { name, repo_id: repoId })
      get().addLog(data.message || 'Dataset pushed', 'ok')
    } catch (e: unknown) {
      get().addLog(`Push dataset failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ hubLoading: null })
    }
  },

  pullDataset: async (repoId, name) => {
    set({ hubLoading: 'pullDataset', hubProgress: null })
    get().addLog(`Downloading dataset from ${repoId}...`)
    try {
      const data = await postJson(`${HUB}/datasets/pull`, { repo_id: repoId, name: name || '' })
      get().addLog(data.message || 'Dataset downloaded', 'ok')
      get().loadDatasets()
    } catch (e: unknown) {
      get().addLog(`Pull dataset failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ hubLoading: null, hubProgress: null })
    }
  },

  pushPolicy: async (name, repoId) => {
    set({ hubLoading: 'pushPolicy' })
    get().addLog(`Pushing policy "${name}" → ${repoId}...`)
    try {
      const data = await postJson(`${HUB}/policies/push`, { name, repo_id: repoId })
      get().addLog(data.message || 'Policy pushed', 'ok')
    } catch (e: unknown) {
      get().addLog(`Push policy failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ hubLoading: null })
    }
  },

  pullPolicy: async (repoId, name) => {
    set({ hubLoading: 'pullPolicy', hubProgress: null })
    get().addLog(`Downloading policy from ${repoId}...`)
    try {
      const data = await postJson(`${HUB}/policies/pull`, { repo_id: repoId, name: name || '' })
      get().addLog(data.message || 'Policy downloaded', 'ok')
      get().loadPolicies()
    } catch (e: unknown) {
      get().addLog(`Pull policy failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ hubLoading: null, hubProgress: null })
    }
  },

  // -- Inference ----------------------------------------------------------

  doInferStart: async (params) => {
    set({ loading: 'infer' })
    get().addLog(`Starting inference...`)
    try {
      await postJson(`${INFER}/start`, params)
      get().addLog('Inference started', 'ok')
    } catch (e: unknown) {
      get().addLog(`Inference start failed: ${(e as Error).message}`, 'err')
    } finally {
      set({ loading: null })
    }
  },

  doInferStop: async () => {
    get().addLog('Stopping inference...')
    try {
      await postJson(`${INFER}/stop`)
      get().addLog('Inference stopped', 'info')
    } catch (e: unknown) {
      get().addLog(`Inference stop failed: ${(e as Error).message}`, 'err')
    }
  },

}))
