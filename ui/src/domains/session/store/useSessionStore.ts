import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'

const SESSION = '/api/session'
const TELEOP = '/api/teleop'
const RECORD = '/api/record'
const REPLAY = '/api/replay'
const INFER = '/api/infer'
const CALIBRATION = '/api/calibration'

export type SessionState =
  | 'idle'
  | 'preparing'
  | 'calibrating'
  | 'teleoperating'
  | 'recording'
  | 'replaying'
  | 'inferring'
  | 'stopping'
  | 'error'

export type RecordPhase =
  | 'idle'
  | 'preparing'
  | 'recording'
  | 'save_requested'
  | 'discard_requested'
  | 'resetting'
  | 'skip_reset_requested'
  | 'stopping'
  | 'error'
export type SessionLoading =
  | 'teleop'
  | 'teleop-stop'
  | 'record'
  | 'record-stop'
  | 'replay'
  | 'replay-stop'
  | 'infer'
  | 'infer-stop'
  | 'auto-calibration'
  | 'auto-calibration-stop'

export interface CalibrationBatchItem {
  alias: string
  status: 'pending' | 'running' | 'success' | 'skipped' | 'failed'
  reason: string
  started_at: number | null
  finished_at: number | null
}

export interface SessionStatus {
  state: SessionState
  record_phase: RecordPhase
  record_pending_command: string
  saved_episodes: number
  current_episode: number
  target_episodes: number
  total_frames: number
  elapsed_seconds: number
  dataset: string | null
  rerun_web_port: number
  error: string
  calibration_mode: '' | 'manual' | 'auto'
  calibration_scope: '' | 'single' | 'batch'
  calibration_phase: string
  calibration_current_arm: string
  calibration_index: number
  calibration_total: number
  calibration_results: CalibrationBatchItem[]
  calibration_error: string
  calibration_step: string
  calibration_arm: string
  calibration_positions: Record<string, { min: number; pos: number; max: number }> | null
  embodiment_owner: string
  prepare_stage: string
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

interface SessionStore {
  session: SessionStatus
  loading: SessionLoading | null
  doDismissError: () => Promise<void>
  fetchSessionStatus: () => Promise<void>
  doTeleopStart: (params?: { fps?: number; arms?: string }) => Promise<void>
  doTeleopStop: () => Promise<void>
  doRecordStart: (params: StartRecordingParams) => Promise<void>
  doRecordStop: () => Promise<void>
  doSaveEpisode: () => Promise<void>
  doDiscardEpisode: () => Promise<void>
  doSkipReset: () => Promise<void>
  doReplayStart: (params: { dataset_name: string; episode?: number; fps?: number }) => Promise<void>
  doReplayStop: () => Promise<void>
  doInferStart: (params: {
    checkpoint_path?: string
    num_episodes?: number
    episode_time_s?: number
  }) => Promise<void>
  doInferStop: () => Promise<void>
  doAutoCalibrationStart: () => Promise<void>
  doAutoCalibrationStop: () => Promise<void>
  handleDashboardEvent: (event: any) => void
}

const defaultSession: SessionStatus = {
  state: 'idle',
  record_phase: 'idle',
  record_pending_command: '',
  saved_episodes: 0,
  current_episode: 0,
  target_episodes: 0,
  total_frames: 0,
  elapsed_seconds: 0,
  dataset: null,
  rerun_web_port: 0,
  error: '',
  calibration_mode: '',
  calibration_scope: '',
  calibration_phase: '',
  calibration_current_arm: '',
  calibration_index: 0,
  calibration_total: 0,
  calibration_results: [],
  calibration_error: '',
  calibration_step: '',
  calibration_arm: '',
  calibration_positions: null,
  embodiment_owner: '',
  prepare_stage: '',
}

type StartKind = Exclude<SessionLoading, `${string}-stop`>
type StopKind = Extract<SessionLoading, `${string}-stop`>

export const useSessionStore = create<SessionStore>((set, get) => {
  const runStart = async <T extends object | void>(
    kind: StartKind,
    url: string,
    body?: T,
  ) => {
    set({ loading: kind })
    try {
      await postJson(url, body)
    } finally {
      set({ loading: null })
    }
  }

  const runStop = async (kind: StopKind, url: string) => {
    set({ loading: kind })
    try {
      await postJson(url)
    } finally {
      try {
        await get().fetchSessionStatus()
      } finally {
        set({ loading: null })
      }
    }
  }

  return {
    session: { ...defaultSession },
    loading: null,

    doDismissError: async () => {
      await postJson(`${SESSION}/dismiss-error`)
    },

    fetchSessionStatus: async () => {
      const data = await api(`${SESSION}/status`)
      set({ session: data })
    },

    doTeleopStart: (params) => runStart('teleop', `${TELEOP}/start`, params || {}),
    doTeleopStop: () => runStop('teleop-stop', `${TELEOP}/stop`),

    doRecordStart: (params) => runStart('record', `${RECORD}/start`, params),
    doRecordStop: () => runStop('record-stop', `${RECORD}/stop`),

    doSaveEpisode: async () => {
      await postJson(`${RECORD}/episode/save`)
    },

    doDiscardEpisode: async () => {
      await postJson(`${RECORD}/episode/discard`)
    },

    doSkipReset: async () => {
      await postJson(`${RECORD}/episode/skip-reset`)
    },

    doReplayStart: (params) => runStart('replay', `${REPLAY}/start`, params),
    doReplayStop: () => runStop('replay-stop', `${REPLAY}/stop`),

    doInferStart: (params) => runStart('infer', `${INFER}/start`, params),
    doInferStop: () => runStop('infer-stop', `${INFER}/stop`),

    doAutoCalibrationStart: () => runStart('auto-calibration', `${CALIBRATION}/auto/start`),
    doAutoCalibrationStop: () => runStop('auto-calibration-stop', `${CALIBRATION}/auto/stop`),

    handleDashboardEvent: (event) => {
      if (event.type !== 'dashboard.session.state_changed') {
        return
      }
      set({
        session: {
          state: event.state || 'idle',
          record_phase: event.record_phase || 'idle',
          record_pending_command: event.record_pending_command || '',
          saved_episodes: event.saved_episodes ?? 0,
          current_episode: event.current_episode ?? 0,
          target_episodes: event.target_episodes ?? 0,
          total_frames: event.total_frames ?? 0,
          elapsed_seconds: event.elapsed_seconds ?? 0,
          dataset: event.dataset || null,
          rerun_web_port: event.rerun_web_port || 0,
          error: event.error || '',
          calibration_mode: event.calibration_mode || '',
          calibration_scope: event.calibration_scope || '',
          calibration_phase: event.calibration_phase || '',
          calibration_current_arm: event.calibration_current_arm || '',
          calibration_index: event.calibration_index ?? 0,
          calibration_total: event.calibration_total ?? 0,
          calibration_results: event.calibration_results || [],
          calibration_error: event.calibration_error || '',
          calibration_step: event.calibration_step || '',
          calibration_arm: event.calibration_arm || '',
          calibration_positions: event.calibration_positions || null,
          embodiment_owner: event.embodiment_owner || '',
          prepare_stage: event.prepare_stage || '',
        },
      })
    },
  }
})
