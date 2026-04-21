import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'

const SESSION = '/api/session'
const TELEOP = '/api/teleop'
const RECORD = '/api/record'
const REPLAY = '/api/replay'
const INFER = '/api/infer'

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

export type EpisodePhase = '' | 'recording' | 'saving' | 'resetting' | 'stopping' | 'discarding'
export type SessionLoading =
  | 'teleop'
  | 'teleop-stop'
  | 'record'
  | 'record-stop'
  | 'replay'
  | 'replay-stop'
  | 'infer'
  | 'infer-stop'

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
  handleDashboardEvent: (event: any) => void
}

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

export const useSessionStore = create<SessionStore>((set, get) => {
  const runStart = async <T extends object | void>(
    kind: Exclude<SessionLoading, `${string}-stop`>,
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

  const runStop = async (
    kind: Exclude<SessionLoading, `${string}-stop`>,
    url: string,
  ) => {
    set({ loading: `${kind}-stop` as SessionLoading })
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
    doTeleopStop: () => runStop('teleop', `${TELEOP}/stop`),

    doRecordStart: (params) => runStart('record', `${RECORD}/start`, params),
    doRecordStop: () => runStop('record', `${RECORD}/stop`),

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
    doReplayStop: () => runStop('replay', `${REPLAY}/stop`),

    doInferStart: (params) => runStart('infer', `${INFER}/start`, params),
    doInferStop: () => runStop('infer', `${INFER}/stop`),

    handleDashboardEvent: (event) => {
      if (event.type !== 'dashboard.session.state_changed') {
        return
      }
      const { type: _type, ...payload } = event
      set({ session: { ...defaultSession, ...payload } })
    },
  }
})
