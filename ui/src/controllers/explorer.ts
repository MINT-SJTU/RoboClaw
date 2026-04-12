import { create } from 'zustand'
import type { JointTrajectoryPayload } from './curation'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FeatureStat {
  name: string
  dtype: string
  shape: unknown[]
  component_names: string[]
  has_dataset_stats: boolean
  count: number | null
  stats_preview: Record<string, { values: unknown[]; truncated: boolean }>
}

export interface ModalityItem {
  id: string
  label: string
  present: boolean
  detail: string
}

export interface FileInventory {
  total_files: number
  parquet_files: number
  video_files: number
  meta_files: number
  other_files: number
}

export interface ExplorerDashboard {
  dataset: string
  summary: {
    total_episodes: number
    total_frames: number
    fps: number
    robot_type: string
    codebase_version: string
    chunks_size: number
  }
  files: FileInventory
  feature_names: string[]
  feature_stats: FeatureStat[]
  feature_type_distribution: Array<{ name: string; value: number }>
  dataset_stats: {
    row_count: number | null
    features_with_stats: number
    vector_features: number
  }
  modality_summary: ModalityItem[]
  episodes: Array<{ episode_index: number; length: number }>
}

export interface EpisodeDetail {
  episode_index: number
  summary: {
    row_count: number
    fps: number
    duration_s: number
    video_count: number
  }
  sample_rows: Array<Record<string, unknown>>
  joint_trajectory: JointTrajectoryPayload
  videos: Array<{ path: string; url: string; stream: string }>
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface ExplorerStore {
  dashboard: ExplorerDashboard | null
  dashboardLoading: boolean
  dashboardError: string
  selectedEpisodeIndex: number | null
  episodeDetail: EpisodeDetail | null
  episodeLoading: boolean
  episodeError: string

  loadDashboard: (dataset: string) => Promise<ExplorerDashboard>
  selectEpisode: (dataset: string, index: number) => Promise<void>
  clearEpisode: () => void
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const contentType = res.headers.get('content-type') || ''
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  if (!contentType.includes('application/json')) {
    const text = await res.text()
    const preview = text.slice(0, 120).replace(/\s+/g, ' ').trim()
    throw new Error(
      `Expected JSON from ${url}, got ${contentType || 'unknown content type'}${preview ? `: ${preview}` : ''}`,
    )
  }
  return res.json()
}

export const useExplorer = create<ExplorerStore>((set) => ({
  dashboard: null,
  dashboardLoading: false,
  dashboardError: '',
  selectedEpisodeIndex: null,
  episodeDetail: null,
  episodeLoading: false,
  episodeError: '',

  loadDashboard: async (dataset: string) => {
    set({
      dashboard: null,
      dashboardLoading: true,
      dashboardError: '',
      selectedEpisodeIndex: null,
      episodeDetail: null,
      episodeError: '',
    })
    try {
      const dashboard = await fetchJson<ExplorerDashboard>(
        `/api/explorer/dashboard?dataset=${encodeURIComponent(dataset)}`,
      )
      set({ dashboard })
      return dashboard
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load explorer dashboard'
      set({
        dashboardError: message,
      })
      throw error instanceof Error ? error : new Error(message)
    } finally {
      set({ dashboardLoading: false })
    }
  },

  selectEpisode: async (dataset: string, index: number) => {
    if (!dataset) return
    set({
      selectedEpisodeIndex: index,
      episodeDetail: null,
      episodeLoading: true,
      episodeError: '',
    })
    try {
      const detail = await fetchJson<EpisodeDetail>(
        `/api/explorer/episode?dataset=${encodeURIComponent(dataset)}&episode_index=${index}`,
      )
      set({ episodeDetail: detail })
    } catch (error) {
      set({
        episodeError: error instanceof Error ? error.message : 'Failed to load episode detail',
      })
    } finally {
      set({ episodeLoading: false })
    }
  },

  clearEpisode: () => {
    set({ selectedEpisodeIndex: null, episodeDetail: null, episodeError: '' })
  },
}))
