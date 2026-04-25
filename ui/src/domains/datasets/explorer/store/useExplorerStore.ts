import { create } from 'zustand'
import type { JointTrajectoryPayload } from '@/domains/curation/store/useCurationStore'

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
}

export interface ExplorerSummary {
  dataset: string
  summary: {
    total_episodes: number
    total_frames: number
    fps: number
    robot_type: string
    codebase_version: string
    chunks_size: number
  }
}

export interface ExplorerEpisodePage {
  dataset: string
  page: number
  page_size: number
  total_episodes: number
  total_pages: number
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
  videos: Array<{
    path: string
    url: string
    stream: string
    from_timestamp?: number | null
    to_timestamp?: number | null
  }>
}

export interface DatasetSuggestion {
  id: string
  label?: string
  path?: string
  source?: 'remote' | 'local' | 'path'
}

export type ExplorerSource = 'remote' | 'local' | 'path'

export interface ExplorerDatasetRef {
  source: ExplorerSource
  dataset?: string
  path?: string
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface ExplorerStore {
  summary: ExplorerSummary | null
  summaryLoading: boolean
  summaryError: string
  dashboard: ExplorerDashboard | null
  dashboardLoading: boolean
  dashboardError: string
  episodePage: ExplorerEpisodePage | null
  episodePageLoading: boolean
  episodePageError: string
  selectedEpisodeIndex: number | null
  episodeDetail: EpisodeDetail | null
  episodeLoading: boolean
  episodeError: string

  loadSummary: (ref: ExplorerDatasetRef) => Promise<ExplorerSummary>
  loadDashboard: (ref: ExplorerDatasetRef) => Promise<ExplorerDashboard>
  loadEpisodePage: (ref: ExplorerDatasetRef, page?: number, pageSize?: number) => Promise<void>
  selectEpisode: (ref: ExplorerDatasetRef, index: number) => Promise<void>
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

export async function searchDatasetSuggestions(
  query: string,
  source: ExplorerSource,
  limit = 8,
): Promise<DatasetSuggestion[]> {
  const needle = query.trim()
  if (!needle) {
    return []
  }
  return fetchJson<DatasetSuggestion[]>(
    `/api/explorer/suggest?q=${encodeURIComponent(needle)}&limit=${limit}&source=${encodeURIComponent(source)}`,
  )
}

export async function listExplorerDatasets(
  source: Extract<ExplorerSource, 'local'> = 'local',
): Promise<DatasetSuggestion[]> {
  return fetchJson<DatasetSuggestion[]>(
    `/api/explorer/datasets?source=${encodeURIComponent(source)}`,
  )
}

function buildExplorerQuery(ref: ExplorerDatasetRef): string {
  const params = new URLSearchParams()
  params.set('source', ref.source)
  if (ref.dataset) {
    params.set('dataset', ref.dataset)
  }
  if (ref.path) {
    params.set('path', ref.path)
  }
  return params.toString()
}

export const useExplorer = create<ExplorerStore>((set) => ({
  summary: null,
  summaryLoading: false,
  summaryError: '',
  dashboard: null,
  dashboardLoading: false,
  dashboardError: '',
  episodePage: null,
  episodePageLoading: false,
  episodePageError: '',
  selectedEpisodeIndex: null,
  episodeDetail: null,
  episodeLoading: false,
  episodeError: '',

  loadSummary: async (ref: ExplorerDatasetRef) => {
    set({
      summary: null,
      summaryLoading: true,
      summaryError: '',
    })
    try {
      const summary = await fetchJson<ExplorerSummary>(
        `/api/explorer/summary?${buildExplorerQuery(ref)}`,
      )
      set({ summary })
      return summary
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load explorer summary'
      set({ summaryError: message })
      throw error instanceof Error ? error : new Error(message)
    } finally {
      set({ summaryLoading: false })
    }
  },

  loadDashboard: async (ref: ExplorerDatasetRef) => {
    set({
      dashboard: null,
      dashboardLoading: true,
      dashboardError: '',
    })
    try {
      const dashboard = await fetchJson<ExplorerDashboard>(
        `/api/explorer/details?${buildExplorerQuery(ref)}`,
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

  loadEpisodePage: async (ref: ExplorerDatasetRef, page = 1, pageSize = 50) => {
    if (!ref.dataset && !ref.path) return
    set({
      episodePageLoading: true,
      episodePageError: '',
      selectedEpisodeIndex: null,
      episodeDetail: null,
      episodeError: '',
    })
    try {
      const episodePage = await fetchJson<ExplorerEpisodePage>(
        `/api/explorer/episodes?${buildExplorerQuery(ref)}&page=${page}&page_size=${pageSize}`,
      )
      set({ episodePage })
    } catch (error) {
      set({
        episodePageError: error instanceof Error ? error.message : 'Failed to load episodes',
      })
    } finally {
      set({ episodePageLoading: false })
    }
  },

  selectEpisode: async (ref: ExplorerDatasetRef, index: number) => {
    if (!ref.dataset && !ref.path) return
    set({
      selectedEpisodeIndex: index,
      episodeDetail: null,
      episodeLoading: true,
      episodeError: '',
    })
    try {
      const detail = await fetchJson<EpisodeDetail>(
        `/api/explorer/episode?${buildExplorerQuery(ref)}&episode_index=${index}`,
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
