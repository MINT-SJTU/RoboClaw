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

export interface ExplorerPageState {
  source: ExplorerSource
  datasetIdInput: string
  remoteDatasetSelected: string
  localDatasetInput: string
  localDatasetPathInput: string
  localDatasetPathSelected: string
  localPathDatasetLabel: string
  prepareStatus: string
  prepareError: string
  preparingForQuality: boolean
  activeDatasetRef: ExplorerDatasetRef | null
}

const EXPLORER_SESSION_KEY = 'roboclaw.dataset_explorer_session'

function createDefaultExplorerPageState(): ExplorerPageState {
  return {
    source: 'remote',
    datasetIdInput: '',
    remoteDatasetSelected: '',
    localDatasetInput: '',
    localDatasetPathInput: '',
    localDatasetPathSelected: '',
    localPathDatasetLabel: '',
    prepareStatus: '',
    prepareError: '',
    preparingForQuality: false,
    activeDatasetRef: null,
  }
}

function normalizeExplorerSource(value: unknown): ExplorerSource {
  return value === 'local' || value === 'path' ? value : 'remote'
}

function normalizeString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function normalizeStoredDatasetRef(value: unknown): ExplorerDatasetRef | null {
  if (!value || typeof value !== 'object') {
    return null
  }
  const record = value as Record<string, unknown>
  const source = normalizeExplorerSource(record.source)
  const dataset = normalizeString(record.dataset).trim()
  const path = normalizeString(record.path).trim()
  if (source === 'path') {
    if (!path) {
      return null
    }
    return {
      source,
      dataset: dataset || undefined,
      path,
    }
  }
  if (!dataset) {
    return null
  }
  return { source, dataset }
}

function normalizeStoredExplorerPageState(value: unknown): ExplorerPageState {
  const fallback = createDefaultExplorerPageState()
  if (!value || typeof value !== 'object') {
    return fallback
  }
  const record = value as Record<string, unknown>
  return {
    source: normalizeExplorerSource(record.source),
    datasetIdInput: normalizeString(record.datasetIdInput),
    remoteDatasetSelected: normalizeString(record.remoteDatasetSelected),
    localDatasetInput: normalizeString(record.localDatasetInput),
    localDatasetPathInput: normalizeString(record.localDatasetPathInput),
    localDatasetPathSelected: normalizeString(record.localDatasetPathSelected),
    localPathDatasetLabel: normalizeString(record.localPathDatasetLabel),
    prepareStatus: normalizeString(record.prepareStatus),
    prepareError: '',
    preparingForQuality: false,
    activeDatasetRef: normalizeStoredDatasetRef(record.activeDatasetRef),
  }
}

function getStoredExplorerPageState(): ExplorerPageState {
  if (typeof window === 'undefined') {
    return createDefaultExplorerPageState()
  }
  const raw = window.localStorage.getItem(EXPLORER_SESSION_KEY)
  if (!raw) {
    return createDefaultExplorerPageState()
  }
  try {
    return normalizeStoredExplorerPageState(JSON.parse(raw))
  } catch {
    window.localStorage.removeItem(EXPLORER_SESSION_KEY)
    return createDefaultExplorerPageState()
  }
}

function persistExplorerPageState(state: ExplorerPageState): void {
  if (typeof window === 'undefined') {
    return
  }
  const stored: ExplorerPageState = {
    source: state.source,
    datasetIdInput: state.datasetIdInput,
    remoteDatasetSelected: state.remoteDatasetSelected,
    localDatasetInput: state.localDatasetInput,
    localDatasetPathInput: state.localDatasetPathInput,
    localDatasetPathSelected: state.localDatasetPathSelected,
    localPathDatasetLabel: state.localPathDatasetLabel,
    prepareStatus: state.preparingForQuality ? '' : state.prepareStatus,
    prepareError: '',
    preparingForQuality: false,
    activeDatasetRef: normalizeStoredDatasetRef(state.activeDatasetRef),
  }
  window.localStorage.setItem(EXPLORER_SESSION_KEY, JSON.stringify(stored))
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface ExplorerStore extends ExplorerPageState {
  summary: ExplorerSummary | null
  summaryRefKey: string
  summaryLoading: boolean
  summaryError: string
  dashboard: ExplorerDashboard | null
  dashboardRefKey: string
  dashboardLoading: boolean
  dashboardError: string
  episodePage: ExplorerEpisodePage | null
  episodePageRefKey: string
  episodePageLoading: boolean
  episodePageError: string
  selectedEpisodeIndex: number | null
  episodeDetail: EpisodeDetail | null
  episodeDetailRefKey: string
  episodeLoading: boolean
  episodeError: string

  loadSummary: (ref: ExplorerDatasetRef) => Promise<ExplorerSummary>
  loadDashboard: (ref: ExplorerDatasetRef) => Promise<ExplorerDashboard>
  loadEpisodePage: (ref: ExplorerDatasetRef, page?: number, pageSize?: number) => Promise<void>
  selectEpisode: (ref: ExplorerDatasetRef, index: number) => Promise<void>
  clearEpisode: () => void
  setPageState: (patch: Partial<ExplorerPageState>) => void
  setActiveDatasetRef: (ref: ExplorerDatasetRef | null) => void
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const contentType = res.headers.get('content-type') || ''
  if (!res.ok) {
    const text = await res.text()
    if (contentType.includes('application/json')) {
      let payload: { detail?: unknown; error?: unknown; message?: unknown } | null = null
      try {
        payload = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown }
      } catch {
        payload = null
      }
      const detail = payload?.detail ?? payload?.error ?? payload?.message
      if (typeof detail === 'string' && detail.trim()) {
        throw new Error(detail)
      }
    }
    throw new Error(`HTTP ${res.status}${text ? `: ${text}` : ''}`)
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

export function buildExplorerRefKey(ref: ExplorerDatasetRef | null | undefined): string {
  if (!ref) {
    return ''
  }
  return `${ref.source}|${ref.dataset?.trim() ?? ''}|${ref.path?.trim() ?? ''}`
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

function assertRemoteDatasetMatchesRequest(
  ref: ExplorerDatasetRef,
  payload: { dataset?: string },
  label: string,
): void {
  if (ref.source !== 'remote' || !ref.dataset || !payload.dataset) {
    return
  }
  if (payload.dataset !== ref.dataset) {
    throw new Error(
      `${label} returned '${payload.dataset}' while '${ref.dataset}' was requested`,
    )
  }
}

export const useExplorer = create<ExplorerStore>((set) => ({
  ...getStoredExplorerPageState(),
  summary: null,
  summaryRefKey: '',
  summaryLoading: false,
  summaryError: '',
  dashboard: null,
  dashboardRefKey: '',
  dashboardLoading: false,
  dashboardError: '',
  episodePage: null,
  episodePageRefKey: '',
  episodePageLoading: false,
  episodePageError: '',
  selectedEpisodeIndex: null,
  episodeDetail: null,
  episodeDetailRefKey: '',
  episodeLoading: false,
  episodeError: '',

  setPageState: (patch) => {
    set((state) => {
      const nextPageState = normalizeStoredExplorerPageState({
        ...state,
        ...patch,
      })
      persistExplorerPageState(nextPageState)
      return patch
    })
  },
  setActiveDatasetRef: (ref) => {
    set((state) => {
      const nextPageState = normalizeStoredExplorerPageState({
        ...state,
        activeDatasetRef: ref,
      })
      persistExplorerPageState(nextPageState)
      return { activeDatasetRef: ref }
    })
  },

  loadSummary: async (ref: ExplorerDatasetRef) => {
    const requestKey = buildExplorerRefKey(ref)
    set({
      summary: null,
      summaryRefKey: requestKey,
      summaryLoading: true,
      summaryError: '',
    })
    try {
      const summary = await fetchJson<ExplorerSummary>(
        `/api/explorer/summary?${buildExplorerQuery(ref)}`,
      )
      assertRemoteDatasetMatchesRequest(ref, summary, 'Explorer summary')
      set((state) => (state.summaryRefKey === requestKey ? { summary } : {}))
      return summary
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load explorer summary'
      set((state) => (state.summaryRefKey === requestKey ? { summaryError: message } : {}))
      throw error instanceof Error ? error : new Error(message)
    } finally {
      set((state) => (state.summaryRefKey === requestKey ? { summaryLoading: false } : {}))
    }
  },

  loadDashboard: async (ref: ExplorerDatasetRef) => {
    const requestKey = buildExplorerRefKey(ref)
    set({
      dashboard: null,
      dashboardRefKey: requestKey,
      dashboardLoading: true,
      dashboardError: '',
    })
    try {
      const dashboard = await fetchJson<ExplorerDashboard>(
        `/api/explorer/details?${buildExplorerQuery(ref)}`,
      )
      assertRemoteDatasetMatchesRequest(ref, dashboard, 'Explorer dashboard')
      set((state) => (state.dashboardRefKey === requestKey ? { dashboard } : {}))
      return dashboard
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load explorer dashboard'
      set((state) => (state.dashboardRefKey === requestKey ? { dashboardError: message } : {}))
      throw error instanceof Error ? error : new Error(message)
    } finally {
      set((state) => (state.dashboardRefKey === requestKey ? { dashboardLoading: false } : {}))
    }
  },

  loadEpisodePage: async (ref: ExplorerDatasetRef, page = 1, pageSize = 50) => {
    if (!ref.dataset && !ref.path) return
    const requestKey = `${buildExplorerRefKey(ref)}|page:${page}|size:${pageSize}`
    set({
      episodePage: null,
      episodePageRefKey: requestKey,
      episodePageLoading: true,
      episodePageError: '',
      selectedEpisodeIndex: null,
      episodeDetail: null,
      episodeDetailRefKey: '',
      episodeError: '',
    })
    try {
      const episodePage = await fetchJson<ExplorerEpisodePage>(
        `/api/explorer/episodes?${buildExplorerQuery(ref)}&page=${page}&page_size=${pageSize}`,
      )
      assertRemoteDatasetMatchesRequest(ref, episodePage, 'Explorer episodes')
      set((state) => (state.episodePageRefKey === requestKey ? { episodePage } : {}))
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load episodes'
      set((state) => (state.episodePageRefKey === requestKey ? { episodePageError: message } : {}))
    } finally {
      set((state) => (state.episodePageRefKey === requestKey ? { episodePageLoading: false } : {}))
    }
  },

  selectEpisode: async (ref: ExplorerDatasetRef, index: number) => {
    if (!ref.dataset && !ref.path) return
    const requestKey = `${buildExplorerRefKey(ref)}|episode:${index}`
    set({
      selectedEpisodeIndex: index,
      episodeDetail: null,
      episodeDetailRefKey: requestKey,
      episodeLoading: true,
      episodeError: '',
    })
    try {
      const detail = await fetchJson<EpisodeDetail>(
        `/api/explorer/episode?${buildExplorerQuery(ref)}&episode_index=${index}`,
      )
      set((state) => (state.episodeDetailRefKey === requestKey ? { episodeDetail: detail } : {}))
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load episode detail'
      set((state) => (state.episodeDetailRefKey === requestKey ? { episodeError: message } : {}))
    } finally {
      set((state) => (state.episodeDetailRefKey === requestKey ? { episodeLoading: false } : {}))
    }
  },

  clearEpisode: () => {
    set({
      selectedEpisodeIndex: null,
      episodeDetail: null,
      episodeDetailRefKey: '',
      episodeError: '',
    })
  },
}))
