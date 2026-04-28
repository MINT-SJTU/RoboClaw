import { create } from 'zustand'
import type { DatasetImportJob, DatasetRef } from '@/domains/datasets/types'

type StageStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error'
const CURRENT_DATASET_KEY = 'roboclaw.current_dataset'
const WORKFLOW_REFRESH_INTERVAL_MS = 6000
export interface StageState {
  status: StageStatus
  summary: Record<string, unknown> | null
}

interface QualityStage extends StageState {
  selected_validators: string[]
}

export type QualityFilterMode = 'passed' | 'failed' | 'all'

interface PrototypeStage extends StageState {
  quality_filter_mode?: string
  selected_episode_indices?: number[]
}

export interface WorkflowState {
  version: number
  dataset: string
  stages: {
    quality_validation: QualityStage
    prototype_discovery: PrototypeStage
    annotation: StageState & {
      annotated_episodes: number[]
      propagated_source_episodes?: number[]
    }
  }
}

export interface QualityEpisodeResult {
  episode_index: number
  passed: boolean
  score: number
  validators: Record<string, { passed: boolean; score: number }>
  issues?: Array<Record<string, unknown>>
}

export interface QualityResults {
  total: number
  passed: number
  failed: number
  overall_score: number
  selected_validators: string[]
  threshold_overrides?: Record<string, number>
  episodes: QualityEpisodeResult[]
  working_parquet_path?: string
  published_parquet_path?: string
}

export interface QualityDefaults {
  dataset: string
  selected_validators: string[]
  threshold_overrides: Record<string, number>
  profile: {
    fps: number
    median_episode_duration_s: number
    video_resolution?: { width: number; height: number } | null
    visual_streams: string[]
    depth_streams: string[]
    has_action: boolean
    has_state: boolean
    has_gripper: boolean
  }
  checks: Record<string, boolean>
}

export interface PrototypeClusterMember {
  record_key: string
  episode_index: number | null
  distance_to_prototype?: number
  distance_to_barycenter?: number
  quality?: {
    score?: number
    passed?: boolean
  }
}

export interface PrototypeCluster {
  cluster_index: number
  prototype_record_key: string
  anchor_record_key: string
  member_count: number
  average_distance?: number
  anchor_distance_to_barycenter?: number
  members: PrototypeClusterMember[]
}

export interface PrototypeResults {
  candidate_count: number
  entry_count: number
  cluster_count: number
  anchor_record_keys: string[]
  quality_filter_mode?: string
  selected_episode_indices?: number[]
  clusters: PrototypeCluster[]
}

export interface PropagationSpan {
  label?: string
  startTime?: number
  endTime?: number | null
  text?: string
  [key: string]: unknown
}

export interface PropagationResultItem {
  episode_index: number
  spans: PropagationSpan[]
  prototype_score?: number
  alignment_method?: string
  source_episode_index?: number | null
}

export interface PropagationResults {
  source_episode_index: number | null
  source_episode_indices: number[]
  target_count: number
  propagated: PropagationResultItem[]
  published_parquet_path?: string
}

export interface TrainingTaskApplyResult {
  status: string
  path: string
  manifest_path: string
  backup_dir: string
  updated_episode_count: number
  updated_episode_file_count: number
  updated_data_file_count: number
  updated_task_file_count: number
  updated_info_file_count: number
  synced_quality_episode_count?: number
  task_count: number
  unmatched_episode_indices: number[]
}

export interface RemoteWorkflowPrepareResult {
  dataset_id: string
  local_path: string
  dataset_name: string
  display_name?: string
}

export interface LocalDirectorySessionResult {
  dataset_name: string
  display_name: string
  local_path: string
}
export interface AnnotationItem {
  id: string
  label: string
  category: string
  color: string
  startTime: number
  endTime: number | null
  text: string
  tags: string[]
  source: string
}

export interface WorkflowTaskContext {
  label?: string
  text?: string
  joint_name?: string
  time_s?: number
  frame_index?: number | null
  action_value?: number | null
  state_value?: number | null
  source?: string
  [key: string]: unknown
}

export interface JointTrajectoryEntry {
  joint_name: string
  action_name: string
  state_name: string
  action_values: Array<number | null>
  state_values: Array<number | null>
}

export interface JointTrajectoryPayload {
  x_axis_key: string
  x_values: number[]
  time_values: number[]
  frame_values: number[]
  joint_trajectories: JointTrajectoryEntry[]
  sampled_points: number
  total_points: number
}

export interface AnnotationWorkspaceSummary {
  episode_index: number
  record_key: string
  task_value: string
  task_label: string
  fps: number
  robot_type: string
  row_count: number
  start_timestamp: number | null
  end_timestamp: number | null
  duration_s: number
  video_count: number
  quality_status?: 'passed' | 'failed' | 'unvalidated'
  quality_score?: number | null
}

export interface AnnotationVideoClip {
  path: string
  url: string
  stream: string
  from_timestamp: number | null
  to_timestamp: number | null
}

export interface SavedAnnotationsPayload {
  episode_index: number
  task_context: WorkflowTaskContext
  annotations: AnnotationItem[]
  version_number: number
  created_at?: string
  updated_at?: string
}

export interface AnnotationWorkspacePayload {
  episode_index: number
  summary: AnnotationWorkspaceSummary
  videos: AnnotationVideoClip[]
  joint_trajectory: JointTrajectoryPayload
  annotations: SavedAnnotationsPayload
  latest_propagation: PropagationResults | null
  quality?: {
    validated: boolean
    passed: boolean | null
    score: number | null
    failed_validators: string[]
    quality_tags: string[]
    issues: Array<Record<string, unknown>>
  }
}

export interface AlignmentOverviewRow {
  episode_index: number
  record_key: string
  task: string
  task_source?: string | null
  task_is_supplemental?: boolean
  semantic_task_text?: string | null
  quality_passed: boolean
  quality_score: number
  quality_status: 'passed' | 'failed'
  validator_scores: Record<string, number>
  failed_validators: string[]
  issues: Array<Record<string, unknown>>
  alignment_status: 'not_started' | 'annotated' | 'propagated'
  annotation_count: number
  propagated_count: number
  annotation_spans?: AlignmentOverviewSpan[]
  propagation_source_episode_index?: number | null
  propagation_alignment_method?: string | null
  propagation_spans?: AlignmentOverviewSpan[]
  prototype_score?: number | null
  updated_at?: string
}

export interface AlignmentOverviewSpan {
  id?: string | null
  label?: string | null
  text?: string | null
  category?: string | null
  startTime?: number | null
  endTime?: number | null
  source?: string | null
  target_record_key?: string | null
  prototype_score?: number | null
  source_start_time?: number | null
  source_end_time?: number | null
  dtw_start_delay_s?: number | null
  dtw_end_delay_s?: number | null
  duration_delta_s?: number | null
}

export interface AlignmentOverview {
  summary: {
    total_checked: number
    passed_count: number
    failed_count: number
    perfect_ratio: number
    aligned_count: number
    annotated_count: number
    propagated_count: number
    prototype_cluster_count: number
    quality_filter_mode: string
  }
  distribution: {
    issue_types: Array<{ label: string; count: number }>
    alignment_status: Array<{ label: string; count: number }>
  }
  rows: AlignmentOverviewRow[]
}

interface WorkflowStore {
  datasets: DatasetRef[]
  datasetsLoading: boolean
  selectedDataset: string | null
  datasetInfo: DatasetRef | null
  workflowState: WorkflowState | null
  selectedValidators: string[]
  alignmentQualityFilter: QualityFilterMode
  qualityThresholds: Record<string, number>
  qualityDefaults: QualityDefaults | null
  qualityResults: QualityResults | null
  qualityRunning: boolean
  prototypeResults: PrototypeResults | null
  prototypeRunning: boolean
  propagationResults: PropagationResults | null
  alignmentOverview: AlignmentOverview | null
  datasetImportJob: DatasetImportJob | null
  selectedDatasetIsRemotePrepared: boolean
  pollInterval: ReturnType<typeof setInterval> | null
  loadDatasets: () => Promise<void>
  selectDataset: (datasetId: string) => Promise<void>
  importDatasetFromHf: (datasetId: string, includeVideos?: boolean) => Promise<void>
  prepareRemoteDatasetForWorkflow: (
    datasetId: string,
    includeVideos?: boolean,
  ) => Promise<RemoteWorkflowPrepareResult>
  createLocalDirectorySession: (
    files: File[],
    relativePaths: string[],
    displayName?: string,
  ) => Promise<LocalDirectorySessionResult>
  toggleValidator: (name: string) => void
  setAlignmentQualityFilter: (mode: QualityFilterMode) => void
  setQualityThreshold: (key: string, value: number) => void
  loadQualityDefaults: () => Promise<QualityDefaults | null>
  runQualityValidation: () => Promise<void>
  pauseQualityValidation: () => Promise<void>
  resumeQualityValidation: () => Promise<void>
  runPrototypeDiscovery: (clusterCount?: number) => Promise<void>
  loadQualityResults: () => Promise<QualityResults | null>
  loadPrototypeResults: () => Promise<PrototypeResults | null>
  loadPropagationResults: () => Promise<PropagationResults | null>
  loadAlignmentOverview: () => Promise<AlignmentOverview | null>
  deleteQualityResults: () => Promise<void>
  publishQualityParquet: () => Promise<{ path: string; row_count: number }>
  publishTextAnnotationsParquet: () => Promise<{ path: string; row_count: number }>
  applyTextAnnotationsToTrainingTasks: () => Promise<TrainingTaskApplyResult>
  getQualityCsvUrl: (failedOnly?: boolean) => string
  fetchAnnotationWorkspace: (episodeIndex: number) => Promise<AnnotationWorkspacePayload>
  saveAnnotations: (
    episodeIndex: number,
    taskContext: WorkflowTaskContext,
    annotations: AnnotationItem[],
  ) => Promise<SavedAnnotationsPayload>
  runPropagation: (sourceEpisodeIndex: number) => Promise<void>
  refreshState: () => Promise<void>
  startPolling: () => void
  stopPolling: () => void
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

function getStoredDataset(): string | null {
  if (typeof window === 'undefined') {
    return null
  }
  return window.localStorage.getItem(CURRENT_DATASET_KEY)
}

function persistDataset(name: string | null): void {
  if (typeof window === 'undefined') {
    return
  }
  if (!name) {
    window.localStorage.removeItem(CURRENT_DATASET_KEY)
    return
  }
  window.localStorage.setItem(CURRENT_DATASET_KEY, name)
}

function normalizeQualityResults(payload: Partial<QualityResults> | null): QualityResults | null {
  if (!payload) return null
  return {
    total: payload.total ?? 0,
    passed: payload.passed ?? 0,
    failed: payload.failed ?? 0,
    overall_score: payload.overall_score ?? 0,
    selected_validators: payload.selected_validators ?? [],
    threshold_overrides:
      payload.threshold_overrides && typeof payload.threshold_overrides === 'object'
        ? payload.threshold_overrides
        : undefined,
    episodes: payload.episodes ?? [],
    working_parquet_path:
      typeof payload.working_parquet_path === 'string'
        ? payload.working_parquet_path
        : undefined,
    published_parquet_path:
      typeof payload.published_parquet_path === 'string'
        ? payload.published_parquet_path
        : undefined,
  }
}

function normalizeQualityFilterMode(value: unknown): QualityFilterMode {
  return value === 'failed' || value === 'all' ? value : 'passed'
}

function normalizePrototypeResults(payload: Partial<PrototypeResults> | null): PrototypeResults | null {
  if (!payload) return null
  return {
    candidate_count: payload.candidate_count ?? 0,
    entry_count: payload.entry_count ?? 0,
    cluster_count: payload.cluster_count ?? 0,
    anchor_record_keys: payload.anchor_record_keys ?? [],
    quality_filter_mode: normalizeQualityFilterMode(payload.quality_filter_mode),
    selected_episode_indices: payload.selected_episode_indices ?? [],
    clusters: payload.clusters ?? [],
  }
}

function normalizePropagationResults(
  payload: Partial<PropagationResults> | null,
): PropagationResults | null {
  if (!payload) return null
  return {
    source_episode_index: payload.source_episode_index ?? null,
    source_episode_indices: payload.source_episode_indices ?? [],
    target_count: payload.target_count ?? 0,
    propagated: payload.propagated ?? [],
    published_parquet_path:
      typeof payload.published_parquet_path === 'string'
        ? payload.published_parquet_path
        : undefined,
  }
}

export const useWorkflow = create<WorkflowStore>((set, get) => ({
  datasets: [],
  datasetsLoading: false,
  selectedDataset: getStoredDataset(),
  datasetInfo: null,
  workflowState: null,
  selectedValidators: ['metadata', 'timing', 'action', 'visual', 'ee_trajectory'],
  alignmentQualityFilter: 'passed',
  qualityThresholds: {
    metadata_require_info_json: 1.0,
    metadata_require_episode_metadata: 1.0,
    metadata_require_data_files: 1.0,
    metadata_require_videos: 1.0,
    metadata_require_task_description: 1.0,
    metadata_min_duration_s: 1.0,
    timing_min_monotonicity: 0.99,
    timing_max_interval_cv: 0.05,
    timing_min_frequency_hz: 20.0,
    timing_max_gap_ratio: 0.01,
    timing_min_frequency_consistency: 0.98,
    action_static_threshold: 0.001,
    action_max_all_static_s: 3.0,
    action_max_key_static_s: 5.0,
    action_max_velocity_rad_s: 3.14,
    action_min_duration_s: 1.0,
    action_max_nan_ratio: 0.01,
    visual_min_resolution_width: 640.0,
    visual_min_resolution_height: 480.0,
    visual_min_frame_rate: 20.0,
    visual_frame_rate_tolerance: 2.0,
    visual_color_shift_max: 0.10,
    visual_overexposure_ratio_max: 0.05,
    visual_underexposure_ratio_max: 0.10,
    visual_abnormal_black_ratio_max: 0.95,
    visual_abnormal_white_ratio_max: 0.95,
    visual_min_video_count: 1.0,
    visual_min_accessible_ratio: 1.0,
    depth_min_stream_count: 0.0,
    depth_min_accessible_ratio: 1.0,
    depth_invalid_pixel_max: 0.10,
    depth_continuity_min: 0.90,
    ee_min_event_count: 1.0,
    ee_min_gripper_span: 0.05,
  },
  qualityDefaults: null,
  qualityResults: null,
  qualityRunning: false,
  prototypeResults: null,
  prototypeRunning: false,
  propagationResults: null,
  alignmentOverview: null,
  datasetImportJob: null,
  selectedDatasetIsRemotePrepared: false,
  pollInterval: null,

  loadDatasets: async () => {
    set({ datasetsLoading: true })
    try {
      const datasets = await fetchJson<DatasetRef[]>('/api/curation/datasets')
      set({ datasets })
    } finally {
      set({ datasetsLoading: false })
    }
  },

  selectDataset: async (datasetId: string) => {
    persistDataset(datasetId)
    set({
      selectedDataset: datasetId,
      datasetInfo: null,
      workflowState: null,
      qualityDefaults: null,
      qualityResults: null,
      prototypeResults: null,
      propagationResults: null,
      alignmentOverview: null,
      selectedDatasetIsRemotePrepared: false,
    })
    const info = await fetchJson<DatasetRef>(
      `/api/curation/datasets/${encodeURIComponent(datasetId)}`,
    )
    set({ datasetInfo: info })
    try {
      await get().loadQualityDefaults()
    } catch (error) {
      console.warn('Failed to load quality defaults', error)
    }
    try {
      await get().refreshState()
      set({ selectedDatasetIsRemotePrepared: true })
    } catch {
      set({ workflowState: null, selectedDatasetIsRemotePrepared: false })
    }
  },

  importDatasetFromHf: async (datasetId: string, includeVideos = true) => {
    const payload = await fetchJson<DatasetImportJob>(
      '/api/curation/datasets/import-hf',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          include_videos: includeVideos,
        }),
      },
    )

    let active = true
    while (active) {
      const job = await fetchJson<DatasetImportJob>(
        `/api/curation/datasets/import-status/${payload.job_id}`,
      )
      set({ datasetImportJob: job })
      if (job.status === 'completed') {
        await get().loadDatasets()
        if (job.imported_dataset_id) {
          persistDataset(job.imported_dataset_id)
          await get().selectDataset(job.imported_dataset_id)
        }
        active = false
      } else if (job.status === 'error') {
        throw new Error(job.message || 'Dataset import failed')
      } else {
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
      }
    }
  },

  prepareRemoteDatasetForWorkflow: async (datasetId: string, includeVideos = false) => {
    const payload = await fetchJson<RemoteWorkflowPrepareResult>('/api/explorer/prepare-remote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        include_videos: includeVideos,
      }),
    })
    await get().loadDatasets()
    persistDataset(payload.dataset_name)
    await get().selectDataset(payload.dataset_name)
    set({ selectedDatasetIsRemotePrepared: true })
    return payload
  },

  createLocalDirectorySession: async (files, relativePaths, displayName) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    relativePaths.forEach((path) => form.append('relative_paths', path))
    if (displayName) {
      form.append('display_name', displayName)
    }
    const response = await fetch('/api/explorer/local-directory-session', {
      method: 'POST',
      body: form,
    })
    if (!response.ok) {
      throw new Error(await response.text())
    }
    const payload = (await response.json()) as LocalDirectorySessionResult
    await get().loadDatasets()
    persistDataset(payload.dataset_name)
    await get().selectDataset(payload.dataset_name)
    return payload
  },

  toggleValidator: (name: string) => {
    const current = get().selectedValidators
    if (current.includes(name)) {
      set({ selectedValidators: current.filter((validator) => validator !== name) })
      return
    }
    set({ selectedValidators: [...current, name] })
  },

  setAlignmentQualityFilter: (mode) => {
    set({ alignmentQualityFilter: mode })
  },

  setQualityThreshold: (key: string, value: number) => {
    set((state) => ({
      qualityThresholds: {
        ...state.qualityThresholds,
        [key]: value,
      },
    }))
  },

  loadQualityDefaults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const defaults = await fetchJson<QualityDefaults>(
      `/api/curation/quality-defaults?dataset=${encodeURIComponent(selectedDataset)}`,
    )
    set((state) => ({
      qualityDefaults: defaults,
      selectedValidators:
        defaults.selected_validators.length > 0
          ? defaults.selected_validators
          : state.selectedValidators,
      qualityThresholds: {
        ...state.qualityThresholds,
        ...defaults.threshold_overrides,
      },
    }))
    return defaults
  },

  runQualityValidation: async () => {
    const { selectedDataset, selectedValidators, qualityThresholds } = get()
    if (!selectedDataset) return
    set({ qualityRunning: true })
    await fetchJson('/api/curation/quality-run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        selected_validators: selectedValidators,
        threshold_overrides: qualityThresholds,
      }),
    })
    get().startPolling()
  },

  pauseQualityValidation: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    await fetchJson('/api/curation/quality-pause', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: selectedDataset }),
    })
    get().startPolling()
  },

  resumeQualityValidation: async () => {
    const { selectedDataset, selectedValidators, qualityThresholds } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    set({ qualityRunning: true })
    await fetchJson('/api/curation/quality-resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        selected_validators: selectedValidators,
        threshold_overrides: qualityThresholds,
      }),
    })
    get().startPolling()
  },

  runPrototypeDiscovery: async (clusterCount?: number) => {
    const { selectedDataset, qualityResults, alignmentQualityFilter } = get()
    if (!selectedDataset) return
    const episodes = qualityResults?.episodes || []
    const selectedEpisodeIndices = episodes
      .filter((episode) => {
        if (alignmentQualityFilter === 'all') return true
        return alignmentQualityFilter === 'passed' ? episode.passed : !episode.passed
      })
      .map((episode) => episode.episode_index)
    set({ prototypeRunning: true })
    await fetchJson('/api/curation/prototype-run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        cluster_count: clusterCount ?? null,
        episode_indices: selectedEpisodeIndices,
        quality_filter_mode: alignmentQualityFilter,
      }),
    })
    get().startPolling()
  },

  loadQualityResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = normalizeQualityResults(
      await fetchJson<QualityResults>(
        `/api/curation/quality-results?dataset=${encodeURIComponent(selectedDataset)}`,
      ),
    )
    set((state) => ({
      qualityResults: results,
      selectedValidators:
        results?.selected_validators && results.selected_validators.length > 0
          ? results.selected_validators
          : state.selectedValidators,
      qualityThresholds:
        results?.threshold_overrides && Object.keys(results.threshold_overrides).length > 0
          ? {
              ...state.qualityThresholds,
              ...results.threshold_overrides,
            }
          : state.qualityThresholds,
    }))
    return results
  },

  loadPrototypeResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = normalizePrototypeResults(
      await fetchJson<PrototypeResults>(
        `/api/curation/prototype-results?dataset=${encodeURIComponent(selectedDataset)}`,
      ),
    )
    set({
      prototypeResults: results,
      ...(results?.quality_filter_mode
        ? { alignmentQualityFilter: normalizeQualityFilterMode(results.quality_filter_mode) }
        : {}),
    })
    return results
  },

  loadPropagationResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = normalizePropagationResults(
      await fetchJson<PropagationResults>(
        `/api/curation/propagation-results?dataset=${encodeURIComponent(selectedDataset)}`,
      ),
    )
    set({ propagationResults: results })
    return results
  },

  loadAlignmentOverview: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = await fetchJson<AlignmentOverview>(
      `/api/curation/alignment-overview?dataset=${encodeURIComponent(selectedDataset)}`,
    )
    set({ alignmentOverview: results })
    return results
  },

  deleteQualityResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    await fetchJson<{ status: string }>(
      `/api/curation/quality-results?dataset=${encodeURIComponent(selectedDataset)}`,
      {
        method: 'DELETE',
      },
    )
    set({ qualityResults: null, qualityRunning: false, prototypeResults: null, alignmentOverview: null })
    await get().refreshState()
  },

  publishQualityParquet: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    const result = await fetchJson<{ path: string; row_count: number }>(
      '/api/curation/quality-publish',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    await get().loadQualityResults()
    await get().loadAlignmentOverview()
    return result
  },

  publishTextAnnotationsParquet: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    const result = await fetchJson<{ path: string; row_count: number }>(
      '/api/curation/text-annotations-publish',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    await get().loadPropagationResults()
    await get().loadAlignmentOverview()
    return result
  },

  applyTextAnnotationsToTrainingTasks: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    const result = await fetchJson<TrainingTaskApplyResult>(
      '/api/curation/text-annotations-apply',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    await get().loadQualityResults()
    await get().loadAlignmentOverview()
    return result
  },

  getQualityCsvUrl: (failedOnly = false) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      return ''
    }
    const params = new URLSearchParams({
      dataset: selectedDataset,
    })
    if (failedOnly) {
      params.set('failed_only', 'true')
    }
    return `/api/curation/quality-results.csv?${params.toString()}`
  },

  fetchAnnotationWorkspace: async (episodeIndex: number) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    return fetchJson<AnnotationWorkspacePayload>(
      `/api/curation/annotation-workspace?dataset=${encodeURIComponent(
        selectedDataset,
      )}&episode_index=${episodeIndex}`,
    )
  },

  saveAnnotations: async (episodeIndex, taskContext, annotations) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }

    const saved = await fetchJson<SavedAnnotationsPayload>('/api/curation/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        episode_index: episodeIndex,
        task_context: taskContext,
        annotations,
      }),
    })

    await get().refreshState()
    await get().loadAlignmentOverview()
    return saved
  },

  runPropagation: async (sourceEpisodeIndex: number) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }

    await fetchJson('/api/curation/propagation-run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        source_episode_index: sourceEpisodeIndex,
      }),
    })
    get().startPolling()
  },

  refreshState: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return

    const state = await fetchJson<WorkflowState>(
      `/api/curation/state?dataset=${encodeURIComponent(selectedDataset)}`,
    )
    const qualityStatus = state.stages.quality_validation.status
    const savedQualityValidators = state.stages.quality_validation.selected_validators

    set((current) => ({
      workflowState: state,
      selectedDatasetIsRemotePrepared: true,
      alignmentQualityFilter: normalizeQualityFilterMode(
        state.stages.prototype_discovery.quality_filter_mode || current.alignmentQualityFilter,
      ),
      selectedValidators:
        ['running', 'paused', 'completed'].includes(qualityStatus) && savedQualityValidators.length > 0
          ? savedQualityValidators
          : current.selectedValidators,
    }))

    const prototypeStatus = state.stages.prototype_discovery.status
    const annotationStatus = state.stages.annotation.status

    if (qualityStatus === 'completed') {
      await get().loadQualityResults()
      set({ qualityRunning: false })
    } else if (qualityStatus === 'running') {
      await get().loadQualityResults()
      set({ qualityRunning: true })
    } else if (qualityStatus === 'paused') {
      await get().loadQualityResults()
      set({ qualityRunning: false })
    } else if (qualityStatus === 'idle') {
      set({ qualityResults: null, qualityRunning: false })
    } else if (qualityStatus === 'error') {
      set({ qualityRunning: false })
    }

    if (prototypeStatus === 'completed') {
      await get().loadPrototypeResults()
      set({ prototypeRunning: false })
    } else if (prototypeStatus === 'error') {
      set({ prototypeRunning: false })
    }

    if (
      annotationStatus === 'completed'
      || state.stages.annotation.annotated_episodes.length > 0
    ) {
      await get().loadPropagationResults()
    }

    if (qualityStatus === 'completed' || qualityStatus === 'paused' || qualityStatus === 'running') {
      await get().loadAlignmentOverview()
    }

    if (qualityStatus !== 'running' && prototypeStatus !== 'running' && annotationStatus !== 'running') {
      get().stopPolling()
    }
  },

  startPolling: () => {
    const existing = get().pollInterval
    if (existing) return
    const interval = setInterval(() => {
      void get().refreshState()
    }, WORKFLOW_REFRESH_INTERVAL_MS)
    set({ pollInterval: interval })
  },

  stopPolling: () => {
    const interval = get().pollInterval
    if (!interval) return
    clearInterval(interval)
    set({ pollInterval: null })
  },
}))
