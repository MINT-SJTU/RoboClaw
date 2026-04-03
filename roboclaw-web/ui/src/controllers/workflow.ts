import { create } from 'zustand'

type StageStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error'
const CURRENT_DATASET_KEY = 'roboclaw.current_dataset'

export interface DatasetSummary {
  name: string
  total_episodes: number
  total_frames: number
  fps: number
  robot_type: string
  episode_lengths?: number[]
  features?: string[]
  source_dataset?: string
}

export interface StageState {
  status: StageStatus
  summary: Record<string, unknown> | null
}

interface QualityStage extends StageState {
  selected_validators: string[]
}

export interface WorkflowState {
  version: number
  dataset: string
  stages: {
    quality_validation: QualityStage
    prototype_discovery: StageState
    annotation: StageState & { annotated_episodes: number[] }
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
}

export interface PropagationResults {
  source_episode_index: number | null
  target_count: number
  propagated: PropagationResultItem[]
  published_parquet_path?: string
}

export interface DatasetImportJob {
  job_id: string
  dataset_id: string
  status: 'queued' | 'running' | 'completed' | 'error'
  include_videos: boolean
  message: string
  imported_dataset?: string | null
  local_path?: string | null
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
}

interface WorkflowStore {
  datasets: DatasetSummary[]
  datasetsLoading: boolean
  selectedDataset: string | null
  datasetInfo: DatasetSummary | null
  workflowState: WorkflowState | null
  selectedValidators: string[]
  qualityThresholds: Record<string, number>
  qualityResults: QualityResults | null
  qualityRunning: boolean
  prototypeResults: PrototypeResults | null
  prototypeRunning: boolean
  propagationResults: PropagationResults | null
  datasetImportJob: DatasetImportJob | null
  pollInterval: ReturnType<typeof setInterval> | null
  loadDatasets: () => Promise<void>
  selectDataset: (name: string) => Promise<void>
  importDatasetFromHf: (datasetId: string, includeVideos?: boolean) => Promise<void>
  toggleValidator: (name: string) => void
  setQualityThreshold: (key: string, value: number) => void
  runQualityValidation: () => Promise<void>
  pauseQualityValidation: () => Promise<void>
  resumeQualityValidation: () => Promise<void>
  runPrototypeDiscovery: (clusterCount?: number) => Promise<void>
  loadQualityResults: () => Promise<QualityResults | null>
  loadPrototypeResults: () => Promise<PrototypeResults | null>
  loadPropagationResults: () => Promise<PropagationResults | null>
  deleteQualityResults: () => Promise<void>
  publishQualityParquet: () => Promise<{ path: string; row_count: number }>
  publishTextAnnotationsParquet: () => Promise<{ path: string; row_count: number }>
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

function normalizePrototypeResults(payload: Partial<PrototypeResults> | null): PrototypeResults | null {
  if (!payload) return null
  return {
    candidate_count: payload.candidate_count ?? 0,
    entry_count: payload.entry_count ?? 0,
    cluster_count: payload.cluster_count ?? 0,
    anchor_record_keys: payload.anchor_record_keys ?? [],
    clusters: payload.clusters ?? [],
  }
}

function normalizePropagationResults(
  payload: Partial<PropagationResults> | null,
): PropagationResults | null {
  if (!payload) return null
  return {
    source_episode_index: payload.source_episode_index ?? null,
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
  selectedValidators: ['metadata', 'timing', 'action', 'visual', 'depth', 'ee_trajectory'],
  qualityThresholds: {
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
  qualityResults: null,
  qualityRunning: false,
  prototypeResults: null,
  prototypeRunning: false,
  propagationResults: null,
  datasetImportJob: null,
  pollInterval: null,

  loadDatasets: async () => {
    set({ datasetsLoading: true })
    try {
      const datasets = await fetchJson<DatasetSummary[]>('/api/workflow/datasets')
      set({ datasets })
    } finally {
      set({ datasetsLoading: false })
    }
  },

  selectDataset: async (name: string) => {
    persistDataset(name)
    set({
      selectedDataset: name,
      datasetInfo: null,
      workflowState: null,
      qualityResults: null,
      prototypeResults: null,
      propagationResults: null,
    })
    const info = await fetchJson<DatasetSummary>(
      `/api/workflow/datasets/${encodeURIComponent(name)}`,
    )
    set({ datasetInfo: info })
    await get().refreshState()
  },

  importDatasetFromHf: async (datasetId: string, includeVideos = true) => {
    const payload = await fetchJson<{ job_id: string; status: string }>(
      '/api/workflow/datasets/import-hf',
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
        `/api/workflow/datasets/import-status/${payload.job_id}`,
      )
      set({ datasetImportJob: job })
      if (job.status === 'completed') {
        await get().loadDatasets()
        if (job.imported_dataset) {
          persistDataset(job.imported_dataset)
          await get().selectDataset(job.imported_dataset)
        }
        active = false
      } else if (job.status === 'error') {
        throw new Error(job.message || 'Dataset import failed')
      } else {
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
      }
    }
  },

  toggleValidator: (name: string) => {
    const current = get().selectedValidators
    if (current.includes(name)) {
      set({ selectedValidators: current.filter((validator) => validator !== name) })
      return
    }
    set({ selectedValidators: [...current, name] })
  },

  setQualityThreshold: (key: string, value: number) => {
    set((state) => ({
      qualityThresholds: {
        ...state.qualityThresholds,
        [key]: value,
      },
    }))
  },

  runQualityValidation: async () => {
    const { selectedDataset, selectedValidators, qualityThresholds } = get()
    if (!selectedDataset) return
    set({ qualityRunning: true })
    await fetchJson('/api/workflow/quality-run', {
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
    await fetchJson('/api/workflow/quality-pause', {
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
    await fetchJson('/api/workflow/quality-resume', {
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
    const { selectedDataset } = get()
    if (!selectedDataset) return
    set({ prototypeRunning: true })
    await fetchJson('/api/workflow/prototype-run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: selectedDataset,
        cluster_count: clusterCount ?? null,
      }),
    })
    get().startPolling()
  },

  loadQualityResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = normalizeQualityResults(
      await fetchJson<QualityResults>(
        `/api/workflow/quality-results?dataset=${encodeURIComponent(selectedDataset)}`,
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
        `/api/workflow/prototype-results?dataset=${encodeURIComponent(selectedDataset)}`,
      ),
    )
    set({ prototypeResults: results })
    return results
  },

  loadPropagationResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) return null
    const results = normalizePropagationResults(
      await fetchJson<PropagationResults>(
        `/api/workflow/propagation-results?dataset=${encodeURIComponent(selectedDataset)}`,
      ),
    )
    set({ propagationResults: results })
    return results
  },

  deleteQualityResults: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    await fetchJson<{ status: string }>(
      '/api/workflow/quality-results/delete',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    set({ qualityResults: null, qualityRunning: false })
    await get().refreshState()
  },

  publishQualityParquet: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    const result = await fetchJson<{ path: string; row_count: number }>(
      '/api/workflow/quality-publish',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    await get().loadQualityResults()
    return result
  },

  publishTextAnnotationsParquet: async () => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    const result = await fetchJson<{ path: string; row_count: number }>(
      '/api/workflow/text-annotations-publish',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset }),
      },
    )
    await get().loadPropagationResults()
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
    return `/api/workflow/quality-results.csv?${params.toString()}`
  },

  fetchAnnotationWorkspace: async (episodeIndex: number) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }
    return fetchJson<AnnotationWorkspacePayload>(
      `/api/workflow/annotation-workspace?dataset=${encodeURIComponent(
        selectedDataset,
      )}&episode_index=${episodeIndex}`,
    )
  },

  saveAnnotations: async (episodeIndex, taskContext, annotations) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }

    const saved = await fetchJson<SavedAnnotationsPayload>('/api/workflow/annotations', {
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
    return saved
  },

  runPropagation: async (sourceEpisodeIndex: number) => {
    const { selectedDataset } = get()
    if (!selectedDataset) {
      throw new Error('No dataset selected')
    }

    await fetchJson('/api/workflow/propagation-run', {
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
      `/api/workflow/state?dataset=${encodeURIComponent(selectedDataset)}`,
    )
    set((current) => ({
      workflowState: state,
      selectedValidators:
        state.stages.quality_validation.selected_validators.length > 0
          ? state.stages.quality_validation.selected_validators
          : current.selectedValidators,
    }))

    const qualityStatus = state.stages.quality_validation.status
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

    if (qualityStatus !== 'running' && prototypeStatus !== 'running' && annotationStatus !== 'running') {
      get().stopPolling()
    }
  },

  startPolling: () => {
    const existing = get().pollInterval
    if (existing) return
    const interval = setInterval(() => {
      void get().refreshState()
    }, 1200)
    set({ pollInterval: interval })
  },

  stopPolling: () => {
    const interval = get().pollInterval
    if (!interval) return
    clearInterval(interval)
    set({ pollInterval: null })
  },
}))
