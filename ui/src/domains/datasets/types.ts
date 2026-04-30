export interface DatasetStats {
  total_episodes: number
  total_frames: number
  fps: number
  robot_type: string
  features: string[]
  episode_lengths: number[]
}

export interface DatasetCapabilities {
  can_replay: boolean
  can_train: boolean
  can_delete: boolean
  can_push: boolean
  can_pull: boolean
  can_curate: boolean
}

export interface DatasetRuntime {
  name: string
  repo_id: string
  local_path: string
}

export interface DatasetRef {
  id: string
  kind: 'local' | 'remote'
  label: string
  slug: string
  source_dataset: string
  stats: DatasetStats
  capabilities: DatasetCapabilities
  runtime: DatasetRuntime | null
}

export interface DatasetImportJob {
  job_id: string
  dataset_id: string
  status: 'queued' | 'running' | 'completed' | 'error'
  include_videos: boolean
  message: string
  dataset: DatasetRef | null
  imported_dataset_id?: string | null
  local_path?: string | null
}
