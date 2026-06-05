export interface DatasetStats {
  total_episodes: number
  total_frames: number
  total_bytes?: number
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

export interface DatasetStorageUsage {
  username: string
  role: string
  quotaBytes: number
  usedBytes: number
  availableBytes: number
  datasetCount: number
  privateBytes: number
  publicBytes: number
  datasets: Array<{
    id: string
    label: string
    visibility: string
    totalBytes: number
    sourceKind: string
    sourceUri: string
    storageMode: string
    canTrain: boolean
    createdAt: string
    ingestedAt: string
    publicationStatus: string
    privateRetentionDays: number | string
    privateExpiresAt: string
  }>
  policy: DataPoolPolicy
}

export interface DataPoolPolicy {
  pendingRetentionDays: number
  privateRetentionDays: number
  billingAlertUsd: number
  billingConfirmUsd: number
  acceptedUploadExtensions: string[]
  recommendedPackaging: string
  privateRetentionByRole: {
    free: number
    contributor: number
    team: number
  }
  publicRetention: string
  quotas: {
    free: number
    contributor: number
    team: number
  }
  user: {
    username: string
    role: string
    quotaBytes: number
    usedBytes: number
    availableBytes: number
  }
}

export interface DatasetStorageStatus {
  provider: string
  configured: boolean
  endpoint: string
  bucket: string
  region: string
  prefix: string
  publicBaseUrl: string
  missingFields: string[]
  clientAvailable: boolean
  layout: {
    pending: string
    approved: string
    previews: string
    redemption: string
  }
  policy: DataPoolPolicy
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
