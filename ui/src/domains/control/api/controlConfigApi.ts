import { api, postJson } from '@/shared/api/client'

export interface ControlRecordConfig {
  task: string
  num_episodes: number
  episode_time_s: number
  reset_time_s: number
  dataset_name: string
  fps: number
  use_cameras: boolean
}

export function fetchControlRecordConfig() {
  return api('/api/system/control-record-config') as Promise<ControlRecordConfig>
}

export function saveControlRecordConfig(payload: ControlRecordConfig) {
  return postJson('/api/system/control-record-config', payload)
}
