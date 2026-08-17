import { api, postJson } from '@/shared/api/client'

export interface HfConfigResponse {
  endpoint: string
  masked_token: string
  proxy: string
}

export interface SaveHfConfigPayload {
  endpoint: string
  token: string
  proxy: string
}

export type HfEndpointMode = 'default' | 'mirror' | 'custom'

export async function fetchHfConfig(): Promise<HfConfigResponse> {
  return api('/api/system/hf-config')
}

export async function saveHfConfig(payload: SaveHfConfigPayload): Promise<HfConfigResponse> {
  return postJson('/api/system/hf-config', payload)
}

export function classifyHfEndpoint(endpoint: string): HfEndpointMode {
  if (!endpoint) return 'default'
  if (endpoint === 'https://hf-mirror.com') return 'mirror'
  return 'custom'
}
