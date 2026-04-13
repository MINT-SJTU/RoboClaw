export interface ProviderOption {
  name: string
  label: string
  oauth: boolean
  local: boolean
  direct: boolean
  configured: boolean
  api_base: string
  has_api_key: boolean
  masked_api_key: string
  extra_headers: Record<string, string>
}

export interface ProviderStatusResponse {
  default_model: string
  default_provider: string
  active_provider: string | null
  active_provider_configured: boolean
  custom_provider: ProviderOption
  providers: ProviderOption[]
}

export interface SaveProviderPayload {
  provider?: string
  api_key?: string
  api_base?: string
  extra_headers?: Record<string, string>
  clear_api_key?: boolean
}

export async function fetchProviderStatus(): Promise<ProviderStatusResponse> {
  const response = await fetch('/api/system/provider-status')
  if (!response.ok) {
    throw new Error('Failed to load provider status.')
  }
  return response.json()
}

export async function saveProviderConfig(payload: SaveProviderPayload): Promise<ProviderStatusResponse> {
  const response = await fetch('/api/system/provider-config', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error('Failed to save provider configuration.')
  }
  return response.json()
}
