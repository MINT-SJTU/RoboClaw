export interface ProviderOption {
  name: string
  label: string
  keywords: string[]
  default_model: string
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
  model?: string
  api_key?: string
  api_base?: string
  extra_headers?: Record<string, string>
  clear_api_key?: boolean
}

export interface ProviderModelsResponse {
  models: string[]
  error?: string
}

export interface ProviderTestPayload extends SaveProviderPayload {
  input?: string
}

export interface ProviderTestResponse {
  ok: boolean
  finish_reason: string
  content?: string
  error?: string
}

async function responseError(response: Response, fallback: string): Promise<Error> {
  const data = await response.json().catch(() => null)
  const detail = data?.detail
  if (detail && typeof detail === 'object') {
    const message = [detail.message, detail.hint].filter(Boolean).join(' ')
    return new Error(message || fallback)
  }
  return new Error(typeof detail === 'string' ? detail : fallback)
}

export async function fetchProviderStatus(): Promise<ProviderStatusResponse> {
  const response = await fetch('/api/system/provider-status')
  if (!response.ok) {
    throw new Error('Failed to load provider status.')
  }
  return response.json()
}

export async function fetchProviderModels(payload: SaveProviderPayload): Promise<ProviderModelsResponse> {
  const response = await fetch('/api/system/provider-models', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await responseError(response, 'Failed to discover provider models.')
  }
  return response.json()
}

export async function testProviderConfig(payload: ProviderTestPayload): Promise<ProviderTestResponse> {
  const response = await fetch('/api/system/provider-test', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await responseError(response, 'Failed to test provider configuration.')
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
    throw await responseError(response, 'Failed to save provider configuration.')
  }
  return response.json()
}
