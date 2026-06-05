import { useI18n } from '@/i18n'
import { useAuthStore } from '@/shared/lib/authStore'

function _authHeaders(): Record<string, string> {
  const { user, isLoggedIn } = useAuthStore.getState()
  if (!isLoggedIn || !user) return {}
  const token = localStorage.getItem('evo_access_token')
  const headers: Record<string, string> = {
    'x-evo-studio-user': user.phone || user.nickname || '',
  }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

export class ApiError extends Error {
  meta: Record<string, any>
  constructor(code: string, meta: Record<string, any>) {
    const raw = useI18n.getState().t(code as any, meta)
    const fallback = typeof meta.message === 'string' && meta.message ? meta.message : code
    super(raw === code ? fallback : raw)
    this.name = 'ApiError'
    this.meta = meta
  }
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function isNetworkFailure(error: unknown): boolean {
  return error instanceof TypeError || String((error as any)?.message || error).toLowerCase().includes('failed to fetch')
}

export async function api(url: string, opts?: RequestInit) {
  let r: Response
  const method = String(opts?.method || 'GET').toUpperCase()
  const maxAttempts = method === 'GET' ? 3 : 2
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      r = await fetch(url, {
        ...opts,
        headers: { ..._authHeaders(), ...(opts?.headers || {}) },
      })
      break
    } catch (error) {
      if (!isNetworkFailure(error) || attempt === maxAttempts) {
        throw new Error('后端暂时断开，正在恢复。请稍后重试，不需要重新填写参数。')
      }
      await sleep(600 * attempt)
    }
  }
  let j: any
  try {
    j = await r!.json()
  } catch (parseErr) {
    throw new Error(`Failed to parse error response: ${parseErr}`)
  }
  if (!r!.ok) {
    const detail = j.detail
    if (detail && typeof detail === 'object' && detail.code) {
      throw new ApiError(detail.code, detail)
    }
    throw new Error(detail || j.error || j.message || `HTTP ${r!.status}`)
  }
  return j
}

export function postJson(url: string, body?: unknown, opts?: RequestInit) {
  return api(url, {
    ...opts,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(opts?.headers || {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
}

export function patchJson(url: string, body: unknown) {
  return api(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function deleteApi(url: string) {
  return api(url, { method: 'DELETE' })
}
