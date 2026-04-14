/**
 * Shared API helpers for all frontend controllers.
 */

export async function api(url: string, opts?: RequestInit) {
  const r = await fetch(url, opts)
  let j: any
  try {
    j = await r.json()
  } catch {
    throw new Error(`HTTP ${r.status}: ${r.statusText}`)
  }
  if (!r.ok || j.error) {
    const detail = j.detail
    if (detail && typeof detail === 'object' && detail.code) {
      const err = new Error(detail.code)
      ;(err as any).meta = detail
      throw err
    }
    throw new Error(detail || j.error || j.message || `HTTP ${r.status}`)
  }
  return j
}

export function postJson(url: string, body?: unknown) {
  return api(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
