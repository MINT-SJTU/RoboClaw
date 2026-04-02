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
    throw new Error(j.detail || j.error || j.message || `HTTP ${r.status}`)
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
