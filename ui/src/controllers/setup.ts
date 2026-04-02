import { create } from 'zustand'
import { postJson } from './api'

const API = '/api/dashboard/setup'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ScannedPort {
  port_id: string
  by_id: string
  by_path: string
  dev: string
  motor_ids: number[]
  delta: number
  moved: boolean
}

export interface ScannedCamera {
  index: number
  by_path: string
  by_id: string
  dev: string
  width: number
  height: number
  preview_url: string | null
}

export interface ConfiguredArm {
  alias: string
  type: string
  port: string
  calibrated: boolean
}

export interface ConfiguredCamera {
  alias: string
  port: string
}

interface SetupStore {
  open: boolean
  setOpen: (v: boolean) => void

  scanning: boolean
  scannedPorts: ScannedPort[]
  scannedCameras: ScannedCamera[]

  motionActive: boolean
  error: string | null

  configuredArms: ConfiguredArm[]
  configuredCameras: ConfiguredCamera[]

  doScan: () => Promise<void>
  doCapturePreview: () => Promise<void>
  startMotion: () => Promise<void>
  pollMotion: () => Promise<void>
  stopMotion: () => Promise<void>

  addArm: (alias: string, armType: string, portId: string) => Promise<void>
  removeArm: (alias: string) => Promise<void>
  renameArm: (alias: string, newAlias: string) => Promise<void>
  addCamera: (alias: string, cameraIndex: number) => Promise<void>
  removeCamera: (alias: string) => Promise<void>

  loadCurrentSetup: () => Promise<void>
}

let motionTimer: ReturnType<typeof setInterval> | null = null

export const useSetup = create<SetupStore>((set, get) => ({
  open: false,
  setOpen: (v) => {
    set({ open: v })
    if (v) get().loadCurrentSetup()
    if (!v && get().motionActive) get().stopMotion()
  },

  scanning: false,
  scannedPorts: [],
  scannedCameras: [],
  motionActive: false,
  error: null,
  configuredArms: [],
  configuredCameras: [],

  doScan: async () => {
    set({ scanning: true, error: null, scannedPorts: [], scannedCameras: [] })
    try {
      const data = await postJson(`${API}/scan`)
      const ports: ScannedPort[] = (data.ports || []).map((p: any) => ({
        port_id: p.by_id || p.dev || '',
        by_id: p.by_id || '',
        by_path: p.by_path || '',
        dev: p.dev || '',
        motor_ids: p.motor_ids || [],
        delta: 0,
        moved: false,
      }))
      const cameras: ScannedCamera[] = (data.cameras || []).map((c: any, i: number) => ({
        index: i,
        by_path: c.by_path || '',
        by_id: c.by_id || '',
        dev: c.dev || '',
        width: c.width || 640,
        height: c.height || 480,
        preview_url: null,
      }))
      set({ scannedPorts: ports, scannedCameras: cameras })
      // Auto-capture previews if cameras found
      if (cameras.length > 0) get().doCapturePreview()
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    } finally {
      set({ scanning: false })
    }
  },

  doCapturePreview: async () => {
    try {
      await postJson(`${API}/camera-previews`)
      set((s) => ({
        scannedCameras: s.scannedCameras.map((c) => ({
          ...c,
          preview_url: `${API}/camera-preview/${c.index}?t=${Date.now()}`,
        })),
      }))
    } catch { /* ignore preview failure */ }
  },

  startMotion: async () => {
    try {
      await postJson(`${API}/motion/start`)
      set({ motionActive: true })
      motionTimer = setInterval(() => get().pollMotion(), 300)
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
  },

  pollMotion: async () => {
    try {
      const res = await fetch(`${API}/motion/poll`)
      if (!res.ok) return
      const data = await res.json()
      const portResults = data.ports || []
      set((s) => ({
        scannedPorts: s.scannedPorts.map((p) => {
          const match = portResults.find((r: any) => r.port_id === p.port_id || r.dev === p.dev)
          if (!match) return p
          return { ...p, delta: match.delta, moved: match.moved }
        }),
      }))
    } catch { /* ignore poll errors */ }
  },

  stopMotion: async () => {
    if (motionTimer) {
      clearInterval(motionTimer)
      motionTimer = null
    }
    set({ motionActive: false })
    try {
      await postJson(`${API}/motion/stop`)
    } catch { /* ignore */ }
  },

  addArm: async (alias, armType, portId) => {
    await postJson(`${API}/arm`, { alias, arm_type: armType, port_id: portId })
    await get().loadCurrentSetup()
  },

  removeArm: async (alias) => {
    await fetch(`${API}/arm/${encodeURIComponent(alias)}`, { method: 'DELETE' })
    await get().loadCurrentSetup()
  },

  renameArm: async (alias, newAlias) => {
    await fetch(`${API}/arm/${encodeURIComponent(alias)}/rename`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_alias: newAlias }),
    })
    await get().loadCurrentSetup()
  },

  addCamera: async (alias, cameraIndex) => {
    await postJson(`${API}/camera`, { alias, camera_index: cameraIndex })
    await get().loadCurrentSetup()
  },

  removeCamera: async (alias) => {
    await fetch(`${API}/camera/${encodeURIComponent(alias)}`, { method: 'DELETE' })
    await get().loadCurrentSetup()
  },

  loadCurrentSetup: async () => {
    try {
      const res = await fetch(`${API}/current`)
      if (!res.ok) return
      const data = await res.json()
      set({
        configuredArms: (data.arms || []).map((a: any) => ({
          alias: a.alias || '',
          type: a.type || '',
          port: a.port || '',
          calibrated: a.calibrated || false,
        })),
        configuredCameras: (data.cameras || []).map((c: any) => ({
          alias: c.alias || c.name || '',
          port: c.port || '',
        })),
      })
    } catch { /* ignore */ }
  },
}))
