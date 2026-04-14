import { create } from 'zustand'
import { api, postJson, patchJson, deleteApi } from './api'

const SETUP = '/api/setup'
const DEVICES = '/api/devices'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CatalogCategory {
  id: string
  supported: boolean
}

export interface CatalogModel {
  name: string
  roles: string[]
}

export interface Catalog {
  categories: CatalogCategory[]
  models: Record<string, CatalogModel[]>
}

export interface ScannedPort {
  stable_id: string
  label: string
  by_id: string
  by_path: string
  dev: string
  motor_ids: number[]
  delta: number
  moved: boolean
}

export interface ScannedCamera {
  stable_id: string
  label: string
  index: number
  by_path: string
  by_id: string
  dev: string
  width: number
  height: number
  preview_url: string | null
}

export interface Assignment {
  alias: string
  spec_name: string
  interface_stable_id: string
}

export interface ConfiguredArm {
  alias: string
  type: string
  port: string
  calibrated: boolean
}

export interface ConfiguredCamera {
  alias: string
  side: 'left' | 'right' | ''
  port: string
}

export interface ConfiguredHand {
  alias: string
  type: string
  port: string
}

export type WizardStep = 'select' | 'scan' | 'identify' | 'review'

export function deviceLabel(device: { label?: string; dev?: string }): string {
  return device.label || device.dev || '?'
}

interface SetupStore {
  // Catalog
  catalog: Catalog | null
  loadCatalog: () => Promise<void>

  // Wizard state
  wizardActive: boolean
  wizardStep: WizardStep
  selectedCategory: string
  selectedModel: string
  startWizard: () => void
  cancelWizard: () => Promise<void>
  setCategory: (c: string) => void
  setModel: (m: string) => void
  goToStep: (s: WizardStep) => void

  // Scan
  scanning: boolean
  scannedPorts: ScannedPort[]
  scannedCameras: ScannedCamera[]
  doScan: () => Promise<void>
  doCapturePreview: () => Promise<void>

  // Motion detection
  motionActive: boolean
  startMotion: () => Promise<void>
  pollMotion: () => Promise<void>
  stopMotion: () => Promise<void>

  // Session assign/commit
  assignments: Assignment[]
  sessionAssign: (stableId: string, alias: string, specName: string, side?: 'left' | 'right' | '') => Promise<void>
  sessionUnassign: (alias: string) => Promise<void>
  sessionCommit: () => Promise<void>
  refreshSession: () => Promise<void>

  // Device CRUD
  devices: { arms: ConfiguredArm[]; cameras: ConfiguredCamera[]; hands: ConfiguredHand[] }
  loadDevices: () => Promise<void>
  removeArm: (alias: string) => Promise<void>
  renameArm: (alias: string, newAlias: string) => Promise<void>
  removeCamera: (alias: string) => Promise<void>
  renameCamera: (alias: string, newAlias: string) => Promise<void>
  removeHand: (alias: string) => Promise<void>
  renameHand: (alias: string, newAlias: string) => Promise<void>

  error: string | null
}

let motionTimer: ReturnType<typeof setInterval> | null = null

export const useSetup = create<SetupStore>((set, get) => ({
  catalog: null,
  wizardActive: false,
  wizardStep: 'select' as WizardStep,
  selectedCategory: '',
  selectedModel: '',
  scanning: false,
  scannedPorts: [],
  scannedCameras: [],
  motionActive: false,
  assignments: [],
  devices: { arms: [], cameras: [], hands: [] },
  error: null,

  // -- Catalog ----------------------------------------------------------------

  loadCatalog: async () => {
    try {
      const data = await api(`${DEVICES}/catalog`)
      set({ catalog: data })
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
  },

  // -- Wizard control ---------------------------------------------------------

  startWizard: () => {
    set({
      wizardActive: true,
      wizardStep: 'select',
      selectedCategory: '',
      selectedModel: '',
      scannedPorts: [],
      scannedCameras: [],
      assignments: [],
      error: null,
    })
  },

  cancelWizard: async () => {
    if (get().motionActive) await get().stopMotion()
    try {
      await postJson(`${SETUP}/session/reset`)
    } catch { /* ignore */ }
    set({
      wizardActive: false,
      wizardStep: 'select',
      selectedCategory: '',
      selectedModel: '',
      scannedPorts: [],
      scannedCameras: [],
      assignments: [],
    })
  },

  setCategory: (c) => set({ selectedCategory: c }),
  setModel: (m) => set({ selectedModel: m }),
  goToStep: (s) => set({ wizardStep: s }),

  // -- Scan -------------------------------------------------------------------

  doScan: async () => {
    const model = get().selectedModel
    set({ scanning: true, error: null, scannedPorts: [], scannedCameras: [] })
    try {
      const data = await postJson(`${SETUP}/scan`, { model })
      const ports: ScannedPort[] = (data.ports || []).map((p: any) => ({
        stable_id: p.stable_id || p.by_id || p.dev || '',
        label: p.label || p.dev || '?',
        by_id: p.by_id || '',
        by_path: p.by_path || '',
        dev: p.dev || '',
        motor_ids: p.motor_ids || [],
        delta: 0,
        moved: false,
      }))
      const cameras: ScannedCamera[] = (data.cameras || []).map((c: any, i: number) => ({
        stable_id: c.stable_id || c.by_path || c.by_id || c.dev || '',
        label: c.label || c.dev || '?',
        index: i,
        by_path: c.by_path || '',
        by_id: c.by_id || '',
        dev: c.dev || '',
        width: c.width || 640,
        height: c.height || 480,
        preview_url: null,
      }))
      set({ scannedPorts: ports, scannedCameras: cameras })
      if (cameras.length > 0) get().doCapturePreview()
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    } finally {
      set({ scanning: false })
    }
  },

  doCapturePreview: async () => {
    try {
      await postJson(`${SETUP}/previews`)
      set((s) => ({
        scannedCameras: s.scannedCameras.map((c) => ({
          ...c,
          preview_url: `${SETUP}/previews/${c.index}?t=${Date.now()}`,
        })),
      }))
    } catch { /* ignore */ }
  },

  // -- Motion detection -------------------------------------------------------

  startMotion: async () => {
    if (get().motionActive) return
    if (motionTimer) { clearInterval(motionTimer); motionTimer = null }
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        await postJson(`${SETUP}/motion/start`)
        set({ motionActive: true, error: null })
        motionTimer = setInterval(() => get().pollMotion(), 300)
        return
      } catch (e: unknown) {
        const msg = (e as Error).message || ''
        if (msg.includes('busy') && attempt < 4) {
          await new Promise((r) => setTimeout(r, 800))
          continue
        }
        set({ error: msg })
      }
    }
  },

  pollMotion: async () => {
    try {
      const data = await api(`${SETUP}/motion/poll`)
      const portResults = data.ports || []
      set((s) => {
        let changed = false
        const updated = s.scannedPorts.map((p) => {
          const match = portResults.find(
            (r: any) => r.stable_id === p.stable_id || r.dev === p.dev,
          )
          if (!match) return p
          if (p.delta === match.delta && p.moved === match.moved) return p
          changed = true
          return { ...p, delta: match.delta, moved: match.moved }
        })
        return changed ? { scannedPorts: updated } : {}
      })
    } catch { /* ignore */ }
  },

  stopMotion: async () => {
    if (motionTimer) {
      clearInterval(motionTimer)
      motionTimer = null
    }
    set({ motionActive: false })
    try {
      await postJson(`${SETUP}/motion/stop`)
    } catch { /* ignore */ }
  },

  // -- Session assign/commit --------------------------------------------------

  sessionAssign: async (stableId, alias, specName, side) => {
    set({ error: null })
    try {
      await postJson(`${SETUP}/session/assign`, {
        interface_stable_id: stableId,
        alias,
        spec_name: specName,
        side: side ?? '',
      })
      await get().refreshSession()
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
  },

  sessionUnassign: async (alias) => {
    set({ error: null })
    try {
      await deleteApi(`${SETUP}/session/assign/${encodeURIComponent(alias)}`)
      await get().refreshSession()
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
  },

  sessionCommit: async () => {
    try {
      if (get().motionActive) await get().stopMotion()
      await postJson(`${SETUP}/session/commit`)
      await get().loadDevices()
      set({ wizardActive: false, assignments: [] })
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
  },

  refreshSession: async () => {
    try {
      const data = await api(`${SETUP}/session`)
      set({ assignments: data.assignments || [] })
    } catch { /* ignore */ }
  },

  // -- Device CRUD ------------------------------------------------------------

  loadDevices: async () => {
    try {
      const data = await api(DEVICES)
      set({
        devices: {
          arms: (data.arms || []).map((a: any) => ({
            alias: a.alias || '',
            type: a.type || a.type_name || '',
            port: a.port || a.interface?.by_id || '',
            calibrated: a.calibrated || false,
          })),
          cameras: (data.cameras || []).map((c: any) => ({
            alias: c.alias || c.name || '',
            side: c.side || '',
            port: c.port || c.interface?.by_id || c.interface?.dev || '',
          })),
          hands: (data.hands || []).map((h: any) => ({
            alias: h.alias || '',
            type: h.type || h.type_name || '',
            port: h.port || h.interface?.by_id || '',
          })),
        },
      })
    } catch { /* ignore */ }
  },

  removeArm: async (alias) => {
    await deleteApi(`${DEVICES}/arms/${encodeURIComponent(alias)}`)
    await get().loadDevices()
  },

  renameArm: async (alias, newAlias) => {
    await patchJson(`${DEVICES}/arms/${encodeURIComponent(alias)}`, { new_alias: newAlias })
    await get().loadDevices()
  },

  removeCamera: async (alias) => {
    await deleteApi(`${DEVICES}/cameras/${encodeURIComponent(alias)}`)
    await get().loadDevices()
  },

  renameCamera: async (alias, newAlias) => {
    await patchJson(`${DEVICES}/cameras/${encodeURIComponent(alias)}`, { new_alias: newAlias })
    await get().loadDevices()
  },

  removeHand: async (alias) => {
    await deleteApi(`${DEVICES}/hands/${encodeURIComponent(alias)}`)
    await get().loadDevices()
  },

  renameHand: async (alias, newAlias) => {
    await patchJson(`${DEVICES}/hands/${encodeURIComponent(alias)}`, { new_alias: newAlias })
    await get().loadDevices()
  },
}))
