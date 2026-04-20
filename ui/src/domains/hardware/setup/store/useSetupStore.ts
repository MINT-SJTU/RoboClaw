import { create } from 'zustand'
import { api, postJson, patchJson, deleteApi } from '@/shared/api/client'

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
  fps: number
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

export interface PermissionStatus {
  serial: { ok: boolean; count: number }
  camera: { ok: boolean; count: number }
  platform: string
  fixed?: boolean
  hint?: string
}

type SetupPhase = 'idle' | 'discovering' | 'assigning' | 'identifying' | 'committed'

interface SetupCandidate {
  stable_id: string
  interface_type: string
  label?: string
  by_id?: string
  by_path?: string
  dev?: string
  motor_ids?: number[]
  width?: number
  height?: number
  fps?: number
}

interface SetupSessionPayload {
  phase: SetupPhase
  model: string
  candidates: SetupCandidate[]
  assignments: Assignment[]
  unassigned: string[]
  busy: boolean
  busy_reason: string
}

interface SetupStore {
  // Catalog
  catalog: Catalog | null
  loadCatalog: () => Promise<void>

  // Permissions
  permissions: PermissionStatus | null
  permFixing: boolean
  checkPermissions: () => Promise<PermissionStatus | null>
  fixPermissions: () => Promise<void>

  // Wizard state
  wizardActive: boolean
  wizardStep: WizardStep
  wizardRequested: boolean
  sessionPhase: SetupPhase
  busy: boolean
  busyReason: string
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
  sessionDismiss: (stableId: string) => Promise<void>
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

function clearMotionTimer(): void {
  if (!motionTimer) {
    return
  }
  clearInterval(motionTimer)
  motionTimer = null
}

function syncMotionTimer(phase: SetupPhase, pollMotion: () => Promise<void>): void {
  if (phase !== 'identifying') {
    clearMotionTimer()
    return
  }
  if (motionTimer) {
    return
  }
  motionTimer = window.setInterval(() => {
    void pollMotion()
  }, 300)
}

function isSessionActive(session: SetupSessionPayload): boolean {
  return (
    session.phase !== 'idle'
    || session.assignments.length > 0
    || session.candidates.length > 0
  )
}

function deriveWizardStep(
  currentStep: WizardStep,
  session: SetupSessionPayload,
): WizardStep {
  if (session.phase === 'identifying') {
    return 'identify'
  }
  if (
    session.phase === 'committed'
    || (session.assignments.length > 0 && session.unassigned.length === 0)
  ) {
    return 'review'
  }
  if (session.phase === 'assigning' || session.phase === 'discovering') {
    if (currentStep === 'identify') {
      return 'identify'
    }
    if (currentStep === 'review' && session.unassigned.length > 0) {
      return 'identify'
    }
    return session.candidates.length > 0 ? 'scan' : 'select'
  }
  return 'select'
}

interface ClearableState {
  sessionPhase: SetupPhase
  busy: boolean
  assignments: Assignment[]
  scannedPorts: ScannedPort[]
  scannedCameras: ScannedCamera[]
}

function isSessionCleared(state: ClearableState): boolean {
  return (
    state.sessionPhase === 'idle'
    && !state.busy
    && state.assignments.length === 0
    && state.scannedPorts.length === 0
    && state.scannedCameras.length === 0
  )
}

const CLEARED_WIZARD_STATE = {
  wizardRequested: false,
  wizardActive: false,
  wizardStep: 'select' as WizardStep,
  selectedCategory: '',
  selectedModel: '',
}

function buildScannedPorts(
  candidates: SetupCandidate[],
  currentPorts: ScannedPort[],
): ScannedPort[] {
  return candidates
    .filter((candidate) => candidate.interface_type === 'serial')
    .map((candidate) => {
      const existing = currentPorts.find((port) => port.stable_id === candidate.stable_id)
      return {
        stable_id: candidate.stable_id || candidate.by_id || candidate.dev || '',
        label: candidate.label || candidate.dev || '?',
        by_id: candidate.by_id || '',
        by_path: candidate.by_path || '',
        dev: candidate.dev || '',
        motor_ids: candidate.motor_ids || [],
        delta: existing?.delta ?? 0,
        moved: existing?.moved ?? false,
      }
    })
}

function findCategoryForModel(catalog: Catalog | null, modelName: string): string {
  if (!catalog || !modelName) {
    return ''
  }
  for (const [category, models] of Object.entries(catalog.models)) {
    if (models.some((model) => model.name === modelName)) {
      return category
    }
  }
  return ''
}

function buildScannedCameras(
  candidates: SetupCandidate[],
  currentCameras: ScannedCamera[],
): ScannedCamera[] {
  return candidates
    .filter((candidate) => candidate.interface_type === 'video')
    .map((candidate, index) => {
      const existing = currentCameras.find((camera) => camera.stable_id === candidate.stable_id)
      return {
        stable_id: candidate.stable_id || candidate.by_path || candidate.by_id || candidate.dev || '',
        label: candidate.label || candidate.dev || '?',
        index,
        by_path: candidate.by_path || '',
        by_id: candidate.by_id || '',
        dev: candidate.dev || '',
        width: candidate.width || 640,
        height: candidate.height || 480,
        fps: candidate.fps || existing?.fps || 0,
        preview_url: existing?.preview_url ?? null,
      }
    })
}

export const useSetup = create<SetupStore>((set, get) => ({
  catalog: null,
  permissions: null,
  permFixing: false,
  wizardActive: false,
  wizardStep: 'select' as WizardStep,
  wizardRequested: false,
  sessionPhase: 'idle',
  busy: false,
  busyReason: '',
  selectedCategory: '',
  selectedModel: '',
  scanning: false,
  scannedPorts: [],
  scannedCameras: [],
  motionActive: false,
  assignments: [],
  devices: { arms: [], cameras: [], hands: [] },
  error: null,

  // -- Permissions --------------------------------------------------------------

  checkPermissions: async () => {
    try {
      const data: PermissionStatus = await api(`${SETUP}/permissions`)
      set({ permissions: data, error: null })
      return data
    } catch (e: unknown) {
      set({ error: (e as Error).message })
      return null
    }
  },

  fixPermissions: async () => {
    set({ permFixing: true })
    try {
      const data: PermissionStatus = await postJson(`${SETUP}/permissions/fix`)
      set({ permissions: data, error: null })
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    } finally {
      set({ permFixing: false })
    }
  },

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
      wizardRequested: true,
      wizardActive: true,
      wizardStep: 'select',
      selectedCategory: '',
      selectedModel: '',
      scannedPorts: [],
      scannedCameras: [],
      assignments: [],
      error: null,
    })
    void get().refreshSession()
  },

  cancelWizard: async () => {
    clearMotionTimer()
    set({ error: null })
    try {
      await postJson(`${SETUP}/session/reset`)
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
    await get().refreshSession()
    if (isSessionCleared(get())) {
      set(CLEARED_WIZARD_STATE)
    }
  },

  setCategory: (c) => set({ selectedCategory: c }),
  setModel: (m) => set({ selectedModel: m }),
  goToStep: (s) => set({ wizardStep: s }),

  // -- Scan -------------------------------------------------------------------

  doScan: async () => {
    const model = get().selectedModel
    set({
      scanning: true,
      error: null,
      wizardRequested: true,
      wizardStep: 'scan',
      scannedPorts: [],
      scannedCameras: [],
    })
    try {
      const data = await postJson(`${SETUP}/scan`, { model })
      const portCandidates: SetupCandidate[] = (data.ports || []).map((p: any) => ({
        ...p, interface_type: 'serial',
      }))
      const cameraCandidates: SetupCandidate[] = (data.cameras || []).map((c: any) => ({
        ...c, interface_type: 'video',
      }))
      const ports = buildScannedPorts(portCandidates, [])
      const cameras = buildScannedCameras(cameraCandidates, [])
      set({ scannedPorts: ports, scannedCameras: cameras, error: null })
      if (cameras.length > 0) {
        await get().doCapturePreview()
      }
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    } finally {
      set({ scanning: false })
    }
    await get().refreshSession()
  },

  doCapturePreview: async () => {
    try {
      const previews = await postJson(`${SETUP}/previews`)
      const ts = Date.now()
      const byStableId = new Map<string, { url: string; width?: number; height?: number }>()
      ;(previews || []).forEach((preview: any) => {
        if (preview.stable_id && preview.preview_url) {
          byStableId.set(preview.stable_id, {
            url: `${preview.preview_url}?t=${ts}`,
            width: typeof preview.width === 'number' ? preview.width : undefined,
            height: typeof preview.height === 'number' ? preview.height : undefined,
          })
        }
      })
      set((s) => ({
        scannedCameras: s.scannedCameras.map((c) => {
          const hit = byStableId.get(c.stable_id)
          if (!hit) return { ...c, preview_url: null }
          return {
            ...c,
            preview_url: hit.url,
            width: hit.width ?? c.width,
            height: hit.height ?? c.height,
          }
        }),
      }))
    } catch (e: unknown) {
      set((s) => ({
        error: (e as Error).message,
        scannedCameras: s.scannedCameras.map((c) => ({
          ...c,
          preview_url: null,
        })),
      }))
    }
  },

  // -- Motion detection -------------------------------------------------------

  startMotion: async () => {
    if (get().motionActive) return
    clearMotionTimer()
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        await postJson(`${SETUP}/motion/start`)
        set({ error: null })
        await get().refreshSession()
        return
      } catch (e: unknown) {
        const msg = (e as Error).message || ''
        if (msg.includes('busy') && attempt < 4) {
          await new Promise((r) => setTimeout(r, 800))
          continue
        }
        set({ error: msg })
        await get().refreshSession()
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
    } catch (e: unknown) {
      clearMotionTimer()
      set({ error: (e as Error).message })
      await get().refreshSession()
    }
  },

  stopMotion: async () => {
    clearMotionTimer()
    try {
      await postJson(`${SETUP}/motion/stop`)
      set({ error: null })
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
    await get().refreshSession()
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

  sessionDismiss: async (stableId) => {
    set({ error: null })
    try {
      await postJson(`${SETUP}/session/dismiss`, {
        interface_stable_id: stableId,
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
    set({ error: null })
    try {
      await postJson(`${SETUP}/session/commit`)
      await get().loadDevices()
      if (get().error) {
        await get().refreshSession()
        return
      }
      await postJson(`${SETUP}/session/reset`)
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
    await get().refreshSession()
    if (isSessionCleared(get())) {
      set(CLEARED_WIZARD_STATE)
    }
  },

  refreshSession: async () => {
    try {
      const data = await api(`${SETUP}/session`) as SetupSessionPayload
      const session: SetupSessionPayload = {
        phase: data.phase || 'idle',
        model: data.model || '',
        candidates: Array.isArray(data.candidates) ? data.candidates : [],
        assignments: Array.isArray(data.assignments) ? data.assignments : [],
        unassigned: Array.isArray(data.unassigned) ? data.unassigned : [],
        busy: Boolean(data.busy),
        busy_reason: data.busy_reason || '',
      }
      syncMotionTimer(session.phase, get().pollMotion)
      set((state) => {
        const wizardActive = state.wizardRequested || isSessionActive(session)
        const selectedModel = session.model || state.selectedModel
        const selectedCategory = state.selectedCategory
          && state.catalog?.models[state.selectedCategory]?.some((model) => model.name === selectedModel)
          ? state.selectedCategory
          : findCategoryForModel(state.catalog, selectedModel)
        return {
          assignments: session.assignments,
          busy: session.busy,
          busyReason: session.busy_reason,
          error:
            session.phase === 'idle' && session.busy && session.busy_reason
              ? `Embodiment busy: ${session.busy_reason}`
              : state.error,
          motionActive: session.phase === 'identifying',
          scannedPorts: buildScannedPorts(session.candidates, state.scannedPorts),
          scannedCameras: buildScannedCameras(session.candidates, state.scannedCameras),
          selectedCategory,
          selectedModel,
          sessionPhase: session.phase,
          wizardActive,
          wizardStep: wizardActive ? deriveWizardStep(state.wizardStep, session) : 'select',
        }
      })
    } catch (e: unknown) {
      clearMotionTimer()
      set({ error: (e as Error).message })
    }
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
        error: null,
      })
    } catch (e: unknown) {
      set({ error: (e as Error).message })
    }
    await get().refreshSession()
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
