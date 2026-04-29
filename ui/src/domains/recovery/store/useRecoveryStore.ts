import { create } from 'zustand'
import { api, postJson } from '@/shared/api/client'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'

const RECOVERY = '/api/recovery'
const RUNTIME_INFO = '/api/system/runtime-info'

export interface RecoveryFault {
  fault_type: string
  device_alias: string
  message: string
  timestamp: number
}

export interface RecoveryGuide {
  can_recheck: boolean
  step_count: number
}

interface RecoveryStore {
  faults: RecoveryFault[]
  guides: Record<string, RecoveryGuide> | null
  restarting: boolean
  fetchFaults: () => Promise<void>
  fetchGuides: () => Promise<void>
  checkDevice: (kind: 'arm' | 'camera', alias: string) => Promise<{ ok: boolean; kind: string; alias: string }>
  checkArmMotors: (alias: string) => Promise<{ ok: boolean; alias: string; missingMotors: string[] }>
  restartDashboard: () => Promise<void>
  handleDashboardEvent: (event: any) => void
}

async function waitForDashboardRecovery(timeoutMs: number = 30000): Promise<void> {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000))
    try {
      const response = await fetch(RUNTIME_INFO, { cache: 'no-store' })
      if (response.ok) {
        window.location.reload()
        return
      }
    } catch {
      // Dashboard still restarting; keep polling until timeout.
    }
  }
  throw new Error('Dashboard restart timed out')
}

export const useRecoveryStore = create<RecoveryStore>((set) => ({
  faults: [],
  guides: null,
  restarting: false,

  fetchFaults: async () => {
    const data = await api(`${RECOVERY}/faults`)
    set({ faults: Array.isArray(data.faults) ? data.faults : [] })
  },

  fetchGuides: async () => {
    set({ guides: await api(`${RECOVERY}/guides`) })
  },

  checkDevice: async (kind, alias) => {
    const data = await postJson(`${RECOVERY}/check-device`, { kind, alias })
    await useHardwareStore.getState().fetchHardwareStatus()
    return {
      ok: Boolean(data.ok),
      kind: String(data.kind || kind),
      alias: String(data.alias || alias),
    }
  },

  checkArmMotors: async (alias) => {
    const data = await postJson(`${RECOVERY}/check-arm-motors`, { alias })
    await useHardwareStore.getState().fetchHardwareStatus()
    return {
      ok: Boolean(data.ok),
      alias: String(data.alias || alias),
      missingMotors: Array.isArray(data.missing_motors) ? data.missing_motors.map(String) : [],
    }
  },

  restartDashboard: async () => {
    set({ restarting: true })
    try {
      await postJson(`${RECOVERY}/restart-dashboard`)
      await waitForDashboardRecovery()
    } finally {
      set({ restarting: false })
    }
  },

  handleDashboardEvent: (event) => {
    if (event.type === 'dashboard.fault.detected') {
      const fault: RecoveryFault = {
        fault_type: event.fault_type,
        device_alias: event.device_alias,
        message: event.message,
        timestamp: event.timestamp,
      }
      set((state) => ({
        faults: [
          ...state.faults.filter(
            (item) => !(item.fault_type === fault.fault_type && item.device_alias === fault.device_alias),
          ),
          fault,
        ],
      }))
      return
    }

    if (event.type === 'dashboard.fault.resolved') {
      set((state) => ({
        faults: state.faults.filter(
          (item) => !(item.fault_type === event.fault_type && item.device_alias === event.device_alias),
        ),
      }))
    }
  },
}))
