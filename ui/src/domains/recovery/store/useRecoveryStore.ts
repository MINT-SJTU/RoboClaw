import { create } from 'zustand'
import { postJson } from '@/shared/api/client'

const RECOVERY = '/api/recovery'
const RUNTIME_INFO = '/api/system/runtime-info'

export interface RecoveryFault {
  fault_type: string
  device_alias: string
  message: string
  timestamp: number
}

interface RecoveryStore {
  faults: RecoveryFault[]
  hasCheckedHardware: boolean
  checkingHardware: boolean
  restarting: boolean
  checkHardware: () => Promise<void>
  restartDashboard: () => Promise<void>
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
  hasCheckedHardware: false,
  checkingHardware: false,
  restarting: false,

  checkHardware: async () => {
    set({ checkingHardware: true })
    try {
      const data = await postJson(`${RECOVERY}/check-hardware`)
      set({
        faults: Array.isArray(data.faults) ? data.faults : [],
        hasCheckedHardware: true,
      })
    } finally {
      set({ checkingHardware: false })
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
}))
