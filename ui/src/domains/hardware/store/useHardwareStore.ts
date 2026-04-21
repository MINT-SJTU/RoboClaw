import { create } from 'zustand'

const HARDWARE = '/api/hardware'
const SYSTEM = '/api/system'

export interface ArmStatus {
  alias: string
  type: string
  role: string
  connected: boolean
  calibrated: boolean
}

export interface CameraStatus {
  alias: string
  connected: boolean
  width: number
  height: number
}

export interface OperationCapability {
  ready: boolean
  missing: string[]
}

export interface HardwareCapabilities {
  teleop: OperationCapability
  record: OperationCapability
  record_without_cameras: OperationCapability
  replay: OperationCapability
  infer: OperationCapability
  infer_without_cameras: OperationCapability
}

export interface HardwareStatus {
  ready: boolean
  missing: string[]
  arms: ArmStatus[]
  cameras: CameraStatus[]
  session_busy: boolean
  capabilities: HardwareCapabilities
}

export interface NetworkInfo {
  host: string
  port: number
  lan_ip: string
}

interface HardwareStore {
  hardwareStatus: HardwareStatus | null
  networkInfo: NetworkInfo | null
  fetchHardwareStatus: () => Promise<void>
  fetchNetworkInfo: () => Promise<void>
}

export const useHardwareStore = create<HardwareStore>((set) => ({
  hardwareStatus: null,
  networkInfo: null,

  fetchHardwareStatus: async () => {
    const res = await fetch(`${HARDWARE}/status`)
    if (!res.ok) {
      throw new Error(`Failed to fetch hardware status: ${res.status}`)
    }
    set({ hardwareStatus: await res.json() })
  },

  fetchNetworkInfo: async () => {
    const res = await fetch(`${SYSTEM}/network`)
    if (!res.ok) {
      throw new Error(`Failed to fetch network info: ${res.status}`)
    }
    set({ networkInfo: await res.json() })
  },
}))
