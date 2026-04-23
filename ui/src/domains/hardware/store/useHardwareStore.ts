import { create } from 'zustand'
import { api } from '@/shared/api/client'

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
  servoPollingEnabled: boolean
  fetchHardwareStatus: () => Promise<void>
  fetchNetworkInfo: () => Promise<void>
  setServoPollingEnabled: (enabled: boolean) => void
}

export const useHardwareStore = create<HardwareStore>((set) => ({
  hardwareStatus: null,
  networkInfo: null,
  servoPollingEnabled: true,

  fetchHardwareStatus: async () => {
    set({ hardwareStatus: await api(`${HARDWARE}/status`) })
  },

  fetchNetworkInfo: async () => {
    set({ networkInfo: await api(`${SYSTEM}/network`) })
  },

  setServoPollingEnabled: (enabled) => {
    set({ servoPollingEnabled: enabled })
  },
}))
