import { create } from 'zustand'
import {
  type MessageRole,
  type Message,
  normalizeTimestamp,
  normalizeHistoryMessage,
} from '@/domains/chat/model/messages'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { useRecoveryStore } from '@/domains/recovery/store/useRecoveryStore'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'
import { useWorkflow } from '@/domains/curation/store/useCurationStore'

export type { MessageRole, Message }

interface WebSocketStore {
  ws: WebSocket | null
  connected: boolean
  sessionId: string
  messages: Message[]
  connect: () => void
  disconnect: () => void
  sendMessage: (content: string) => void
  addMessage: (message: Message) => void
  replaceMessages: (messages: Message[]) => void
}

const STORAGE_KEY = 'roboclaw.web.chat_id'

function createSessionId(): string {
  return `web-${Math.random().toString(36).slice(2, 10)}`
}

function getOrCreateSessionId(): string {
  const existing = window.localStorage.getItem(STORAGE_KEY)
  if (existing) {
    return existing
  }
  const created = createSessionId()
  window.localStorage.setItem(STORAGE_KEY, created)
  return created
}

function persistSessionId(sessionId: string): void {
  window.localStorage.setItem(STORAGE_KEY, sessionId)
}

let reconnectTimer: ReturnType<typeof setTimeout> | null = null

function resolveWebSocketUrl(sessionId: string): string {
  const override = import.meta.env.VITE_WEBSOCKET_URL as string | undefined
  const url = override
    ? new URL(override)
    : new URL('/ws', window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.searchParams.set('chat_id', sessionId)
  return url.toString()
}

function buildAppContext(): Record<string, unknown> {
  const workflow = useWorkflow.getState()
  const training = useTrainingStore.getState()
  const session = useSessionStore.getState().session
  const hardware = useHardwareStore.getState().hardwareStatus
  const stageStatuses = workflow.workflowState
    ? Object.fromEntries(
      Object.entries(workflow.workflowState.stages).map(([name, stage]: [string, any]) => [
        name,
        stage?.status ?? 'unknown',
      ]),
    )
    : null

  return {
    route: window.location.pathname,
    search: window.location.search,
    hash: window.location.hash,
    href: window.location.href,
    selected_dataset: workflow.selectedDataset,
    selected_dataset_label: workflow.datasetInfo?.label ?? null,
    selected_dataset_prepared: workflow.selectedDatasetIsRemotePrepared,
    workflow: stageStatuses,
    quality: {
      running: workflow.qualityRunning,
      validators: workflow.selectedValidators,
      defaults_loaded: Boolean(workflow.qualityDefaults),
    },
    training: {
      current_job_id: training.currentTrainJobId,
      loading: training.trainingLoading,
      stop_loading: training.trainingStopLoading,
    },
    session: {
      state: session.state,
      dataset: session.dataset,
      episode_phase: session.episode_phase,
    },
    hardware: hardware
      ? {
        ready: hardware.ready,
        missing: hardware.missing,
        arms: hardware.arms.map((arm) => ({
          alias: arm.alias,
          connected: arm.connected,
          calibrated: arm.calibrated,
        })),
        cameras: hardware.cameras.map((camera) => ({
          alias: camera.alias,
          connected: camera.connected,
        })),
      }
      : null,
    client_timestamp: Date.now(),
  }
}

function navigateClient(route: string): void {
  if (!route.startsWith('/') || route.startsWith('//')) {
    return
  }
  const target = `${route}`
  if (`${window.location.pathname}${window.location.search}${window.location.hash}` === target) {
    return
  }
  window.history.pushState(null, '', target)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

export const useChatSocket = create<WebSocketStore>((set, get) => ({
  ws: null,
  connected: false,
  sessionId: '',
  messages: [],

  connect: () => {
    const current = get()
    if (current.ws || current.connected) {
      return
    }

    const sessionId = current.sessionId || getOrCreateSessionId()
    const ws = new WebSocket(resolveWebSocketUrl(sessionId))
    set({ ws, connected: false, sessionId })

    ws.onopen = () => {
      if (get().ws !== ws) {
        return
      }
      set({ connected: true, sessionId })
    }

    ws.onmessage = (event) => {
      if (get().ws !== ws) {
        return
      }
      let data: any
      try {
        data = JSON.parse(event.data)
      } catch {
        console.warn('Non-JSON websocket message:', event.data)
        return
      }

      if (data.type?.startsWith('dashboard.')) {
        useSessionStore.getState().handleDashboardEvent(data)
        useRecoveryStore.getState().handleDashboardEvent(data)
        useHubTransferStore.getState().handleDashboardEvent(data)
        return
      }

      if (data.type === 'session.init') {
        const resolvedSessionId = String(data.chat_id || sessionId)
        persistSessionId(resolvedSessionId)
        set({
          sessionId: resolvedSessionId,
          messages: Array.isArray(data.history) ? data.history.map(normalizeHistoryMessage) : [],
        })
        return
      }

      if (data.type === 'chat.message') {
        const appEvent = data.metadata?.app_event
        if (appEvent?.type === 'app.navigate' && typeof appEvent.route === 'string') {
          navigateClient(appEvent.route)
          if (!String(data.content ?? '').trim()) {
            return
          }
        }
        get().addMessage({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          role: data.role === 'user' ? 'user' : 'assistant',
          content: String(data.content ?? ''),
          timestamp: normalizeTimestamp(data.timestamp),
          metadata: data.metadata ?? {},
        })
      }
    }

    ws.onclose = () => {
      if (get().ws !== ws) {
        return
      }
      set({ connected: false, ws: null })
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        if (!get().connected && !get().ws) {
          get().connect()
        }
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  },

  disconnect: () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    const { ws } = get()
    set({ ws: null, connected: false })
    if (ws) {
      ws.close()
    }
  },

  sendMessage: (content: string) => {
    const { ws, connected } = get()
    if (!connected || !ws) {
      console.error('WebSocket not connected')
      return
    }

    get().addMessage({
      id: `${Date.now()}-user`,
      role: 'user',
      content,
      timestamp: Date.now(),
      metadata: {},
    })

    ws.send(
      JSON.stringify({
        type: 'chat.send',
        content,
        metadata: {
          app_context: buildAppContext(),
        },
      }),
    )
  },

  addMessage: (message: Message) => {
    set((state) => ({
      messages: [...state.messages, message],
    }))
  },

  replaceMessages: (messages: Message[]) => {
    set({ messages })
  },
}))
