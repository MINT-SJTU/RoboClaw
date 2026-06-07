import { create } from 'zustand'
import {
  type MessageRole,
  type Message,
  normalizeTimestamp,
  normalizeHistoryMessage,
} from '@/domains/chat/model/messages'
import { useSessionStore } from '@/domains/session/store/useSessionStore'
import { useHubTransferStore } from '@/domains/hub/store/useHubTransferStore'
import { useTrainingStore } from '@/domains/training/store/useTrainingStore'

export type { MessageRole, Message }

export type ChatConnectionStatus = 'connected' | 'disconnected' | 'reconnecting'

interface WebSocketStore {
  ws: WebSocket | null
  connected: boolean
  connectionStatus: ChatConnectionStatus
  connectionError: string
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

export const useChatSocket = create<WebSocketStore>((set, get) => ({
  ws: null,
  connected: false,
  connectionStatus: 'disconnected',
  connectionError: '',
  sessionId: '',
  messages: [],

  connect: () => {
    const current = get()
    if (current.ws || current.connected) {
      return
    }

    const sessionId = current.sessionId || getOrCreateSessionId()
    const ws = new WebSocket(resolveWebSocketUrl(sessionId))
    set({ ws, connected: false, connectionStatus: 'reconnecting', connectionError: '', sessionId })

    ws.onopen = () => {
      if (get().ws !== ws) {
        return
      }
      set({ connected: true, connectionStatus: 'connected', connectionError: '', sessionId })
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

      if (data.type === 'dashboard.training.state_changed') {
        useTrainingStore.getState().handleTrainingWebSocketEvent(data.payload)
        return
      }

      if (data.type?.startsWith('dashboard.')) {
        useSessionStore.getState().handleDashboardEvent(data)
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
      set({
        connected: false,
        connectionStatus: 'disconnected',
        connectionError: '连接已断开，正在重连...',
        ws: null,
      })
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        if (!get().connected && !get().ws) {
          set({ connectionStatus: 'reconnecting', connectionError: '连接已断开，正在重连...' })
          get().connect()
        }
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      if (get().ws === ws) {
        set({
          connected: false,
          connectionStatus: 'disconnected',
          connectionError: '连接已断开，正在重连...',
        })
      }
    }
  },

  disconnect: () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    const { ws } = get()
    set({ ws: null, connected: false, connectionStatus: 'disconnected' })
    if (ws) {
      ws.close()
    }
  },

  sendMessage: (content: string) => {
    const { ws, connected } = get()
    if (!connected || !ws) {
      const error = new Error('连接已断开，正在重连...')
      set({ connectionError: error.message, connectionStatus: 'disconnected' })
      window.dispatchEvent(new CustomEvent('roboclaw:connection-error', { detail: { message: error.message } }))
      throw error
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
        metadata: {},
      }),
    )
  },

  addMessage: (message: Message) => {
    const consult = message.metadata?.evoStudioAgentConsult
    if (message.role === 'assistant' && consult && typeof consult === 'object' && !Array.isArray(consult)) {
      useTrainingStore.getState().applyAgentConsultPlan(consult as Record<string, unknown>)
    }
    set((state) => ({
      messages: [...state.messages, message],
    }))
  },

  replaceMessages: (messages: Message[]) => {
    set({ messages })
  },
}))
