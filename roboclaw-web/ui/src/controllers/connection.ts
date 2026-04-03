import { create } from 'zustand'
import {
  type MessageRole,
  type Message,
  type ChatAttachment,
  normalizeTimestamp,
  normalizeHistoryMessage,
} from './chat'
import { useDashboard } from './dashboard'

export type { MessageRole, Message, ChatAttachment }

interface WebSocketStore {
  ws: WebSocket | null
  connected: boolean
  sessionId: string
  messages: Message[]
  connect: () => void
  disconnect: () => void
  sendMessage: (content: string, attachments?: ChatAttachment[]) => void
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

export const useWebSocket = create<WebSocketStore>((set, get) => ({
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
        useDashboard.getState().handleDashboardEvent(data)
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

  sendMessage: (content: string, attachments = []) => {
    const { ws, connected } = get()
    if (!connected || !ws) {
      console.error('WebSocket not connected')
      return
    }
    const normalizedContent = content.trim()
    const metadata = attachments.length > 0 ? { attachments } : {}

    get().addMessage({
      id: `${Date.now()}-user`,
      role: 'user',
      content: normalizedContent || '[image]',
      timestamp: Date.now(),
      metadata,
    })

    ws.send(
      JSON.stringify({
        type: 'chat.send',
        content: normalizedContent,
        media: attachments
          .map((attachment) => attachment.media_path)
          .filter((path): path is string => Boolean(path)),
        metadata,
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
