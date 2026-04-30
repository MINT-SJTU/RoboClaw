export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  timestamp: number
  metadata?: Record<string, unknown>
}

export function normalizeTimestamp(value: unknown): number {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Date.parse(value)
    if (!Number.isNaN(parsed)) {
      return parsed
    }
  }
  return Date.now()
}

export function normalizeHistoryMessage(message: any): Message {
  return {
    id: String(message.id ?? `${message.role ?? 'assistant'}-${Math.random()}`),
    role: message.role === 'user' ? 'user' : 'assistant',
    content: String(message.content ?? ''),
    timestamp: normalizeTimestamp(message.timestamp),
    metadata: message.metadata ?? {},
  }
}
