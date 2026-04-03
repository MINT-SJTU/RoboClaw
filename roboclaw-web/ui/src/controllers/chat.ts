export type MessageRole = 'user' | 'assistant'

export interface ChatAttachment {
  id: string
  name: string
  preview_url: string
  media_path?: string
  mime_type?: string
  size?: number
}

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

export function getMessageAttachments(message: Message): ChatAttachment[] {
  const attachments = message.metadata?.attachments
  if (!Array.isArray(attachments)) {
    return []
  }

  const normalized: ChatAttachment[] = []

  attachments.forEach((item, index) => {
    if (!item || typeof item !== 'object') {
      return
    }
    const attachment = item as Record<string, unknown>
    const previewUrl = attachment.preview_url
    if (typeof previewUrl !== 'string' || !previewUrl) {
      return
    }
    normalized.push({
      id:
        typeof attachment.id === 'string' && attachment.id
          ? attachment.id
          : `attachment-${index}`,
      name:
        typeof attachment.name === 'string' && attachment.name
          ? attachment.name
          : 'image',
      preview_url: previewUrl,
      media_path:
        typeof attachment.media_path === 'string'
          ? attachment.media_path
          : undefined,
      mime_type:
        typeof attachment.mime_type === 'string'
          ? attachment.mime_type
          : undefined,
      size:
        typeof attachment.size === 'number'
          ? attachment.size
          : undefined,
    })
  })

  return normalized
}
