import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { useWebSocket } from '../controllers/connection'
import { fetchProviderStatus } from '../controllers/provider'
import { useI18n } from '../controllers/i18n'
import { ActionButton, GlassPanel, StatusPill } from './ux'
import {
  getMessageAttachments,
  type ChatAttachment,
  type Message,
} from '../controllers/chat'

type ChatPanelVariant = 'page' | 'widget'

function extractImageFiles(fileList: FileList | File[]): File[] {
  return Array.from(fileList).filter((file) => file.type.startsWith('image/'))
}

function extractClipboardImageFiles(event: React.ClipboardEvent<HTMLTextAreaElement>): File[] {
  const items = Array.from(event.clipboardData?.items || [])
  return items
    .map((item) => (item.kind === 'file' ? item.getAsFile() : null))
    .filter((file): file is File => Boolean(file && file.type.startsWith('image/')))
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error('Failed to read image file.'))
      }
    }
    reader.onerror = () => reject(new Error('Failed to read image file.'))
    reader.readAsDataURL(file)
  })
}

async function uploadChatImage(
  sessionId: string,
  file: File,
): Promise<ChatAttachment> {
  const dataUrl = await readFileAsDataUrl(file)
  const response = await fetch('/api/chat/uploads/image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chat_id: sessionId,
      filename: file.name,
      data_url: dataUrl,
    }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Upload failed (${response.status})`)
  }

  const payload = await response.json()
  return {
    id: String(payload.id ?? `${Date.now()}-${file.name}`),
    name: String(payload.name ?? file.name),
    preview_url: String(payload.preview_url ?? ''),
    media_path: typeof payload.media_path === 'string' ? payload.media_path : undefined,
    mime_type: typeof payload.mime_type === 'string' ? payload.mime_type : file.type,
    size: typeof payload.size === 'number' ? payload.size : file.size,
  }
}

function MessageAttachments({
  message,
  compact,
}: {
  message: Message
  compact: boolean
}) {
  const attachments = getMessageAttachments(message)
  if (!attachments.length || message.role !== 'user') {
    return null
  }

  return (
    <div className={`mb-3 grid gap-2 ${compact ? 'grid-cols-2' : 'grid-cols-3'}`}>
      {attachments.map((attachment) => (
        <a
          key={attachment.id}
          href={attachment.preview_url}
          target="_blank"
          rel="noreferrer"
          className="group overflow-hidden rounded-2xl border border-white/20 bg-white/10"
        >
          <img
            src={attachment.preview_url}
            alt={attachment.name}
            className={`block w-full object-cover transition duration-200 group-hover:scale-[1.02] ${compact ? 'h-24' : 'h-32'}`}
          />
        </a>
      ))}
    </div>
  )
}

export default function ChatPanel({
  variant = 'page',
  onClose,
}: {
  variant?: ChatPanelVariant
  onClose?: () => void
}) {
  const compact = variant === 'widget'
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<ChatAttachment[]>([])
  const [uploadingImages, setUploadingImages] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [providerConfigured, setProviderConfigured] = useState(true)
  const { messages, sendMessage, connected, sessionId } = useWebSocket()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { t } = useI18n()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    let cancelled = false

    async function loadProviderStatus() {
      try {
        const payload = await fetchProviderStatus()
        if (cancelled) return
        setProviderConfigured(payload.active_provider_configured)
      } catch (_error) {
        if (!cancelled) setProviderConfigured(false)
      }
    }

    loadProviderStatus()
    return () => {
      cancelled = true
    }
  }, [])

  function submitCurrentMessage(): void {
    if ((!input.trim() && attachments.length === 0) || !connected) return
    sendMessage(input, attachments)
    setInput('')
    setAttachments([])
    setUploadError('')
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    submitCurrentMessage()
  }

  async function uploadFiles(files: File[]): Promise<void> {
    const imageFiles = extractImageFiles(files)
    if (!imageFiles.length || !sessionId) {
      return
    }

    setUploadingImages(true)
    setUploadError('')
    try {
      const uploaded = await Promise.all(
        imageFiles.map((file) => uploadChatImage(sessionId, file)),
      )
      setAttachments((current) => [...current, ...uploaded])
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Image upload failed')
    } finally {
      setUploadingImages(false)
    }
  }

  const content = (
    <>
      {compact ? (
        <div className="flex items-center justify-end px-4 pt-4">
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-[color:rgba(109,153,211,0.16)] bg-white text-sm font-semibold text-tx transition hover:bg-[rgba(141,184,236,0.12)]"
              aria-label="Close chat"
            >
              X
            </button>
          )}
        </div>
      ) : (
          <div className="border-b border-[color:rgba(29,43,54,0.08)] px-4 py-4 md:px-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-base font-semibold text-tx">Conversation</div>
              <div className="mt-1 text-sm text-tx2">
                {messages.length > 0 ? `${messages.length} messages in this session` : 'Ready for a live conversation'}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              <StatusPill active={connected}>
                {connected ? t('connected') : t('disconnected')}
              </StatusPill>

              <div className="rounded-full bg-white/70 px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-tx2">
                Session {sessionId || 'pending'}
              </div>
            </div>
          </div>
        </div>
      )}

      {!providerConfigured && (
        compact ? (
          <div className="px-4 pt-2 text-xs text-yl">
            {t('providerWarning')}{' '}
            <Link to="/settings" className="font-semibold underline underline-offset-4">
              {t('settingsPage')}
            </Link>
          </div>
        ) : (
          <div className="border-b border-[color:rgba(29,43,54,0.08)] px-4 pb-4 md:px-6">
            <div className="rounded-[22px] border border-[rgba(109,153,211,0.18)] bg-[linear-gradient(135deg,rgba(255,255,255,0.94),rgba(226,239,255,0.72))] px-4 py-3 text-sm text-tx2">
              {t('providerWarning')}{' '}
              <Link to="/settings" className="font-semibold text-ac underline underline-offset-4">
                {t('settingsPage')}
              </Link>{' '}
              {t('providerWarningEnd')}
            </div>
          </div>
        )
      )}

      <div className={`sidebar-scroll flex-1 overflow-y-auto ${compact ? 'px-4 py-4' : 'px-4 py-5 md:px-6'}`}>
        {messages.length === 0 ? (
          <div className={`grid place-items-center text-center ${compact ? 'min-h-[280px]' : 'min-h-[320px] rounded-[28px] border border-dashed border-[color:rgba(109,153,211,0.18)] bg-white/56 p-6'}`}>
            <div className="space-y-3">
              <div className={`${compact ? 'text-lg' : 'text-2xl'} font-semibold text-tx`}>
                {t('startChat')}
              </div>
              <div className={`${compact ? 'max-w-sm text-sm leading-6' : 'max-w-md text-sm leading-7'} text-tx2`}>
                Ask RoboClaw to inspect hardware state, suggest next recording steps, or summarize recent robot activity.
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message, index) => {
              const isUser = message.role === 'user'
              return (
                <div
                  key={message.id}
                  className={`tile-enter flex ${isUser ? 'justify-end' : 'justify-start'}`}
                  style={{ animationDelay: `${Math.min(index * 40, 260)}ms` }}
                >
                  <div
                    className={`px-4 py-3 ${
                      compact ? 'max-w-[92%] rounded-[20px]' : 'max-w-3xl rounded-[28px] px-5 py-4 shadow-[0_18px_34px_rgba(88,67,47,0.09)]'
                    } ${
                      isUser
                        ? 'bg-ac text-white'
                        : compact
                          ? 'border border-[color:rgba(109,153,211,0.14)] bg-[rgba(141,184,236,0.12)] text-tx'
                          : 'border border-white/60 bg-white/74 text-tx'
                    }`}
                  >
                    {!compact && (
                      <div className={`mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] ${isUser ? 'text-white/70' : 'text-tx2'}`}>
                        <span>{isUser ? 'Operator' : 'RoboClaw'}</span>
                        <span className={`inline-flex h-2 w-2 rounded-full ${isUser ? 'bg-white/80' : 'bg-ac'}`} />
                      </div>
                    )}
                    <MessageAttachments message={message} compact={compact} />
                    <ReactMarkdown className={`chat-markdown ${isUser ? '[&_code]:bg-white/10 [&_code]:text-white' : ''}`}>
                      {message.content}
                    </ReactMarkdown>
                    {!compact && (
                      <div className={`mt-4 text-[11px] ${isUser ? 'text-white/60' : 'text-tx2'}`}>
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className={`${compact ? 'border-t border-[color:rgba(29,43,54,0.08)] px-4 pb-4 pt-3' : 'border-t border-[color:rgba(29,43,54,0.08)] bg-white/45 px-4 py-4 md:px-6'}`}>
        <form onSubmit={handleSubmit} className={compact ? '' : 'space-y-3'}>
          {(attachments.length > 0 || uploadError) && (
            <div className={`${compact ? 'mb-3 space-y-2' : 'space-y-2'}`}>
              {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {attachments.map((attachment) => (
                    <div
                      key={attachment.id}
                      className="relative overflow-hidden rounded-2xl border border-[color:rgba(47,111,228,0.16)] bg-white"
                    >
                      <img
                        src={attachment.preview_url}
                        alt={attachment.name}
                        className="h-20 w-20 object-cover"
                      />
                      <button
                        type="button"
                        onClick={() =>
                          setAttachments((current) =>
                            current.filter((item) => item.id !== attachment.id),
                          )
                        }
                        className="absolute right-1 top-1 inline-flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-xs font-semibold text-white"
                        aria-label={t('removeImage')}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {uploadError && (
                <div className="text-xs text-rd">{uploadError}</div>
              )}
            </div>
          )}
          <div
            className={`field-shell relative flex items-end gap-3 ${compact ? 'rounded-[20px] bg-white px-4 py-3' : 'px-4 py-3'} ${
              dragActive ? 'ring-2 ring-ac/30 bg-[rgba(47,111,228,0.06)]' : ''
            }`}
            onDragOver={(event) => {
              const hasImage = Array.from(event.dataTransfer?.items || []).some(
                (item) => item.kind === 'file' && item.type.startsWith('image/'),
              )
              if (!hasImage) {
                return
              }
              event.preventDefault()
              event.dataTransfer.dropEffect = 'copy'
              setDragActive(true)
            }}
            onDragLeave={(event) => {
              if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
                return
              }
              setDragActive(false)
            }}
            onDrop={(event) => {
              event.preventDefault()
              setDragActive(false)
              void uploadFiles(Array.from(event.dataTransfer.files || []))
            }}
          >
            {dragActive && (
              <div className="pointer-events-none absolute inset-2 grid place-items-center rounded-[18px] border border-dashed border-ac/30 bg-white/80 text-xs font-semibold text-ac">
                {t('dropImageHere')}
              </div>
            )}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  if (!connected || (!input.trim() && attachments.length === 0)) return
                  submitCurrentMessage()
                }
              }}
              onPaste={(event) => {
                const files = extractClipboardImageFiles(event)
                if (!files.length) {
                  return
                }
                event.preventDefault()
                void uploadFiles(files)
              }}
              placeholder={connected ? t('inputPlaceholder') : t('waitingConnection')}
              disabled={!connected}
              rows={compact ? 3 : 2}
              className="min-h-[64px] flex-1 resize-none border-0 bg-transparent px-0 py-1 text-sm text-tx outline-none placeholder:text-tx2 disabled:opacity-50"
            />
            <ActionButton
              type="submit"
              disabled={!connected || uploadingImages || (!input.trim() && attachments.length === 0)}
              className="shrink-0"
            >
              {t('send')}
            </ActionButton>
          </div>

          {!compact && (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-tx2">
              <span>{t('chatInputHint')}</span>
              <span>{connected ? 'Connection healthy' : 'Waiting for WebSocket reconnect'}</span>
            </div>
          )}
        </form>
      </div>
    </>
  )

  if (compact) {
    return (
      <div className="flex h-[min(78vh,720px)] w-[min(calc(100vw-24px),520px)] flex-col overflow-hidden rounded-[28px] border border-[color:rgba(29,43,54,0.12)] bg-white shadow-[0_24px_56px_rgba(29,43,54,0.14)]">
        {content}
      </div>
    )
  }

  return (
    <GlassPanel className="flex min-h-[calc(100vh-210px)] flex-col overflow-hidden p-0">
      {content}
    </GlassPanel>
  )
}
