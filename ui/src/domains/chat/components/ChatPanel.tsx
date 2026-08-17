import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { useChatSocket } from '@/domains/chat/store/useChatSocket'
import { fetchProviderStatus } from '@/domains/provider/api/providerApi'
import { useI18n } from '@/i18n'
import { GlassPanel, StatusPill } from '@/shared/ui'

type ChatPanelVariant = 'page' | 'widget'

export default function ChatPanel({
  variant = 'page',
  onClose,
}: {
  variant?: ChatPanelVariant
  onClose?: () => void
}) {
  const compact = variant === 'widget'
  const [input, setInput] = useState('')
  const [providerConfigured, setProviderConfigured] = useState(true)
  const { messages, sendMessage, connected, sessionId } = useChatSocket()
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
        if (!cancelled) {
          setProviderConfigured(payload.active_provider_configured)
        }
      } catch (_error) {
        if (!cancelled) {
          setProviderConfigured(false)
        }
      }
    }

    loadProviderStatus()
    return () => {
      cancelled = true
    }
  }, [])

  function submitCurrentMessage(): void {
    const content = input.trim()
    if (!content || !connected) return
    sendMessage(content)
    setInput('')
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    submitCurrentMessage()
  }

  return (
    <GlassPanel
      className={
        compact
          ? 'flex h-[min(680px,calc(100vh-124px))] w-[min(460px,calc(100vw-28px))] flex-col overflow-hidden rounded-[30px] p-0'
          : 'page-enter flex h-[calc(100vh-176px)] min-h-[620px] flex-col overflow-hidden p-0'
      }
    >
      {compact ? (
        <div className="flex items-center justify-between gap-3 border-b border-[color:rgba(29,43,54,0.08)] px-4 py-4">
          <div>
            <div className="text-sm font-semibold text-tx">RoboClaw AI</div>
            <div className="mt-1 text-xs text-tx2">
              {messages.length > 0 ? `${messages.length} messages` : t('startChat')}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusPill active={connected}>
              {connected ? t('connected') : t('disconnected')}
            </StatusPill>
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
        </div>
      ) : (
        <div className="border-b border-[color:rgba(29,43,54,0.08)] px-5 py-5 md:px-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-base font-semibold text-tx">Conversation</div>
              <div className="mt-1 text-sm text-tx2">
                {messages.length > 0
                  ? `${messages.length} messages in this session`
                  : 'Ready for a live conversation'}
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
          <div className="px-4 pt-3 text-xs text-yl">
              {t('providerWarning')}{' '}
              <Link to="/settings/provider" className="font-semibold underline underline-offset-4">
                {t('settingsPage')}
              </Link>{' '}
            {t('providerWarningEnd')}
          </div>
        ) : (
          <div className="border-b border-[color:rgba(29,43,54,0.08)] px-5 py-4 md:px-6">
            <div className="rounded-[22px] border border-[rgba(109,153,211,0.18)] bg-[linear-gradient(135deg,rgba(255,255,255,0.94),rgba(226,239,255,0.72))] px-4 py-3 text-sm text-tx2">
              {t('providerWarning')}{' '}
              <Link to="/settings/provider" className="font-semibold text-ac underline underline-offset-4">
                {t('settingsPage')}
              </Link>{' '}
              {t('providerWarningEnd')}
            </div>
          </div>
        )
      )}

      <div className={`sidebar-scroll flex-1 overflow-y-auto ${compact ? 'px-4 py-4' : 'px-5 py-5 md:px-6'}`}>
        {messages.length === 0 ? (
          <div
            className={
              compact
                ? 'grid min-h-[280px] place-items-center text-center'
                : 'grid min-h-[320px] place-items-center rounded-[28px] border border-dashed border-[color:rgba(109,153,211,0.18)] bg-white/56 p-6 text-center'
            }
          >
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
                    className={`${
                      compact ? 'max-w-[92%] rounded-[20px] px-4 py-3' : 'max-w-3xl rounded-[28px] px-5 py-4 shadow-[0_18px_34px_rgba(88,67,47,0.09)]'
                    } ${
                      isUser
                        ? 'bg-ac text-white'
                        : compact
                          ? 'border border-[color:rgba(109,153,211,0.14)] bg-[rgba(141,184,236,0.12)] text-tx'
                          : 'border border-white/60 bg-white/74 text-tx'
                    }`}
                  >
                    {!compact && (
                      <div
                        className={`mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] ${
                          isUser ? 'text-white/70' : 'text-tx2'
                        }`}
                      >
                        <span>{isUser ? 'Operator' : 'RoboClaw'}</span>
                        <span className={`inline-flex h-2 w-2 rounded-full ${isUser ? 'bg-white/80' : 'bg-ac'}`} />
                      </div>
                    )}

                    <ReactMarkdown
                      className={`chat-markdown ${isUser ? '[&_code]:bg-white/10 [&_code]:text-white' : ''}`}
                    >
                      {message.content}
                    </ReactMarkdown>

                    <div className={`mt-4 text-[11px] ${isUser ? 'text-white/60' : 'text-tx2'}`}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div
        className={
          compact
            ? 'border-t border-[color:rgba(29,43,54,0.08)] px-4 pb-4 pt-3'
            : 'border-t border-[color:rgba(29,43,54,0.08)] bg-white/45 px-5 py-4 md:px-6'
        }
      >
        <form onSubmit={handleSubmit} className={compact ? '' : 'space-y-3'}>
          <div className="field-shell px-4 py-3">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  submitCurrentMessage()
                }
              }}
              placeholder={connected ? t('inputPlaceholder') : t('waitingConnection')}
              disabled={!connected}
              rows={compact ? 3 : 4}
              className="w-full resize-none border-0 bg-transparent text-sm leading-7 text-tx outline-none placeholder:text-tx3 disabled:opacity-50"
            />
          </div>

          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-tx2">
              {compact ? 'Enter to send' : 'Enter to send, Shift+Enter for a new line.'}
            </div>

            <button
              type="submit"
              disabled={!connected || !input.trim()}
              className="inline-flex items-center justify-center rounded-full bg-ac px-5 py-2.5 text-sm font-semibold text-white shadow-[0_14px_28px_rgba(47,111,228,0.22)] transition hover:-translate-y-0.5 hover:bg-[color:#245dc7] disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {t('send')}
            </button>
          </div>
        </form>
      </div>
    </GlassPanel>
  )
}
