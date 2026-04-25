import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { useChatSocket } from '@/domains/chat/store/useChatSocket'
import { fetchProviderStatus } from '@/domains/provider/api/providerApi'
import { useI18n } from '@/i18n'
import { cn } from '@/shared/lib/cn'

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

  if (compact) {
    return (
      <section className="chat-widget__surface" aria-label="RoboClaw AI chat">
        <div className="chat-widget__conversation" aria-live="polite">
          {!providerConfigured ? (
            <div className="chat-widget__notice">
              {t('providerWarning')}{' '}
              <Link to="/settings/provider" className="chat-widget__notice-link">
                {t('settingsPage')}
              </Link>{' '}
              {t('providerWarningEnd')}
            </div>
          ) : messages.length === 0 ? (
            <div className="chat-widget__empty">
              <span
                className={cn('chat-widget__status', connected && 'chat-widget__status--live')}
                aria-hidden="true"
              />
              <span>RoboClaw AI</span>
            </div>
          ) : (
            <div className="chat-widget__message-stack">
              {messages.map((message, index) => {
                const isUser = message.role === 'user'
                return (
                  <article
                    key={message.id}
                    className={cn('chat-message', isUser && 'chat-message--user')}
                    style={{ animationDelay: `${Math.min(index * 28, 180)}ms` }}
                  >
                    <ReactMarkdown className="chat-markdown">
                      {message.content}
                    </ReactMarkdown>
                    <time className="chat-message__time" dateTime={new Date(message.timestamp).toISOString()}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </time>
                  </article>
                )
              })}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="chat-composer" aria-label="RoboClaw AI message">
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
            rows={1}
            className="chat-composer__input"
          />
          <button
            type="submit"
            disabled={!connected || !input.trim()}
            className="chat-composer__send"
            aria-label={t('send')}
          >
            <span aria-hidden="true" />
          </button>
        </form>
      </section>
    )
  }

  return (
    <section className="chat-panel">
      <header className="chat-panel__header">
        <div className="chat-panel__identity">
          <span className="chat-panel__avatar" aria-hidden="true">AI</span>
          <div className="chat-panel__title-group">
            <h2 className="chat-panel__title">{compact ? 'RoboClaw AI' : 'Conversation'}</h2>
            {!compact && (
              <div className="chat-panel__meta">
                {messages.length > 0 ? `${messages.length} messages` : 'Ready for a live conversation'}
              </div>
            )}
          </div>
        </div>

        <div className="chat-panel__actions">
          <span className={cn('chat-panel__status', connected && 'chat-panel__status--live')}>
            <span className="chat-panel__status-dot" aria-hidden="true" />
            {connected ? t('connected') : t('disconnected')}
          </span>

          {!compact && (
            <span className="chat-panel__session">Session {sessionId || 'pending'}</span>
          )}

          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="chat-panel__close"
              aria-label="Close chat"
            >
              <span aria-hidden="true">x</span>
            </button>
          )}
        </div>
      </header>

      {!providerConfigured && (
        <div className="chat-panel__notice">
          {t('providerWarning')}{' '}
          <Link to="/settings/provider" className="chat-panel__notice-link">
            {t('settingsPage')}
          </Link>{' '}
          {t('providerWarningEnd')}
        </div>
      )}

      <div className="chat-panel__messages">
        {messages.length === 0 ? (
          <div className="chat-panel__empty">
            <h3>{t('startChat')}</h3>
          </div>
        ) : (
          <div className="chat-panel__message-stack">
            {messages.map((message, index) => {
              const isUser = message.role === 'user'
              return (
                <article
                  key={message.id}
                  className={cn('chat-message', isUser && 'chat-message--user')}
                  style={{ animationDelay: `${Math.min(index * 28, 180)}ms` }}
                >
                  {!compact && (
                    <div className="chat-message__author">{isUser ? 'Operator' : 'RoboClaw'}</div>
                  )}

                  <ReactMarkdown className="chat-markdown">
                    {message.content}
                  </ReactMarkdown>

                  <time className="chat-message__time" dateTime={new Date(message.timestamp).toISOString()}>
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </time>
                </article>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <footer className="chat-panel__footer">
        <form onSubmit={handleSubmit} className="chat-panel__form">
          <div className="chat-panel__input-shell">
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
              rows={compact ? 2 : 4}
              className="chat-panel__input"
            />
          </div>

          <div className="chat-panel__form-row">
            <button
              type="submit"
              disabled={!connected || !input.trim()}
              className="chat-panel__send"
            >
              {t('send')}
            </button>
          </div>
        </form>
      </footer>
    </section>
  )
}
