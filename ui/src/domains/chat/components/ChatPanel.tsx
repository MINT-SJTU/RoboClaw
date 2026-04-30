import {
  type CSSProperties,
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { useChatSocket } from '@/domains/chat/store/useChatSocket'
import { fetchProviderStatus } from '@/domains/provider/api/providerApi'
import { useI18n } from '@/i18n'
import { cn } from '@/shared/lib/cn'

type ChatPanelVariant = 'page' | 'widget'

type LiquidGlassStyle = CSSProperties & {
  '--chat-liquid-glass-filter': string
}

type LiquidGlassTexture = {
  x: number
  y: number
}

type LiquidGlassShaderMap = {
  dataUrl: string
  scale: number
  width: number
  height: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function smoothStep(edge0: number, edge1: number, value: number): number {
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return t * t * (3 - 2 * t)
}

function length(x: number, y: number): number {
  return Math.sqrt(x * x + y * y)
}

function roundedRectSDF(
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): number {
  const qx = Math.abs(x) - width + radius
  const qy = Math.abs(y) - height + radius
  return Math.min(Math.max(qx, qy), 0) + length(Math.max(qx, 0), Math.max(qy, 0)) - radius
}

function texture(x: number, y: number): LiquidGlassTexture {
  return { x, y }
}

function createLiquidGlassShaderMap(width: number, height: number): LiquidGlassShaderMap | null {
  if (typeof document === 'undefined') return null

  const canvas = document.createElement('canvas')
  const canvasDPI = 1
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) return null

  const data = new Uint8ClampedArray(width * height * 4)
  const rawValues: number[] = []
  let maxScale = 0

  for (let index = 0; index < data.length; index += 4) {
    const x = (index / 4) % width
    const y = Math.floor(index / 4 / width)
    const uv = { x: x / width, y: y / height }
    const ix = uv.x - 0.5
    const iy = uv.y - 0.5
    const distanceToEdge = roundedRectSDF(ix, iy, 0.3, 0.2, 0.6)
    const displacement = smoothStep(0.8, 0, distanceToEdge - 0.15)
    const scaled = smoothStep(0, 1, displacement)
    const position = texture(ix * scaled + 0.5, iy * scaled + 0.5)
    const dx = position.x * width - x
    const dy = position.y * height - y

    maxScale = Math.max(maxScale, Math.abs(dx), Math.abs(dy))
    rawValues.push(dx, dy)
  }

  maxScale *= 0.5
  const mapScale = maxScale || 1
  let rawIndex = 0
  for (let index = 0; index < data.length; index += 4) {
    const red = rawValues[rawIndex++] / mapScale + 0.5
    const green = rawValues[rawIndex++] / mapScale + 0.5
    data[index] = Math.round(clamp(red, 0, 1) * 255)
    data[index + 1] = Math.round(clamp(green, 0, 1) * 255)
    data[index + 2] = 0
    data[index + 3] = 255
  }

  context.putImageData(new ImageData(data, width, height), 0, 0)
  return {
    dataUrl: canvas.toDataURL('image/png'),
    scale: maxScale / canvasDPI,
    width,
    height,
  }
}

function LiquidGlassFilter({
  filterId,
  targetRef,
}: {
  filterId: string
  targetRef: RefObject<HTMLElement>
}) {
  const [shaderMap, setShaderMap] = useState<LiquidGlassShaderMap | null>(null)

  useLayoutEffect(() => {
    let disposed = false
    let frameId = 0
    let retryFrameId = 0
    let removeResizeListener: (() => void) | null = null
    let resizeObserver: ResizeObserver | null = null
    let lastWidth = 0
    let lastHeight = 0

    function refreshShaderMap(observedElement: HTMLElement): void {
      const rect = observedElement.getBoundingClientRect()
      const width = Math.max(1, Math.round(rect.width))
      const height = Math.max(1, Math.round(rect.height))

      if (width === lastWidth && height === lastHeight) return

      lastWidth = width
      lastHeight = height
      setShaderMap(createLiquidGlassShaderMap(width, height))
    }

    function updateShaderMap(observedElement: HTMLElement): void {
      window.cancelAnimationFrame(frameId)
      frameId = window.requestAnimationFrame(() => refreshShaderMap(observedElement))
    }

    function connectFilter(): void {
      if (disposed) return

      const observedElement = targetRef.current
      if (!observedElement) {
        retryFrameId = window.requestAnimationFrame(connectFilter)
        return
      }

      const handleResize = () => updateShaderMap(observedElement)
      resizeObserver = new ResizeObserver(handleResize)
      resizeObserver.observe(observedElement)
      window.addEventListener('resize', handleResize)
      removeResizeListener = () => window.removeEventListener('resize', handleResize)
      refreshShaderMap(observedElement)
    }

    connectFilter()

    return () => {
      disposed = true
      window.cancelAnimationFrame(frameId)
      window.cancelAnimationFrame(retryFrameId)
      resizeObserver?.disconnect()
      removeResizeListener?.()
    }
  }, [targetRef])

  return (
    <svg
      className="liquid-glass-filter"
      width="0"
      height="0"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <filter
          id={`${filterId}_filter`}
          filterUnits="userSpaceOnUse"
          x="0"
          y="0"
          width={shaderMap?.width ?? 1}
          height={shaderMap?.height ?? 1}
          colorInterpolationFilters="sRGB"
        >
          <feImage
            id={`${filterId}_map`}
            href={shaderMap?.dataUrl ?? ''}
            x="0"
            y="0"
            width={shaderMap?.width ?? 1}
            height={shaderMap?.height ?? 1}
            preserveAspectRatio="none"
            result={`${filterId}_map`}
          />
          <feDisplacementMap
            in="SourceGraphic"
            in2={`${filterId}_map`}
            scale={shaderMap?.scale ?? 0}
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
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
  const filterId = `roboclaw-liquid-glass-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const liquidGlassStyle = {
    '--chat-liquid-glass-filter': `url(#${filterId}_filter) blur(0.25px) contrast(1.2) brightness(1.05) saturate(1.1)`,
  } as LiquidGlassStyle
  const [input, setInput] = useState('')
  const [providerConfigured, setProviderConfigured] = useState(true)
  const [widgetCollapsed, setWidgetCollapsed] = useState(compact)
  const { messages, sendMessage, connected, sessionId } = useChatSocket()
  const collapsedTriggerRef = useRef<HTMLButtonElement>(null)
  const conversationRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const pagePanelRef = useRef<HTMLElement>(null)
  const widgetSurfaceRef = useRef<HTMLElement>(null)
  const { t } = useI18n()

  function scrollToLatestMessage(behavior: ScrollBehavior = 'smooth'): void {
    const conversation = conversationRef.current
    if (conversation) {
      conversation.scrollTo({
        top: conversation.scrollHeight,
        behavior,
      })
      return
    }

    messagesEndRef.current?.scrollIntoView({ behavior })
  }

  useEffect(() => {
    scrollToLatestMessage()
  }, [messages])

  useEffect(() => {
    if (!compact || widgetCollapsed) return undefined

    let frameId = window.requestAnimationFrame(() => {
      scrollToLatestMessage('auto')
      frameId = window.requestAnimationFrame(() => scrollToLatestMessage('auto'))
    })

    return () => window.cancelAnimationFrame(frameId)
  }, [compact, widgetCollapsed])

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
    if (widgetCollapsed) {
      return (
        <>
          <LiquidGlassFilter filterId={filterId} targetRef={collapsedTriggerRef} />
          <button
            ref={collapsedTriggerRef}
            type="button"
            className="chat-widget__collapsed-trigger"
            style={liquidGlassStyle}
            onClick={() => setWidgetCollapsed(false)}
            aria-label="Open RoboClaw AI chat"
          >
            <span
              className={cn(
                'chat-widget__collapsed-dot',
                connected && 'chat-widget__collapsed-dot--live',
              )}
              aria-hidden="true"
            />
            <span className="chat-widget__collapsed-label" aria-hidden="true">AI</span>
          </button>
        </>
      )
    }

    return (
      <>
        <LiquidGlassFilter filterId={filterId} targetRef={widgetSurfaceRef} />
        <section
          ref={widgetSurfaceRef}
          className="chat-widget__surface"
          style={liquidGlassStyle}
          aria-label="RoboClaw AI chat"
        >
          <button
            type="button"
            className="chat-widget__minimize"
            onClick={() => setWidgetCollapsed(true)}
            aria-label="Close RoboClaw AI chat"
          >
            <span aria-hidden="true">×</span>
          </button>

          <div ref={conversationRef} className="chat-widget__conversation" aria-live="polite">
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
      </>
    )
  }

  return (
    <section ref={pagePanelRef} className="chat-panel liquid-glass-panel" style={liquidGlassStyle}>
      <LiquidGlassFilter filterId={filterId} targetRef={pagePanelRef} />
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
