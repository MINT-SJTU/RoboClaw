import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useWebSocket } from '../controllers/connection'
import ReactMarkdown from 'react-markdown'
import { fetchProviderStatus } from '../controllers/provider'
import { useI18n } from '../controllers/i18n'

export default function ChatView() {
  const [input, setInput] = useState('')
  const [providerConfigured, setProviderConfigured] = useState(true)
  const { messages, sendMessage, connected } = useWebSocket()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { t } = useI18n()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
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
    return () => { cancelled = true }
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !connected) return
    sendMessage(input)
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {!providerConfigured && (
        <div className="border-b border-yl/30 bg-yl/10 px-4 py-2.5 text-sm text-yl">
          {t('providerWarning')}{' '}
          <Link to="/settings" className="font-semibold underline">{t('settingsPage')}</Link>
          {' '}{t('providerWarningEnd')}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-tx2 mt-8">
            {t('startChat')}
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-lg p-3.5 ${
                message.role === 'user'
                  ? 'bg-ac/10 border border-ac/20 text-tx'
                  : 'bg-sf border border-bd text-tx'
              }`}
            >
              <ReactMarkdown className="prose max-w-none text-sm leading-relaxed [&_p]:my-1 [&_code]:text-ac [&_code]:bg-sf [&_code]:px-1 [&_code]:rounded-sm">
                {message.content}
              </ReactMarkdown>
              <div className="text-2xs text-tx2 mt-1.5">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="bg-sf border-t border-bd p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={connected ? t('inputPlaceholder') : t('waitingConnection')}
            disabled={!connected}
            className="flex-1 bg-bg border border-bd text-tx rounded px-3 py-2 text-sm focus:outline-none focus:border-ac disabled:opacity-30"
          />
          <button
            type="submit"
            disabled={!connected || !input.trim()}
            className="border border-ac text-ac px-5 py-2 rounded text-sm transition-colors hover:bg-ac/10 active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {t('send')}
          </button>
        </div>
      </form>
    </div>
  )
}
