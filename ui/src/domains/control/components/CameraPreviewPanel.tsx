import { useCallback, useState } from 'react'
import { useI18n } from '@/i18n'
import { postJson } from '@/shared/api/client'

export function CameraPreviewPanel({ cameras, busy }: { cameras: any[]; busy: boolean }) {
  const [previews, setPreviews] = useState<Record<string, string>>({})
  const [capturing, setCapturing] = useState(false)
  const { t } = useI18n()

  const capture = useCallback(async () => {
    if (busy || capturing) return
    setCapturing(true)
    try {
      const data = await postJson('/api/hardware/previews')
      const ts = Date.now()
      const map: Record<string, string> = {}
      data.forEach((preview: any) => {
        if (preview.alias && preview.preview_url) {
          map[preview.alias] = `${preview.preview_url}?t=${ts}`
        }
      })
      setPreviews(map)
    } catch {
      setPreviews({})
    } finally {
      setCapturing(false)
    }
  }, [busy, capturing, cameras])

  return (
    <div className="bg-sf rounded-lg p-4 shadow-card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-2xs text-tx3 font-mono uppercase tracking-widest">{t('cameras')}</h3>
        <button
          onClick={capture}
          disabled={busy || capturing}
          className="px-2.5 py-0.5 bg-ac/10 text-ac rounded text-xs font-medium hover:bg-ac/20 transition-colors disabled:opacity-30"
        >
          {capturing ? '...' : t('refresh')}
        </button>
      </div>
      {Object.keys(previews).length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {cameras.filter((c: any) => c.connected).map((camera: any) => (
            <div key={camera.alias} className="flex-1 min-w-[180px] max-w-[360px] relative rounded-lg overflow-hidden border border-bd/30">
              {previews[camera.alias] ? (
                <img
                  src={previews[camera.alias]}
                  alt={camera.alias}
                  className="w-full aspect-video object-contain bg-tx/5"
                />
              ) : (
                <div className="w-full aspect-video bg-tx/5 flex items-center justify-center text-sm text-tx3">
                  {t('noCameraFeed')}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-sm text-tx3">{t('noCameraFeed')}</div>
      )}
    </div>
  )
}
