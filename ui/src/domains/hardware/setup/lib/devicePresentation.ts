import type { ScannedCamera, ScannedPort } from '@/domains/hardware/setup/store/useSetupStore'

type PresentationKey =
  | 'camera'
  | 'setupExternalCamera'
  | 'setupCameraCurrentStream'
  | 'setupCameraDetecting'
type Translate = (key: PresentationKey, vars?: Record<string, string | number>) => string

function isBuiltinCameraName(name: string): boolean {
  return /MacBook|FaceTime|built[- ]?in|内建/i.test(name)
}

export function presentPortLabel(port: Pick<ScannedPort, 'label' | 'dev'>): string {
  const raw = port.label || port.dev || '?'
  if (!raw.includes('/')) {
    return raw
  }
  return raw.split('/').pop() || raw
}

export function presentCameraLabel(
  camera: Pick<ScannedCamera, 'stable_id' | 'label'>,
  cameras: Array<Pick<ScannedCamera, 'stable_id' | 'label'>>,
  t: Translate,
): string {
  const raw = camera.label || ''
  if (isBuiltinCameraName(raw)) {
    return raw || t('camera')
  }

  let externalIndex = 0
  for (const item of cameras) {
    const itemRaw = item.label || ''
    if (isBuiltinCameraName(itemRaw)) {
      continue
    }
    externalIndex += 1
    if (item.stable_id === camera.stable_id) {
      return t('setupExternalCamera', { count: externalIndex })
    }
  }
  return raw || t('camera')
}

export function presentCameraStream(
  camera: Pick<ScannedCamera, 'width' | 'height' | 'fps' | 'preview_url'>,
  t: Translate,
): string {
  if (!camera.preview_url || !camera.width || !camera.height) {
    return t('setupCameraDetecting')
  }
  const fps = camera.fps ? ` @ ${camera.fps}fps` : ''
  return t('setupCameraCurrentStream', {
    width: camera.width,
    height: camera.height,
    fps,
  })
}
