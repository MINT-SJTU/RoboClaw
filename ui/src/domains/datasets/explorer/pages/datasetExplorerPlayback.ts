export interface EpisodeVideo {
  path: string
  url: string
  stream: string
  from_timestamp?: number | null
  to_timestamp?: number | null
}

const CLIP_TIME_EPSILON = 0.033

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function getClipStart(video: EpisodeVideo | null | undefined): number {
  return isFiniteNumber(video?.from_timestamp) ? video.from_timestamp : 0
}

function getClipEnd(video: EpisodeVideo | null | undefined): number | null {
  return isFiniteNumber(video?.to_timestamp) ? video.to_timestamp : null
}

export function clampAbsolutePlaybackTime(
  video: EpisodeVideo | null | undefined,
  absoluteTime: number,
  duration: number,
  options: { loopToStart?: boolean } = {},
): number {
  const clipStart = getClipStart(video)
  const clipEnd = getClipEnd(video)
  let nextTime = Number.isFinite(absoluteTime) ? absoluteTime : clipStart

  if (
    options.loopToStart &&
    isFiniteNumber(clipEnd) &&
    nextTime >= clipEnd - CLIP_TIME_EPSILON
  ) {
    return clipStart
  }

  nextTime = Math.max(nextTime, clipStart)
  if (isFiniteNumber(clipEnd)) {
    nextTime = Math.min(nextTime, clipEnd)
  }
  if (Number.isFinite(duration)) {
    nextTime = Math.min(nextTime, duration)
  }
  return nextTime
}

export function getRelativePlaybackTime(
  video: EpisodeVideo | null | undefined,
  absoluteTime: number,
): number {
  return Math.max(absoluteTime - getClipStart(video), 0)
}

export function shouldLoopVideo(video: EpisodeVideo | null | undefined): boolean {
  return !isFiniteNumber(getClipEnd(video))
}

export function formatClipWindowLabel(video: EpisodeVideo | null | undefined): string {
  const clipStart = getClipStart(video)
  const clipEnd = getClipEnd(video)
  if (!isFiniteNumber(clipEnd) && clipStart <= 0) {
    return ''
  }
  const start = clipStart.toFixed(2)
  if (!isFiniteNumber(clipEnd)) {
    return `${start}s+`
  }
  return `${start}s-${clipEnd.toFixed(2)}s`
}
