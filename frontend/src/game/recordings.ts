import type { Recording } from '../types/recording'
import { MONTHS } from './dateFormat'

/** Pulls the filename the server suggested (via `_recording_download_name` in
 * scripts/server.py) out of a `Content-Disposition` response header, so the
 * browser's immediate post-recording download uses the exact same "song
 * name, then timestamp" naming the "My recordings" list's download links use
 * -- rather than a second, independently-hardcoded name drifting out of sync
 * with it. Returns null if the header is missing or has no filename part.
 *
 * Checks the RFC 5987 `filename*=utf-8''...` form first -- Starlette's
 * `FileResponse` (what the server uses) sends *only* that form, percent-
 * encoded, whenever the filename has any non-ASCII character, which is a
 * real case here: this app supports Cantonese ("yue") songs, so a recording
 * title can easily be non-ASCII, not just an exotic edge case. */
export function parseContentDispositionFilename(header: string | null): string | null {
  if (!header) return null

  const extended = header.match(/filename\*=utf-8''([^;]+)/i)
  if (extended) {
    try {
      return decodeURIComponent(extended[1])
    } catch {
      return null
    }
  }

  const plain = header.match(/filename="?([^";]+)"?/)
  return plain ? plain[1] : null
}

/** Newest-first by `recorded_at`. Unlike songs' `processed_at`, every
 * recording always has a timestamp (it's generated server-side at save
 * time), so there's no "missing timestamp" case to sort last. */
export function sortRecordingsByRecency(recordings: Recording[]): Recording[] {
  return [...recordings].sort((a, b) => {
    if (a.recorded_at === b.recorded_at) return 0
    return a.recorded_at < b.recorded_at ? 1 : -1
  })
}

/**
 * Formats an ISO `recorded_at` timestamp for display in the recordings list
 * (e.g. "Jul 29, 2026, 3:45 PM UTC"). Includes the time (unlike
 * `formatProcessedAt` for songs) since multiple recordings of the same song
 * on the same day need to be told apart. Uses UTC getters, labeled
 * explicitly, so the result is deterministic regardless of where it renders.
 * Returns null for a missing or unparseable timestamp.
 */
export function formatRecordedAt(recordedAt: string): string | null {
  const date = new Date(recordedAt)
  if (Number.isNaN(date.getTime())) return null

  const hours24 = date.getUTCHours()
  const period = hours24 >= 12 ? 'PM' : 'AM'
  const hours12 = hours24 % 12 === 0 ? 12 : hours24 % 12
  const minutes = date.getUTCMinutes().toString().padStart(2, '0')

  return (
    `${MONTHS[date.getUTCMonth()]} ${date.getUTCDate()}, ${date.getUTCFullYear()}, ` +
    `${hours12}:${minutes} ${period} UTC`
  )
}
