import type { Song } from '../types/song'

/**
 * Newest-first by `processed_at`. Songs with no timestamp (e.g. published via
 * the CLI scripts before this field existed) sort after every timestamped
 * song, keeping their relative order.
 */
export function sortSongsByRecency(songs: Song[]): Song[] {
  return [...songs].sort((a, b) => {
    if (a.processed_at === b.processed_at) return 0
    if (a.processed_at === null) return 1
    if (b.processed_at === null) return -1
    if (a.processed_at < b.processed_at) return 1
    if (a.processed_at > b.processed_at) return -1
    return 0
  })
}

const MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

/**
 * Formats an ISO `processed_at` timestamp for display next to a song in the
 * library (e.g. "Jul 28, 2026"). Uses UTC getters rather than the browser's
 * local timezone so the result is deterministic regardless of where it
 * renders. Returns null for a missing or unparseable timestamp.
 */
export function formatProcessedAt(processedAt: string | null): string | null {
  if (processedAt === null) return null
  const date = new Date(processedAt)
  if (Number.isNaN(date.getTime())) return null
  return `${MONTHS[date.getUTCMonth()]} ${date.getUTCDate()}, ${date.getUTCFullYear()}`
}
