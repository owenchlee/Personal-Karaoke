import { describe, expect, it } from 'vitest'
import { formatProcessedAt, sortSongLibrary, sortSongsByRecency } from './songLibrary'
import type { Song } from '../types/song'

const song = (slug: string, processed_at: string | null, starred = false): Song => ({
  slug,
  title: slug,
  processed_at,
  starred,
})

describe('sortSongsByRecency', () => {
  it('returns an empty array unchanged', () => {
    expect(sortSongsByRecency([])).toEqual([])
  })

  it('orders newest first', () => {
    const older = song('a', '2026-01-01T00:00:00Z')
    const newer = song('b', '2026-06-01T00:00:00Z')

    expect(sortSongsByRecency([older, newer])).toEqual([newer, older])
  })

  it('sorts songs with no timestamp after every timestamped song, preserving their relative order', () => {
    const timed = song('a', '2026-01-01T00:00:00Z')
    const untimed1 = song('b', null)
    const untimed2 = song('c', null)

    expect(sortSongsByRecency([untimed1, timed, untimed2])).toEqual([timed, untimed1, untimed2])
  })

  it('does not mutate the input array', () => {
    const input = [song('a', '2026-01-01T00:00:00Z'), song('b', '2026-06-01T00:00:00Z')]
    const copy = [...input]

    sortSongsByRecency(input)

    expect(input).toEqual(copy)
  })
})

describe('sortSongLibrary', () => {
  it('returns an empty array unchanged', () => {
    expect(sortSongLibrary([])).toEqual([])
  })

  it('puts starred songs before unstarred, regardless of recency', () => {
    const olderStarred = song('a', '2026-01-01T00:00:00Z', true)
    const newerUnstarred = song('b', '2026-06-01T00:00:00Z', false)

    expect(sortSongLibrary([newerUnstarred, olderStarred])).toEqual([olderStarred, newerUnstarred])
  })

  it('sorts newest first within the starred and unstarred groups', () => {
    const starredOld = song('a', '2026-01-01T00:00:00Z', true)
    const starredNew = song('b', '2026-03-01T00:00:00Z', true)
    const unstarredOld = song('c', '2026-02-01T00:00:00Z', false)
    const unstarredNew = song('d', '2026-04-01T00:00:00Z', false)

    expect(sortSongLibrary([starredOld, unstarredOld, starredNew, unstarredNew])).toEqual([
      starredNew,
      starredOld,
      unstarredNew,
      unstarredOld,
    ])
  })

  it('does not mutate the input array', () => {
    const input = [song('a', '2026-01-01T00:00:00Z', false), song('b', '2026-06-01T00:00:00Z', true)]
    const copy = [...input]

    sortSongLibrary(input)

    expect(input).toEqual(copy)
  })
})

describe('formatProcessedAt', () => {
  it('returns null for a missing timestamp', () => {
    expect(formatProcessedAt(null)).toBeNull()
  })

  it('returns null for an unparseable timestamp', () => {
    expect(formatProcessedAt('not-a-date')).toBeNull()
  })

  it('formats an ISO timestamp in UTC, independent of local timezone', () => {
    expect(formatProcessedAt('2026-07-28T18:49:02.261568+00:00')).toBe('Jul 28, 2026')
  })

  it('does not zero-pad single-digit days', () => {
    expect(formatProcessedAt('2026-01-05T00:00:00Z')).toBe('Jan 5, 2026')
  })
})
