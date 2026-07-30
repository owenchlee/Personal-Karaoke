import { describe, expect, it } from 'vitest'
import { formatRecordedAt, parseContentDispositionFilename, sortRecordingsByRecency } from './recordings'
import type { Recording } from '../types/recording'

const recording = (filename: string, recorded_at: string): Recording => ({
  filename,
  slug: 'test-song',
  title: 'Test Song',
  recorded_at,
})

describe('sortRecordingsByRecency', () => {
  it('returns an empty array unchanged', () => {
    expect(sortRecordingsByRecency([])).toEqual([])
  })

  it('orders newest first', () => {
    const older = recording('a', '2026-01-01T00:00:00Z')
    const newer = recording('b', '2026-06-01T00:00:00Z')

    expect(sortRecordingsByRecency([older, newer])).toEqual([newer, older])
  })

  it('does not mutate the input array', () => {
    const input = [recording('a', '2026-01-01T00:00:00Z'), recording('b', '2026-06-01T00:00:00Z')]
    const copy = [...input]

    sortRecordingsByRecency(input)

    expect(input).toEqual(copy)
  })
})

describe('formatRecordedAt', () => {
  it('returns null for an unparseable timestamp', () => {
    expect(formatRecordedAt('not-a-date')).toBeNull()
  })

  it('formats an ISO timestamp with date and time in UTC', () => {
    expect(formatRecordedAt('2026-07-29T15:45:00Z')).toBe('Jul 29, 2026, 3:45 PM UTC')
  })

  it('pads single-digit minutes', () => {
    expect(formatRecordedAt('2026-07-29T15:05:00Z')).toBe('Jul 29, 2026, 3:05 PM UTC')
  })

  it('renders midnight as 12 AM, not 0 AM', () => {
    expect(formatRecordedAt('2026-07-29T00:15:00Z')).toBe('Jul 29, 2026, 12:15 AM UTC')
  })

  it('renders noon as 12 PM, not 0 PM', () => {
    expect(formatRecordedAt('2026-07-29T12:15:00Z')).toBe('Jul 29, 2026, 12:15 PM UTC')
  })
})

describe('parseContentDispositionFilename', () => {
  it('returns null when the header is missing', () => {
    expect(parseContentDispositionFilename(null)).toBeNull()
  })

  it('extracts a quoted filename', () => {
    expect(parseContentDispositionFilename('attachment; filename="Test Song-2026-07-29-154500.mp3"')).toBe(
      'Test Song-2026-07-29-154500.mp3',
    )
  })

  it('extracts an unquoted filename', () => {
    expect(parseContentDispositionFilename('attachment; filename=recording.mp3')).toBe('recording.mp3')
  })

  it('decodes an RFC 5987 utf-8 filename, e.g. a non-ASCII (Cantonese) song title', () => {
    expect(
      parseContentDispositionFilename("attachment; filename*=utf-8''%E6%B5%81%E6%B0%B4-2026-07-29-154500.mp3"),
    ).toBe('流水-2026-07-29-154500.mp3')
  })
})
