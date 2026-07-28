import { describe, expect, it } from 'vitest'
import { getPitchRange, getVisibleNotes, pitchToY, timeToX } from './coords'
import type { NoteEvent } from '../types/note'

function makeNote(overrides: Partial<NoteEvent>): NoteEvent {
  return {
    pitch_midi: 60,
    pitch_hz: 261.63,
    onset: 0,
    offset: 1,
    duration: 1,
    velocity: 0.5,
    ...overrides,
  }
}

describe('getPitchRange', () => {
  it('pads the min/max pitch of a normal range', () => {
    const notes = [makeNote({ pitch_midi: 50 }), makeNote({ pitch_midi: 70 }), makeNote({ pitch_midi: 60 })]
    expect(getPitchRange(notes)).toEqual({ min: 48, max: 72 })
  })

  it('handles a single distinct pitch without collapsing the range', () => {
    const notes = [makeNote({ pitch_midi: 60 }), makeNote({ pitch_midi: 60 })]
    const range = getPitchRange(notes)
    expect(range.max).toBeGreaterThan(range.min)
  })

  it('returns a default range for an empty note list', () => {
    expect(getPitchRange([])).toEqual({ min: 48, max: 72 })
  })
})

describe('pitchToY', () => {
  const range = { min: 48, max: 72 }

  it('maps the max pitch to the top margin', () => {
    expect(pitchToY(72, range, 200, 10, 10)).toBeCloseTo(10)
  })

  it('maps the min pitch to canvasHeight minus the bottom margin', () => {
    expect(pitchToY(48, range, 200, 10, 10)).toBeCloseTo(190)
  })

  it('maps the midpoint pitch to the vertical center of the usable area', () => {
    expect(pitchToY(60, range, 200, 10, 10)).toBeCloseTo(100)
  })
})

describe('timeToX', () => {
  it('places a note at the playhead when its time equals currentTime', () => {
    expect(timeToX(5, 5, 200, 100)).toBe(100)
  })

  it('places a future note to the right of the playhead', () => {
    expect(timeToX(6, 5, 200, 100)).toBe(300)
  })

  it('places a past note to the left of the playhead', () => {
    expect(timeToX(4, 5, 200, 100)).toBe(-100)
  })
})

describe('getVisibleNotes', () => {
  const notes = [
    makeNote({ onset: 0, offset: 1 }),
    makeNote({ onset: 10, offset: 11 }),
    makeNote({ onset: 100, offset: 101 }),
  ]

  it('excludes notes far outside the visible window', () => {
    const visible = getVisibleNotes(notes, 10.5, 0.5, 5)
    expect(visible).toEqual([notes[1]])
  })

  it('includes a note still within the past buffer', () => {
    const visible = getVisibleNotes(notes, 1.4, 0.5, 5)
    expect(visible).toContainEqual(notes[0])
  })

  it('returns an empty array for an empty note list', () => {
    expect(getVisibleNotes([], 0, 0.5, 5)).toEqual([])
  })
})
