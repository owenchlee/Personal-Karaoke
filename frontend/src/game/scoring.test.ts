import { describe, expect, it } from 'vitest'
import {
  accuracyFraction,
  createEmptyAccuracy,
  findActiveNoteIndex,
  isNoteHit,
  isPitchMatch,
  pitchClassDistanceCents,
  recordSample,
  songAccuracyScore,
} from './scoring'
import type { NoteEvent } from '../types/note'

function makeNote(overrides: Partial<NoteEvent> = {}): NoteEvent {
  return {
    pitch_midi: 60,
    pitch_hz: 261.63,
    onset: 1,
    offset: 2,
    duration: 1,
    velocity: 0.8,
    ...overrides,
  }
}

describe('pitchClassDistanceCents', () => {
  it('is 0 for the same pitch class in different octaves', () => {
    expect(pitchClassDistanceCents(60, 72)).toBe(0)
    expect(pitchClassDistanceCents(60, 48)).toBe(0)
  })

  it('is 100 cents per semitone within an octave', () => {
    expect(pitchClassDistanceCents(60, 61)).toBe(100)
  })

  it('wraps around the octave boundary (takes the shorter direction)', () => {
    expect(pitchClassDistanceCents(60, 71)).toBe(100) // 11 semitones one way, 1 the other
  })
})

describe('isPitchMatch', () => {
  it('matches an exact pitch class regardless of octave', () => {
    expect(isPitchMatch(48, 60)).toBe(true)
  })

  it('matches within the default 100 cent tolerance', () => {
    expect(isPitchMatch(60.9, 60)).toBe(true)
  })

  it('rejects a pitch outside tolerance', () => {
    expect(isPitchMatch(62, 60)).toBe(false)
  })
})

describe('accuracy accumulation', () => {
  it('starts empty', () => {
    expect(accuracyFraction(createEmptyAccuracy())).toBe(0)
  })

  it('accumulates hits and misses into a fraction', () => {
    let accuracy = createEmptyAccuracy()
    accuracy = recordSample(accuracy, true)
    accuracy = recordSample(accuracy, true)
    accuracy = recordSample(accuracy, false)
    expect(accuracy).toEqual({ hitCount: 2, totalCount: 3 })
    expect(accuracyFraction(accuracy)).toBeCloseTo(2 / 3, 5)
  })
})

describe('findActiveNoteIndex', () => {
  const notes = [makeNote({ onset: 1, offset: 2 }), makeNote({ onset: 5, offset: 6 })]

  it('finds the note covering the given time', () => {
    expect(findActiveNoteIndex(notes, 1.5)).toBe(0)
    expect(findActiveNoteIndex(notes, 5.5)).toBe(1)
  })

  it('returns -1 in a gap between notes', () => {
    expect(findActiveNoteIndex(notes, 3)).toBe(-1)
  })
})

describe('isNoteHit', () => {
  it('is false for a note with no samples', () => {
    expect(isNoteHit(createEmptyAccuracy())).toBe(false)
  })

  it('is true once at least half the samples matched', () => {
    let accuracy = createEmptyAccuracy()
    accuracy = recordSample(accuracy, true)
    accuracy = recordSample(accuracy, false)
    expect(isNoteHit(accuracy)).toBe(true)
  })

  it('is false when fewer than half the samples matched', () => {
    let accuracy = createEmptyAccuracy()
    accuracy = recordSample(accuracy, true)
    accuracy = recordSample(accuracy, false)
    accuracy = recordSample(accuracy, false)
    expect(isNoteHit(accuracy)).toBe(false)
  })
})

describe('songAccuracyScore', () => {
  const notes = [
    makeNote({ onset: 0, offset: 1 }),
    makeNote({ onset: 2, offset: 3 }),
    makeNote({ onset: 4, offset: 5 }),
    makeNote({ onset: 6, offset: 7 }),
  ]

  it('is 100 with no notes at all', () => {
    expect(songAccuracyScore([], [], 0)).toBe(100)
  })

  it('starts at 100 before any note has finished, even with no samples yet', () => {
    const accuracies = notes.map(() => createEmptyAccuracy())
    expect(songAccuracyScore(notes, accuracies, 0.5)).toBe(100)
  })

  it('does not penalize a note that has not finished yet', () => {
    const accuracies = notes.map(() => createEmptyAccuracy())
    expect(songAccuracyScore(notes, accuracies, 0.9)).toBe(100)
  })

  it('docks one share of 100/notes.length per finished note missed', () => {
    const accuracies = [
      recordSample(createEmptyAccuracy(), true), // hit
      createEmptyAccuracy(), // missed (no samples, finished)
      createEmptyAccuracy(), // missed (no samples, finished)
      createEmptyAccuracy(), // not finished yet -- no penalty
    ]
    // Notes 0-2 have finished by t=5.5; note 1 and 2 were missed out of 4 bars total.
    expect(songAccuracyScore(notes, accuracies, 5.5)).toBe(50)
  })

  it('reaches 0 when every note in the song is missed', () => {
    const accuracies = notes.map(() => createEmptyAccuracy())
    expect(songAccuracyScore(notes, accuracies, 100)).toBe(0)
  })
})
