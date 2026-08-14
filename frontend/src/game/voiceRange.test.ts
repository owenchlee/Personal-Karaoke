import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import type { NoteEvent } from '../types/note'
import {
  clearVoiceRange,
  computeBestShift,
  hasVoiceRange,
  loadVoiceRange,
  saveVoiceRange,
} from './voiceRange'

function note(pitch_midi: number, duration = 1): NoteEvent {
  return { pitch_midi, pitch_hz: 0, onset: 0, offset: duration, duration, velocity: 0.5 }
}

describe('computeBestShift', () => {
  it('picks shift 0 when the song already fits entirely inside the voice range', () => {
    const notes = [note(55), note(60), note(65)]
    expect(computeBestShift(notes, { lowMidi: 50, highMidi: 70 })).toBe(0)
  })

  it('shifts down when the song sits entirely above the voice range', () => {
    // Song sits at MIDI 70-74, voice range is 50-60 -- shifting down by 12 (one octave) lands it
    // at 58-62, mostly inside the range and far better than any smaller shift.
    const notes = [note(70), note(72), note(74)]
    const shift = computeBestShift(notes, { lowMidi: 50, highMidi: 60 })
    expect(shift).toBeLessThan(0)
  })

  it('shifts up when the song sits entirely below the voice range', () => {
    const notes = [note(40), note(42), note(44)]
    const shift = computeBestShift(notes, { lowMidi: 50, highMidi: 62 })
    expect(shift).toBeGreaterThan(0)
  })

  it('is duration-weighted -- a rare short outlier note does not dominate the fit', () => {
    // One brief high note (0.1s) way outside the range shouldn't drag the whole song's shift
    // around when the other 10s of singing already fits comfortably at shift 0.
    const notes = [note(55, 10), note(85, 0.1)]
    expect(computeBestShift(notes, { lowMidi: 50, highMidi: 60 })).toBe(0)
  })

  it('breaks ties toward the smallest absolute shift', () => {
    // A single note at MIDI 60 sits outside a narrow [70,71] voice range at every shift equally
    // badly in terms of "still out of range", but shift 0 changes the original key least.
    const notes = [note(0, 1)] // absurdly low, unreachable at any shift within the search window
    const shift = computeBestShift(notes, { lowMidi: 200, highMidi: 201 }, 3)
    expect(shift).toBe(0)
  })

  it('returns 0 for a song with no notes', () => {
    expect(computeBestShift([], { lowMidi: 50, highMidi: 70 })).toBe(0)
  })
})

describe('voice range storage', () => {
  // Pure `game/*.ts` modules in this project are deliberately tested without a DOM environment
  // (see game/calibration.test.ts) -- a minimal in-memory `localStorage` stub keeps that
  // convention instead of pulling in jsdom for the whole suite just for this one module.
  class MemoryStorage {
    private store = new Map<string, string>()
    getItem(key: string) {
      return this.store.has(key) ? this.store.get(key)! : null
    }
    setItem(key: string, value: string) {
      this.store.set(key, value)
    }
    removeItem(key: string) {
      this.store.delete(key)
    }
  }

  beforeAll(() => {
    ;(globalThis as Record<string, unknown>).window = { localStorage: new MemoryStorage() }
  })
  afterAll(() => {
    delete (globalThis as Record<string, unknown>).window
  })
  afterEach(() => {
    clearVoiceRange()
  })

  it('defaults to null when nothing has been saved', () => {
    expect(loadVoiceRange()).toBeNull()
    expect(hasVoiceRange()).toBe(false)
  })

  it('round-trips a saved range', () => {
    saveVoiceRange({ lowMidi: 48, highMidi: 67 })
    expect(loadVoiceRange()).toEqual({ lowMidi: 48, highMidi: 67 })
    expect(hasVoiceRange()).toBe(true)
  })

  it('clears back to null', () => {
    saveVoiceRange({ lowMidi: 48, highMidi: 67 })
    clearVoiceRange()
    expect(loadVoiceRange()).toBeNull()
  })
})
