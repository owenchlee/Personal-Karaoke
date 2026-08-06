import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import {
  clearCalibrationOffsetSeconds,
  computeCalibrationResult,
  hasCalibrationOffset,
  loadCalibrationOffsetSeconds,
  pairBeatsToTaps,
  saveCalibrationOffsetSeconds,
} from './calibration'

function closeArray(values: number[], expected: number[]) {
  expect(values).toHaveLength(expected.length)
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 6))
}

describe('pairBeatsToTaps', () => {
  it('pairs each beat with the nearest tap, in order', () => {
    const beats = [1, 2, 3, 4]
    const taps = [1.05, 2.04, 3.06, 4.03]
    closeArray(pairBeatsToTaps(beats, taps), [0.05, 0.04, 0.06, 0.03])
  })

  it('drops a beat with no tap inside the match window', () => {
    const beats = [1, 2, 3]
    const taps = [1.05, 3.05] // beat 2 has no nearby tap
    closeArray(pairBeatsToTaps(beats, taps), [0.05, 0.05])
  })

  it('never uses the same tap for two beats', () => {
    const beats = [1, 1.1]
    const taps = [1.05]
    // Beat 1 claims the only nearby tap; beat 1.1 is left unmatched rather than double-counting it.
    closeArray(pairBeatsToTaps(beats, taps), [0.05])
  })

  it('ignores taps unrelated to any beat (outside the match window)', () => {
    const beats = [1, 2]
    const taps = [1.02, 50] // stray noise far from any beat
    closeArray(pairBeatsToTaps(beats, taps), [0.02])
  })

  it('respects a custom match window', () => {
    const beats = [1]
    const taps = [1.3]
    expect(pairBeatsToTaps(beats, taps, 0.2)).toEqual([])
    closeArray(pairBeatsToTaps(beats, taps, 0.4), [0.3])
  })
})

describe('computeCalibrationResult', () => {
  it('returns null when too few beats matched', () => {
    const beats = [1, 2, 3]
    const taps = [1.05, 2.05] // only 2 matches, below the minimum
    expect(computeCalibrationResult(beats, taps)).toBeNull()
  })

  it('computes the median offset across matched beats, resisting one mistimed tap', () => {
    const beats = [1, 2, 3, 4, 5]
    // Four consistent ~50ms-late taps, one much-later outlier (still inside the match window) --
    // the median should sit near the consistent cluster rather than being dragged toward it.
    const taps = [1.05, 2.05, 3.05, 4.05, 5.35]
    const result = computeCalibrationResult(beats, taps)
    expect(result).not.toBeNull()
    expect(result!.offsetSeconds).toBeCloseTo(0.05, 5)
    expect(result!.matchedBeats).toBe(5)
    expect(result!.totalBeats).toBe(5)
  })

  it('reports a negative offset when taps consistently arrive early', () => {
    const beats = [1, 2, 3, 4]
    const taps = [0.97, 1.98, 2.96, 3.97]
    const result = computeCalibrationResult(beats, taps)
    expect(result!.offsetSeconds).toBeLessThan(0)
  })
})

describe('calibration offset storage', () => {
  // Pure `game/*.ts` modules in this project are deliberately tested without a DOM environment
  // (see the other `game/*.test.ts` files) -- a minimal in-memory `localStorage` stub here keeps
  // that convention rather than pulling in jsdom for the whole suite just for this one module.
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
    clearCalibrationOffsetSeconds()
  })

  it('defaults to 0 when nothing has been saved', () => {
    expect(loadCalibrationOffsetSeconds()).toBe(0)
  })

  it('round-trips a saved value', () => {
    saveCalibrationOffsetSeconds(0.0423)
    expect(loadCalibrationOffsetSeconds()).toBeCloseTo(0.0423, 5)
  })

  it('clears back to the 0 default', () => {
    saveCalibrationOffsetSeconds(0.1)
    clearCalibrationOffsetSeconds()
    expect(loadCalibrationOffsetSeconds()).toBe(0)
  })

  it('distinguishes "never calibrated" from a real result of exactly 0', () => {
    expect(hasCalibrationOffset()).toBe(false)
    saveCalibrationOffsetSeconds(0)
    expect(hasCalibrationOffset()).toBe(true)
    expect(loadCalibrationOffsetSeconds()).toBe(0)
  })
})
