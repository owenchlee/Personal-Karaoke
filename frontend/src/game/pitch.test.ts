import { describe, expect, it } from 'vitest'
import {
  appendTracePoint,
  foldIntoRange,
  foldToNearestOctave,
  hzToMidi,
  isPlausiblePitchJump,
  signedCentsOff,
  smoothMidi,
} from './pitch'

describe('hzToMidi', () => {
  it('converts A4 (440Hz) to MIDI 69', () => {
    expect(hzToMidi(440)).toBeCloseTo(69, 5)
  })

  it('converts middle C (261.63Hz) to MIDI 60', () => {
    expect(hzToMidi(261.63)).toBeCloseTo(60, 1)
  })
})

describe('signedCentsOff', () => {
  it('is zero for an exact match', () => {
    expect(signedCentsOff(60, 60)).toBe(0)
  })

  it('is positive when sharp (detected above reference)', () => {
    expect(signedCentsOff(60.2, 60)).toBe(20)
  })

  it('is negative when flat (detected below reference)', () => {
    expect(signedCentsOff(59.7, 60)).toBe(-30)
  })
})

describe('foldToNearestOctave', () => {
  it('shifts a lower octave up to match the reference', () => {
    expect(foldToNearestOctave(48, 60)).toBe(60) // C3 sung against a C4 reference
  })

  it('shifts a higher octave down to match the reference', () => {
    expect(foldToNearestOctave(72, 60)).toBe(60)
  })

  it('leaves an already-matching pitch class unchanged', () => {
    expect(foldToNearestOctave(61, 60)).toBe(61)
  })

  it('picks the nearer octave for a pitch a few semitones off', () => {
    expect(foldToNearestOctave(58, 60)).toBe(58) // 2 semitones away, same octave is nearer
    expect(foldToNearestOctave(51, 60)).toBe(63) // 9 semitones away, +1 octave is nearer
  })
})

describe('foldIntoRange', () => {
  const range = { min: 60, max: 71 } // C4-B4

  it('leaves a pitch already inside the range unchanged', () => {
    expect(foldIntoRange(64, range)).toBe(64)
  })

  it('shifts a high E (76) down an octave to land in range', () => {
    expect(foldIntoRange(76, range)).toBe(64)
  })

  it('shifts a low E (52) up an octave to land in range', () => {
    expect(foldIntoRange(52, range)).toBe(64)
  })

  it('shifts down multiple octaves if needed', () => {
    expect(foldIntoRange(100, range)).toBe(64)
  })

  it('picks whichever octave lands closest when the range is narrower than an octave', () => {
    const narrow = { min: 60, max: 62 }
    // 63 is already only 1 semitone above the range; shifting an octave either way lands much
    // farther away (51 is 9 below, 75 is 13 above), so it's left as the closest fit.
    expect(foldIntoRange(63, narrow)).toBe(63)
  })

  it('returns the input unchanged for an invalid (min > max) range', () => {
    expect(foldIntoRange(64, { min: 10, max: 0 })).toBe(64)
  })
})

describe('isPlausiblePitchJump', () => {
  it('accepts a small frame-to-frame move', () => {
    expect(isPlausiblePitchJump(60, 62)).toBe(true)
  })

  it('accepts no change at all', () => {
    expect(isPlausiblePitchJump(60, 60)).toBe(true)
  })

  it('rejects a jump bigger than a real voice can make in one frame', () => {
    expect(isPlausiblePitchJump(60, 65)).toBe(false)
    expect(isPlausiblePitchJump(60, 55)).toBe(false)
  })

  it('is symmetric in direction', () => {
    expect(isPlausiblePitchJump(60, 66)).toBe(isPlausiblePitchJump(66, 60))
  })

  it('respects a custom threshold', () => {
    expect(isPlausiblePitchJump(60, 63, 2)).toBe(false)
    expect(isPlausiblePitchJump(60, 63, 4)).toBe(true)
  })
})

describe('smoothMidi', () => {
  it('returns null for an empty window', () => {
    expect(smoothMidi([])).toBeNull()
  })

  it('returns the value itself for a single sample', () => {
    expect(smoothMidi([60])).toBe(60)
  })

  it('is resistant to a single-frame outlier, unlike a mean would be', () => {
    // A brief misread (e.g. an octave blip) shouldn't swing the result nearly as far as it would
    // swing a plain average.
    expect(smoothMidi([60, 60, 60, 60, 72])).toBe(60)
  })

  it('averages the two middle values for an even-sized window', () => {
    expect(smoothMidi([60, 62])).toBe(61)
  })
})

describe('appendTracePoint', () => {
  it('appends the new point and keeps it', () => {
    const trace = appendTracePoint([], { time: 1, midi: 60 }, 2)
    expect(trace).toEqual([{ time: 1, midi: 60 }])
  })

  it('drops points older than maxAgeSeconds relative to the new point', () => {
    const trace = appendTracePoint(
      [
        { time: 0, midi: 60 },
        { time: 0.9, midi: 61 },
      ],
      { time: 2, midi: 62 },
      1.5,
    )
    expect(trace).toEqual([
      { time: 0.9, midi: 61 },
      { time: 2, midi: 62 },
    ])
  })
})
