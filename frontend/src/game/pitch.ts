import type { PitchRange } from './coords'

export function hzToMidi(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440)
}

/** Signed cents between `detectedMidi` and `referenceMidi` (positive = sharp, negative = flat),
 * for a human-readable "how far off" readout. Assumes `detectedMidi` is already octave-folded
 * close to `referenceMidi` (e.g. via `foldToNearestOctave`) -- this does no wraparound of its
 * own, so a caller passing two far-apart pitches would get a large, not-cent-scale number back. */
export function signedCentsOff(detectedMidi: number, referenceMidi: number): number {
  return Math.round((detectedMidi - referenceMidi) * 100)
}

/** Shifts `rawMidi` by whole octaves to land as close as possible to `referenceMidi`, for
 * drawing a sung pitch aligned with whichever reference note it's being compared against
 * (scoring itself is octave-agnostic -- this is purely so the live trace visually lines up). */
export function foldToNearestOctave(rawMidi: number, referenceMidi: number): number {
  const octaveShift = Math.round((referenceMidi - rawMidi) / 12)
  return rawMidi + octaveShift * 12
}

function distanceOutsideRange(value: number, range: PitchRange): number {
  if (value < range.min) return range.min - value
  if (value > range.max) return value - range.max
  return 0
}

/** Shifts `midi` by whole octaves to land inside `range` (e.g. a high E above the top of the
 * highway maps down to the nearest lower E that's on-screen), always preserving pitch class, so
 * the indicator never drifts off the visible pitch range. Picks whichever octave shift ends up
 * closest to (or inside) the range if `range` is narrower than an octave and none lands exactly
 * inside it. */
export function foldIntoRange(midi: number, range: PitchRange): number {
  if (range.min > range.max) return midi

  let best = midi
  let bestDistance = distanceOutsideRange(midi, range)
  for (let octaves = -4; octaves <= 4; octaves++) {
    const candidate = midi + octaves * 12
    const distance = distanceOutsideRange(candidate, range)
    if (distance < bestDistance) {
      best = candidate
      bestDistance = distance
    }
  }
  return best
}

const MAX_PLAUSIBLE_JUMP_SEMITONES = 4

/** Whether moving from `previousMidi` to `nextMidi` (both already octave-folded to the same
 * anchor) is a plausible single-frame vocal pitch change, as opposed to a pitch-detector
 * misread. `foldToNearestOctave` only corrects whole-octave errors -- a detector that locks onto
 * a false fifth or third of the true fundamental (common with vocal timbre) still passes through
 * it, landing up to 6 semitones from the anchor since that's as close as *any* octave of a wrong
 * pitch class can get. A real singer's fundamental frequency can't swing more than a few
 * semitones between two ~16ms detection frames, so a jump bigger than that is much more likely
 * detector noise than an actual note change -- see `NoteHighway`, which drops such samples
 * instead of plotting them. */
export function isPlausiblePitchJump(
  previousMidi: number,
  nextMidi: number,
  maxSemitones = MAX_PLAUSIBLE_JUMP_SEMITONES,
): boolean {
  return Math.abs(nextMidi - previousMidi) <= maxSemitones
}

export interface TracePoint {
  time: number
  midi: number
}

/** Appends `point` to `trace` and drops anything older than `maxAgeSeconds` relative to it --
 * used to keep a short rolling window of recent samples (for smoothing), not an ever-growing
 * history. */
export function appendTracePoint(
  trace: TracePoint[],
  point: TracePoint,
  maxAgeSeconds: number,
): TracePoint[] {
  const cutoff = point.time - maxAgeSeconds
  return [...trace.filter((p) => p.time >= cutoff), point]
}

/** Median of recent midi values. Raw frame-to-frame pitch detection is noisy (vibrato, formants,
 * occasional misreads), and a single detected sample jumps around too much to plot directly --
 * the median of a short recent window is far more resistant to single-frame outliers than a mean
 * would be. Returns null for an empty window. */
export function smoothMidi(midiValues: number[]): number | null {
  if (midiValues.length === 0) return null
  const sorted = [...midiValues].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}
