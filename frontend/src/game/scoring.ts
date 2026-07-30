import type { NoteEvent } from '../types/note'

// Widened from 50 -> 70 -> 80 across earlier "reduce difficulty" passes, then to 100 (a full
// semitone either way) on further request.
const CENTS_TOLERANCE = 100

/** Distance in cents between two pitches' *pitch classes* (octave-agnostic), 0-600. */
export function pitchClassDistanceCents(a: number, b: number): number {
  const diffSemitones = Math.abs(a - b) % 12
  const folded = Math.min(diffSemitones, 12 - diffSemitones)
  return folded * 100
}

export function isPitchMatch(
  sampleMidi: number,
  noteMidi: number,
  toleranceCents = CENTS_TOLERANCE,
): boolean {
  return pitchClassDistanceCents(sampleMidi, noteMidi) <= toleranceCents
}

export interface NoteAccuracy {
  hitCount: number
  totalCount: number
}

export function createEmptyAccuracy(): NoteAccuracy {
  return { hitCount: 0, totalCount: 0 }
}

export function recordSample(accuracy: NoteAccuracy, isHit: boolean): NoteAccuracy {
  return {
    hitCount: accuracy.hitCount + (isHit ? 1 : 0),
    totalCount: accuracy.totalCount + 1,
  }
}

export function accuracyFraction(accuracy: NoteAccuracy): number {
  return accuracy.totalCount === 0 ? 0 : accuracy.hitCount / accuracy.totalCount
}

// Lowered from 0.5 -> 0.4 -> 0.35 across "reduce difficulty" passes -- see NOTES.md.
const NOTE_HIT_FRACTION_THRESHOLD = 0.35

/** Whether a note counts as "hit" -- matched pitch for at least `NOTE_HIT_FRACTION_THRESHOLD` of
 * the samples taken while it was active. A note with no samples at all (never sung) is not a hit. */
export function isNoteHit(accuracy: NoteAccuracy): boolean {
  return accuracy.totalCount > 0 && accuracyFraction(accuracy) >= NOTE_HIT_FRACTION_THRESHOLD
}

/** Index of the note covering `time`, or -1 if `time` falls in a gap between notes. */
export function findActiveNoteIndex(notes: NoteEvent[], time: number): number {
  return notes.findIndex((note) => time >= note.onset && time <= note.offset)
}

/** Score starts at 100 and is docked one `100 / notes.length` share for every note that's
 * finished (its offset has passed) without being hit -- so missing every bar in the song brings
 * the score to 0, rather than averaging only the notes attempted so far. Notes still upcoming
 * don't count against it yet. */
export function songAccuracyScore(
  notes: NoteEvent[],
  accuracies: NoteAccuracy[],
  currentTime: number,
): number {
  if (notes.length === 0) return 100
  let missed = 0
  for (let i = 0; i < notes.length; i++) {
    if (notes[i].offset <= currentTime && !isNoteHit(accuracies[i])) missed++
  }
  return Math.round(100 - (missed / notes.length) * 100)
}
