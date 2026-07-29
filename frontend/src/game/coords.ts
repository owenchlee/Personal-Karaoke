import type { NoteEvent } from '../types/note'

export interface PitchRange {
  min: number
  max: number
}

const PITCH_PAD_SEMITONES = 2
const DEFAULT_PITCH_RANGE: PitchRange = { min: 48, max: 72 }

// Lowered from 200 -- see the "reduce difficulty" pass in NOTES.md. A slower scroll gives more
// reaction time before a note reaches the playhead without changing the underlying note timing
// (still driven straight off `audioRef.current.currentTime`, so it stays sample-accurate).
export const DEFAULT_PX_PER_SECOND = 160
export const DEFAULT_PAST_BUFFER_SECONDS = 0.5

/** Mapping strategy: the vertical axis is *relative*, not absolute -- it always auto-fits to
 * exactly this song's own sung range (lowest note to highest note, +/- `PITCH_PAD_SEMITONES`),
 * then `pitchToY` linearly interpolates any given pitch within that span to a canvas y. This
 * means the same MIDI note (e.g. 60 = C4) lands at a different pixel row in two different songs,
 * and -- the part that reads as a bug if you don't know it -- a note can be the literal lowest
 * note in the whole song (so it renders pinned to the very bottom of the highway) while still
 * being an objectively unremarkable pitch, if that same song separately has a much higher note
 * elsewhere (e.g. a belted climax) stretching the span. That's not a mapping error: verified
 * against `jamie-miller-with-salem-ilese-here-s-your-perfect-lyrics`, whose intro sits at MIDI
 * ~54 (F#3) purely because the same song's bridge independently hits MIDI 85 (C#6, a genuine
 * belt, confirmed with an independent pyin pitch tracker against the vocal stem) -- a ~2.7 octave
 * span, wider than most songs. The intro note is really the bottom of *this* song, not "low"
 * in any absolute sense. `getPitchAxisTicks`/`midiToNoteName` exist so the UI can label gridlines
 * with real note names, making that relative-vs-absolute distinction visible instead of implicit. */
export function getPitchRange(notes: NoteEvent[]): PitchRange {
  if (notes.length === 0) {
    return DEFAULT_PITCH_RANGE
  }
  let min = notes[0].pitch_midi
  let max = notes[0].pitch_midi
  for (const note of notes) {
    if (note.pitch_midi < min) min = note.pitch_midi
    if (note.pitch_midi > max) max = note.pitch_midi
  }
  return { min: min - PITCH_PAD_SEMITONES, max: max + PITCH_PAD_SEMITONES }
}

export function pitchToY(
  pitch: number,
  range: PitchRange,
  canvasHeight: number,
  marginTop: number,
  marginBottom: number,
): number {
  const usableHeight = canvasHeight - marginTop - marginBottom
  const span = range.max - range.min
  const normalized = span === 0 ? 0.5 : (pitch - range.min) / span
  return marginTop + (1 - normalized) * usableHeight
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

/** Standard scientific pitch notation (MIDI 60 = C4, MIDI 69 = A4 = 440Hz). */
export function midiToNoteName(midi: number): string {
  const rounded = Math.round(midi)
  const octave = Math.floor(rounded / 12) - 1
  const name = NOTE_NAMES[((rounded % 12) + 12) % 12]
  return `${name}${octave}`
}

const TARGET_AXIS_TICK_COUNT = 6

/** Picks evenly-spaced semitone gridlines to label on the pitch axis. Step size scales with the
 * range's span (rounded to a whole number of semitones, minimum 1) so a wide-ranging song doesn't
 * cram in an unreadable label every semitone while a narrow one still gets a useful number of
 * them -- aiming for roughly `TARGET_AXIS_TICK_COUNT` labels regardless of span. This is what
 * turns the otherwise-implicit relative mapping in `getPitchRange`/`pitchToY` into something a
 * player can actually read (e.g. seeing "F#3" next to the intro and "C#6" next to the climax
 * makes clear those are real, very different notes, not the same "low" the auto-fit bottom edge
 * would otherwise imply).*/
export function getPitchAxisTicks(range: PitchRange): number[] {
  const span = range.max - range.min
  if (span <= 0) return [range.min]
  const step = Math.max(1, Math.round(span / TARGET_AXIS_TICK_COUNT))
  const ticks: number[] = []
  const start = Math.ceil(range.min / step) * step
  for (let midi = start; midi <= range.max; midi += step) {
    ticks.push(midi)
  }
  return ticks
}

export function timeToX(
  noteTime: number,
  currentTime: number,
  pxPerSecond: number,
  playheadX: number,
): number {
  return playheadX + (noteTime - currentTime) * pxPerSecond
}

export function getVisibleNotes(
  notes: NoteEvent[],
  currentTime: number,
  pastBufferSeconds: number,
  futureWindowSeconds: number,
): NoteEvent[] {
  return notes.filter(
    (note) =>
      note.offset >= currentTime - pastBufferSeconds &&
      note.onset <= currentTime + futureWindowSeconds,
  )
}
