import type { NoteEvent } from '../types/note'

export interface PitchRange {
  min: number
  max: number
}

const PITCH_PAD_SEMITONES = 2
const DEFAULT_PITCH_RANGE: PitchRange = { min: 48, max: 72 }

export const DEFAULT_PX_PER_SECOND = 200
export const DEFAULT_PAST_BUFFER_SECONDS = 0.5

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
