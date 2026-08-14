import type { NoteEvent } from '../types/note'

export interface VoiceRange {
  lowMidi: number
  highMidi: number
}

const STORAGE_KEY = 'karaoke:voiceRange'

/** The player's saved vocal range (from `VoiceRangeScreen`'s sing-your-lowest/-highest capture),
 * or `null` if they've never synced one -- callers should treat that as "leave the song in its
 * original key", same as `game/calibration.ts`'s "never calibrated" convention. */
export function loadVoiceRange(): VoiceRange | null {
  if (typeof window === 'undefined' || !window.localStorage) return null
  const raw = window.localStorage.getItem(STORAGE_KEY)
  if (raw === null) return null
  try {
    const parsed = JSON.parse(raw) as Partial<VoiceRange>
    if (typeof parsed.lowMidi !== 'number' || typeof parsed.highMidi !== 'number') return null
    if (!Number.isFinite(parsed.lowMidi) || !Number.isFinite(parsed.highMidi)) return null
    return { lowMidi: parsed.lowMidi, highMidi: parsed.highMidi }
  } catch {
    return null
  }
}

export function hasVoiceRange(): boolean {
  return loadVoiceRange() !== null
}

export function saveVoiceRange(range: VoiceRange): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(range))
}

export function clearVoiceRange(): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.removeItem(STORAGE_KEY)
}

const DEFAULT_MAX_SHIFT_SEMITONES = 12

/** Searches every integer semitone shift in `[-maxShift, maxShift]` and picks the one that leaves
 * the least *duration* of singing outside `voiceRange` once every note's `pitch_midi` is shifted
 * by that amount -- duration-weighted (not a bare min/max of the song's range) so one rare outlier
 * note doesn't dominate the fit the way comparing outer envelopes would. Ties are broken toward the
 * smallest `|shift|`, preferring the least change to the original key. Returns `0` (no change) for
 * a song with no notes, since there's nothing to fit.
 *
 * This is a *singability* fit, not a scoring-difficulty one: `game/scoring.ts`'s
 * `pitchClassDistanceCents` already folds pitch matching to be octave-agnostic, so a shift that
 * gets a song's real, physical pitch close to the player's comfortable register is "good enough" --
 * it doesn't need to land every note exactly inside the range for scoring to still work well. */
export function computeBestShift(
  notes: NoteEvent[],
  voiceRange: VoiceRange,
  maxShift = DEFAULT_MAX_SHIFT_SEMITONES,
): number {
  if (notes.length === 0) return 0

  let bestShift = 0
  let bestOutOfRangeDuration = Infinity

  for (let shift = -maxShift; shift <= maxShift; shift++) {
    let outOfRangeDuration = 0
    for (const note of notes) {
      const shiftedPitch = note.pitch_midi + shift
      if (shiftedPitch < voiceRange.lowMidi || shiftedPitch > voiceRange.highMidi) {
        outOfRangeDuration += note.duration
      }
    }

    const isBetter =
      outOfRangeDuration < bestOutOfRangeDuration ||
      (outOfRangeDuration === bestOutOfRangeDuration && Math.abs(shift) < Math.abs(bestShift))
    if (isBetter) {
      bestOutOfRangeDuration = outOfRangeDuration
      bestShift = shift
    }
  }

  return bestShift
}
