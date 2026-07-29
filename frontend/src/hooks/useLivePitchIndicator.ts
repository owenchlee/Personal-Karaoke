import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { NoteEvent } from '../types/note'
import type { PitchSample } from './useMicPitch'
import { getPitchRange, midiToNoteName } from '../game/coords'
import {
  appendTracePoint,
  foldIntoRange,
  foldToNearestOctave,
  isPlausiblePitchJump,
  signedCentsOff,
  smoothMidi,
  type TracePoint,
} from '../game/pitch'
import { findActiveNoteIndex } from '../game/scoring'

// Matches the values previously inlined in NoteHighway.tsx -- see the git history / NOTES.md for
// why each one is what it is. Centralized here now so every consumer of the live pitch (the
// canvas pill, the always-visible "Target vs You" readout, the debug panel) reads the exact same
// computed value instead of each recomputing its own slightly-different copy.
// Lowered from 0.25 -- NOTES.md flagged this stacked with NoteHighway's own render-easing tau as
// the likely source of the pill visibly lagging real singing (median smoothing lags a pitch
// change by up to about half this window). Shrinking it trims that lag while still resisting
// single-frame outliers over a ~9-sample window at the ~60fps detection rate.
const SMOOTHING_WINDOW_SECONDS = 0.15
const MAX_SAMPLE_AGE_SECONDS = 0.15
const HOLD_SECONDS = 0.6
// How often the human-readable display state is allowed to re-render -- the underlying computation
// runs every animation frame (for the canvas pill), but text doesn't need 60fps churn.
const DISPLAY_UPDATE_INTERVAL_MS = 100

export interface LiveIndicatorState {
  activeNote: NoteEvent | null
  sample: PitchSample | null
  sampleAge: number
  /** Folded + smoothed MIDI value driving the pill, or null if there's no recent voicing. */
  smoothedMidi: number | null
}

export interface LiveIndicatorDisplay {
  targetNoteName: string | null
  detectedNoteName: string | null
  /** Signed cents off the active note (positive = sharp, negative = flat), or null when there's
   * no active note or no current detection to compare it against. */
  centsOff: number | null
  /** Raw (pre-fold, pre-smoothing) detected pitch, for the debug panel -- lets a player check
   * what the mic actually picked up, independent of any of the fold/smoothing logic above. */
  rawHz: number | null
  rawClarity: number | null
  sampleAge: number | null
}

interface UseLivePitchIndicatorOptions {
  audioRef: RefObject<HTMLAudioElement | null>
  notes: NoteEvent[]
  latestSampleRef?: RefObject<PitchSample | null>
}

const EMPTY_STATE: LiveIndicatorState = {
  activeNote: null,
  sample: null,
  sampleAge: Infinity,
  smoothedMidi: null,
}

const EMPTY_DISPLAY: LiveIndicatorDisplay = {
  targetNoteName: null,
  detectedNoteName: null,
  centsOff: null,
  rawHz: null,
  rawClarity: null,
  sampleAge: null,
}

/** Owns the fold/smoothing state machine for the live mic pitch indicator (previously private to
 * `NoteHighway`), so every UI element that shows "what note is being sung right now" -- the
 * canvas pill, a plain-language readout, a debug panel -- reads from one shared computation
 * instead of drifting apart. Exposes both a per-frame ref (`stateRef`, for pixel-perfect canvas
 * rendering with no re-render cost) and throttled React state (`display`, for text). */
export function useLivePitchIndicator({ audioRef, notes, latestSampleRef }: UseLivePitchIndicatorOptions) {
  const stateRef = useRef<LiveIndicatorState>(EMPTY_STATE)
  const [display, setDisplay] = useState<LiveIndicatorDisplay>(EMPTY_DISPLAY)

  const traceRef = useRef<TracePoint[]>([])
  const octaveAnchorRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastDisplayUpdateRef = useRef<number>(0)

  useEffect(() => {
    const pitchRange = getPitchRange(notes)

    const tick = (timestamp: number) => {
      const currentTime = audioRef.current?.currentTime ?? 0
      const activeNoteIndex = findActiveNoteIndex(notes, currentTime)
      const activeNote = activeNoteIndex !== -1 ? notes[activeNoteIndex] : null

      const sample = latestSampleRef?.current ?? null
      const sampleAge = sample ? currentTime - sample.time : Infinity

      if (sample && sampleAge <= MAX_SAMPLE_AGE_SECONDS) {
        // See NoteHighway's git history for the reasoning: fold toward the note actually being
        // judged right now (ties the pill to the same ground truth `useScoring` uses), falling
        // back to self-anchoring only when nothing is active (a silence gap/instrumental break).
        const noAnchorEstablishedYet = !activeNote && octaveAnchorRef.current === null
        const anchor = activeNote ? activeNote.pitch_midi : (octaveAnchorRef.current ?? sample.midi)
        const foldedMidi = foldIntoRange(foldToNearestOctave(sample.midi, anchor), pitchRange)

        if (noAnchorEstablishedYet || isPlausiblePitchJump(anchor, foldedMidi)) {
          octaveAnchorRef.current = foldedMidi
          traceRef.current = appendTracePoint(
            traceRef.current,
            { time: currentTime, midi: foldedMidi },
            SMOOTHING_WINDOW_SECONDS,
          )
        }
      }

      const hasRecentVoicing = sampleAge <= HOLD_SECONDS
      let smoothedMidi: number | null = null
      if (hasRecentVoicing && traceRef.current.length > 0) {
        smoothedMidi = smoothMidi(traceRef.current.map((point) => point.midi))
      } else if (!hasRecentVoicing) {
        octaveAnchorRef.current = null
        traceRef.current = []
      }

      stateRef.current = { activeNote, sample, sampleAge, smoothedMidi }

      if (timestamp - lastDisplayUpdateRef.current >= DISPLAY_UPDATE_INTERVAL_MS) {
        lastDisplayUpdateRef.current = timestamp
        const targetNoteName = activeNote ? midiToNoteName(activeNote.pitch_midi) : null
        const detectedNoteName = smoothedMidi !== null ? midiToNoteName(smoothedMidi) : null
        const centsOff =
          activeNote && smoothedMidi !== null ? signedCentsOff(smoothedMidi, activeNote.pitch_midi) : null
        const rawHz = sample ? Math.round(sample.hz * 10) / 10 : null
        const rawClarity = sample ? Math.round(sample.clarity * 100) / 100 : null
        const roundedAge = sample ? Math.round(sampleAge * 100) / 100 : null
        setDisplay((previous) =>
          previous.targetNoteName === targetNoteName &&
          previous.detectedNoteName === detectedNoteName &&
          previous.centsOff === centsOff &&
          previous.rawHz === rawHz &&
          previous.rawClarity === rawClarity &&
          previous.sampleAge === roundedAge
            ? previous
            : { targetNoteName, detectedNoteName, centsOff, rawHz, rawClarity, sampleAge: roundedAge },
        )
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [audioRef, notes, latestSampleRef])

  return { stateRef, display }
}
