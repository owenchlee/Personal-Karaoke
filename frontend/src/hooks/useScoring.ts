import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { NoteEvent } from '../types/note'
import type { PitchSample } from './useMicPitch'
import {
  createEmptyAccuracy,
  findActiveNoteIndex,
  isPitchMatch,
  recordSample,
  songAccuracyScore,
  type NoteAccuracy,
} from '../game/scoring'

// A sample older than this relative to the current playhead is stale (e.g. mic was silent, or
// detection dropped a frame) and shouldn't be scored against whatever note is active now.
const STALE_SAMPLE_SECONDS = 0.2

interface UseScoringOptions {
  audioRef: RefObject<HTMLAudioElement | null>
  notes: NoteEvent[]
  latestSampleRef: RefObject<PitchSample | null>
  active: boolean
}

export function useScoring({ audioRef, notes, latestSampleRef, active }: UseScoringOptions) {
  const accuraciesRef = useRef<NoteAccuracy[]>(notes.map(() => createEmptyAccuracy()))
  const [runningScore, setRunningScore] = useState(100)

  const reset = useCallback(() => {
    accuraciesRef.current = notes.map(() => createEmptyAccuracy())
    setRunningScore(100)
  }, [notes])

  // A new song (different notes array) starts with a clean scorecard.
  useEffect(() => {
    reset()
  }, [reset])

  useEffect(() => {
    if (!active) return
    let rafId: number

    const tick = () => {
      const currentTime = audioRef.current?.currentTime ?? 0
      const sample = latestSampleRef.current

      // Staleness is judged against `capturedAt`, not the calibration-corrected `time` -- see the
      // note above `bufferLatencySeconds` in `useMicPitch.ts`. The note a fresh sample is judged
      // against, though, does use `sample.time`: a sample only ever arrives after detection-buffer
      // + mic/calibration latency has elapsed, so the note active "right now" has often already
      // scrolled past the moment the singer actually produced it -- see the matching note in
      // `useLivePitchIndicator.ts`.
      if (sample && currentTime - sample.capturedAt <= STALE_SAMPLE_SECONDS) {
        const noteIndex = findActiveNoteIndex(notes, sample.time)
        if (noteIndex !== -1) {
          const isHit = isPitchMatch(sample.midi, notes[noteIndex].pitch_midi)
          accuraciesRef.current[noteIndex] = recordSample(accuraciesRef.current[noteIndex], isHit)
        }
      }

      setRunningScore(songAccuracyScore(notes, accuraciesRef.current, currentTime))

      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(rafId)
  }, [active, audioRef, notes, latestSampleRef])

  const getFinalScore = useCallback(
    () => songAccuracyScore(notes, accuraciesRef.current, Infinity),
    [notes],
  )

  return { runningScore, getFinalScore, reset, accuraciesRef }
}
