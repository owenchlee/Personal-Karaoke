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
      const noteIndex = findActiveNoteIndex(notes, currentTime)

      if (noteIndex !== -1 && sample && currentTime - sample.time <= STALE_SAMPLE_SECONDS) {
        const isHit = isPitchMatch(sample.midi, notes[noteIndex].pitch_midi)
        accuraciesRef.current[noteIndex] = recordSample(accuraciesRef.current[noteIndex], isHit)
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
