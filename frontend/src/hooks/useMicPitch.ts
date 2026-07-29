import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { PitchDetector } from 'pitchy'
import { hzToMidi } from '../game/pitch'

export type MicStatus = 'idle' | 'requesting' | 'active' | 'denied'

export interface PitchSample {
  time: number
  hz: number
  midi: number
  clarity: number
}

const FFT_SIZE = 2048
const CLARITY_THRESHOLD = 0.8

// Speaker playback (not a headset) means the mic can pick up the instrumental bleeding back in
// through the room, on top of whatever `echoCancellation` below manages to cancel -- that
// constraint is a request Chrome tries to honor, not a guarantee, and real speakers at listening
// volume put the echo path well outside what mic-side AEC is tuned for (built for a laptop's own
// speakers a few inches away). A quiet room-bleed signal is much lower level than someone actually
// singing into the mic, so gate on loudness too: same threshold already validated server-side for
// vocal-stem silence detection (`_SILENCE_RMS_GATE` in melody_extraction.py). This won't catch
// bleed that's genuinely as loud as the singer's voice at the mic -- there's no way to tell those
// apart from level alone -- but it does stop quiet backing-track leakage from registering as pitch.
const MIN_RMS = 0.01

function rms(buffer: Float32Array): number {
  let sumSquares = 0
  for (const sample of buffer) sumSquares += sample * sample
  return Math.sqrt(sumSquares / buffer.length)
}

/** Live mic pitch detection via `pitchy`. Exposes the latest detected pitch through a ref (not
 * React state) so a 60fps detection loop doesn't trigger 60fps re-renders -- same pattern as
 * NoteHighway/LyricsDisplay reading `audioRef.current.currentTime` directly. Each sample is
 * timestamped with the *audio's* currentTime (not wall-clock time) so it lines up with note
 * onsets/offsets, which are also in that timeline. */
export function useMicPitch(audioRef: RefObject<HTMLAudioElement | null>) {
  const [status, setStatus] = useState<MicStatus>('idle')
  const latestSampleRef = useRef<PitchSample | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)

  const stop = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    analyserRef.current = null
    latestSampleRef.current = null
    setStatus('idle')
  }

  const start = async () => {
    setStatus('requesting')
    try {
      // Explicit constraints (not just `audio: true`) so the browser's acoustic echo canceller
      // is guaranteed on -- without it, the instrumental playing out of the speakers bleeds back
      // into the mic and gets read as pitch alongside (or instead of) the singer's voice.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      streamRef.current = stream

      const context = audioContextRef.current ?? new AudioContext()
      audioContextRef.current = context
      const analyser = context.createAnalyser()
      analyser.fftSize = FFT_SIZE
      context.createMediaStreamSource(stream).connect(analyser)
      analyserRef.current = analyser

      const detector = PitchDetector.forFloat32Array(analyser.fftSize)
      const buffer = new Float32Array(detector.inputLength)

      setStatus('active')

      const tick = () => {
        const currentAnalyser = analyserRef.current
        if (currentAnalyser) {
          currentAnalyser.getFloatTimeDomainData(buffer)
          const [hz, clarity] = detector.findPitch(buffer, context.sampleRate)
          if (hz > 0 && clarity >= CLARITY_THRESHOLD && rms(buffer) >= MIN_RMS) {
            latestSampleRef.current = {
              time: audioRef.current?.currentTime ?? 0,
              hz,
              midi: hzToMidi(hz),
              clarity,
            }
          }
        }
        rafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch {
      setStatus('denied')
    }
  }

  useEffect(() => stop, [])

  return { status, start, stop, latestSampleRef }
}
