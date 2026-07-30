import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { PitchDetector } from 'pitchy'
import { hzToMidi } from '../game/pitch'
import { loadCalibrationOffsetSeconds } from '../game/calibration'

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
// speakers a few inches away). This gate exists only to reject that kind of quiet backing-track
// leakage (or plain silence/noise floor) -- it is deliberately *not* the primary "is this really a
// sung pitch" signal, `clarity` above is: pitchy's clarity is a normalized measure of how strongly
// periodic the signal is, and holds up just as well for a genuine clean tone at low volume as at
// high volume, unlike raw RMS. This constant originally borrowed `_SILENCE_RMS_GATE` from
// `melody_extraction.py`, but that value was only ever validated against a Demucs-separated vocal
// stem, not live mic input -- real feedback reported a genuinely-sung long note trailing off
// quietly (a decrescendo) getting cut from detection well before the singer actually stopped.
// Lowered substantially so a quiet-but-real tail isn't killed by loudness alone.
const MIN_RMS = 0.003

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
  // Read once per mic session (not per-sample) -- a fresh calibration only takes effect the next
  // time the mic is (re)started, same as any other one-time setup value.
  const calibrationOffsetSecondsRef = useRef(0)

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

      // `getFloatTimeDomainData` returns the most recent `fftSize` samples as of *now* -- the
      // pitch it implies was actually sung starting one whole buffer-length in the past, not at
      // the moment we finish reading/processing it. Left uncompensated this silently shifts every
      // detected sample ~40-50ms later than its true position in the song, biasing hit detection
      // right at note boundaries. Combined with the one-time mic/speaker round-trip latency
      // measured by the calibration screen (`useCalibration`/`game/calibration.ts`), this is the
      // full correction applied before a sample is timestamped.
      const bufferLatencySeconds = analyser.fftSize / context.sampleRate
      calibrationOffsetSecondsRef.current = loadCalibrationOffsetSeconds()

      setStatus('active')

      const tick = () => {
        const currentAnalyser = analyserRef.current
        if (currentAnalyser) {
          currentAnalyser.getFloatTimeDomainData(buffer)
          const [hz, clarity] = detector.findPitch(buffer, context.sampleRate)
          if (hz > 0 && clarity >= CLARITY_THRESHOLD && rms(buffer) >= MIN_RMS) {
            const songTimeNow = audioRef.current?.currentTime ?? 0
            latestSampleRef.current = {
              time: songTimeNow - bufferLatencySeconds - calibrationOffsetSecondsRef.current,
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

  return { status, start, stop, latestSampleRef, streamRef }
}
