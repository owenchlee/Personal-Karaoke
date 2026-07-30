import { useCallback, useEffect, useRef, useState } from 'react'
import { computeCalibrationResult, type CalibrationResult } from '../game/calibration'

export type CalibrationStatus = 'idle' | 'requesting-mic' | 'running' | 'done' | 'error'

const BPM = 100
const BEAT_INTERVAL_SECONDS = 60 / BPM
const BEAT_COUNT = 8
const LEAD_IN_SECONDS = 1.5
const CLICK_DURATION_SECONDS = 0.06
const CLICK_FREQUENCY_HZ = 1000

// Standard Web Audio "lookahead scheduler" pattern (see Chris Wilson's "A Tale of Two Clocks"): a
// setInterval running far more often than playback needs, which only ever schedules audio nodes a
// short window ahead of `context.currentTime` rather than playing anything directly from a
// UI/timer callback (which is subject to main-thread jitter). This is the one place in the app
// that actually schedules audio *playback* events -- everywhere else (the note highway, lyrics)
// only ever reads a clock, never schedules against one, so this pattern isn't needed there.
const SCHEDULER_INTERVAL_MS = 25
const SCHEDULE_AHEAD_SECONDS = 0.1

// A clap/tap is a loud, sudden transient -- comfortably above normal room noise and above the
// level the click track itself could leak back in at even with echo cancellation on.
const CLAP_RMS_THRESHOLD = 0.15
const CLAP_REFRACTORY_SECONDS = 0.25
const CLAP_FFT_SIZE = 1024

function rms(buffer: Float32Array): number {
  let sumSquares = 0
  for (const sample of buffer) sumSquares += sample * sample
  return Math.sqrt(sumSquares / buffer.length)
}

export interface UseCalibrationResult {
  status: CalibrationStatus
  beatsScheduled: number
  totalBeats: number
  tapsDetected: number
  result: CalibrationResult | null
  start: () => Promise<void>
  cancel: () => void
}

/** Runs a one-time "play a click, clap along" calibration -- the same approach UltraStar and
 * similar games use -- to measure the combined speaker-output + human-reaction + mic-input
 * latency that otherwise silently offsets every mic pitch sample relative to the song's own
 * timeline (see `useMicPitch`). Both the scheduled click times and the detected clap times are
 * read from the *same* AudioContext's `currentTime`, so the measured offset is a clean latency in
 * seconds with no cross-clock drift -- this hook is the one place in the app where that clock, not
 * the `<audio>` element's, is the right single source of truth, since no `<audio>` element is
 * involved in the calibration flow at all. */
export function useCalibration(): UseCalibrationResult {
  const [status, setStatus] = useState<CalibrationStatus>('idle')
  const [beatsScheduled, setBeatsScheduled] = useState(0)
  const [tapsDetected, setTapsDetected] = useState(0)
  const [result, setResult] = useState<CalibrationResult | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const schedulerIntervalRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)
  const finishTimeoutRef = useRef<number | null>(null)

  const cleanup = useCallback(() => {
    if (schedulerIntervalRef.current !== null) {
      window.clearInterval(schedulerIntervalRef.current)
      schedulerIntervalRef.current = null
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (finishTimeoutRef.current !== null) {
      window.clearTimeout(finishTimeoutRef.current)
      finishTimeoutRef.current = null
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }, [])

  const cancel = useCallback(() => {
    cleanup()
    setStatus('idle')
  }, [cleanup])

  const start = useCallback(async () => {
    cleanup()
    setStatus('requesting-mic')
    setBeatsScheduled(0)
    setTapsDetected(0)
    setResult(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      streamRef.current = stream

      const context = audioContextRef.current ?? new AudioContext()
      audioContextRef.current = context

      const analyser = context.createAnalyser()
      analyser.fftSize = CLAP_FFT_SIZE
      context.createMediaStreamSource(stream).connect(analyser)
      const buffer = new Float32Array(analyser.fftSize)

      const startTime = context.currentTime + LEAD_IN_SECONDS
      const beatTimes = Array.from(
        { length: BEAT_COUNT },
        (_, i) => startTime + i * BEAT_INTERVAL_SECONDS,
      )
      const tapTimes: number[] = []
      let nextBeatIndex = 0
      let lastClapTime = -Infinity

      setStatus('running')

      schedulerIntervalRef.current = window.setInterval(() => {
        while (
          nextBeatIndex < beatTimes.length &&
          beatTimes[nextBeatIndex] < context.currentTime + SCHEDULE_AHEAD_SECONDS
        ) {
          const beatTime = beatTimes[nextBeatIndex]
          const oscillator = context.createOscillator()
          const gain = context.createGain()
          oscillator.frequency.value = CLICK_FREQUENCY_HZ
          gain.gain.setValueAtTime(0.3, beatTime)
          gain.gain.exponentialRampToValueAtTime(0.0001, beatTime + CLICK_DURATION_SECONDS)
          oscillator.connect(gain).connect(context.destination)
          oscillator.start(beatTime)
          oscillator.stop(beatTime + CLICK_DURATION_SECONDS)
          nextBeatIndex += 1
          setBeatsScheduled(nextBeatIndex)
        }
      }, SCHEDULER_INTERVAL_MS)

      // Clap/tap onset detection: a transient RMS spike crossing the threshold, gated by a short
      // refractory period so one clap isn't counted several times across consecutive frames.
      const detect = () => {
        analyser.getFloatTimeDomainData(buffer)
        const level = rms(buffer)
        const now = context.currentTime
        if (level >= CLAP_RMS_THRESHOLD && now - lastClapTime >= CLAP_REFRACTORY_SECONDS) {
          lastClapTime = now
          tapTimes.push(now)
          setTapsDetected(tapTimes.length)
        }
        rafRef.current = requestAnimationFrame(detect)
      }
      rafRef.current = requestAnimationFrame(detect)

      const totalDurationSeconds =
        LEAD_IN_SECONDS + BEAT_COUNT * BEAT_INTERVAL_SECONDS + CLAP_REFRACTORY_SECONDS * 2
      finishTimeoutRef.current = window.setTimeout(() => {
        cleanup()
        setResult(computeCalibrationResult(beatTimes, tapTimes))
        setStatus('done')
      }, totalDurationSeconds * 1000)
    } catch {
      cleanup()
      setStatus('error')
    }
  }, [cleanup])

  useEffect(() => cleanup, [cleanup])

  return { status, beatsScheduled, totalBeats: BEAT_COUNT, tapsDetected, result, start, cancel }
}
