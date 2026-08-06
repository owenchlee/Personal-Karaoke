// Mic/speaker latency calibration -- the same "play a click, have the user tap/clap on beat"
// approach UltraStar and similar games use. `useCalibration` schedules a click track and records
// detected tap/clap times, both already expressed in one AudioContext's own `currentTime` clock;
// the pure functions here turn those two timestamp lists into a single correction (seconds) that
// `useMicPitch` folds into every pitch sample's timestamp.

// A tap/clap more than this far from a given beat isn't a response to that beat -- either a missed
// beat (no tap counted) or noise unrelated to any beat (dropped rather than force-matched).
const MAX_MATCH_WINDOW_SECONDS = 0.4

// Fewer matched beats than this isn't enough data to trust -- the run is reported as inconclusive
// (null) rather than saving a wild guess.
const MIN_MATCHED_BEATS = 4

/** Greedily pairs each scheduled beat with the nearest not-yet-used tap within
 * `maxMatchWindowSeconds`, in beat order. Returns the per-beat (tap - beat) offsets for whichever
 * beats found a match -- a positive offset means the tap arrived late relative to the beat, which
 * is the expected sign of real speaker-output + mic-input + processing latency. */
export function pairBeatsToTaps(
  beatTimes: number[],
  tapTimes: number[],
  maxMatchWindowSeconds = MAX_MATCH_WINDOW_SECONDS,
): number[] {
  const usedTapIndices = new Set<number>()
  const offsets: number[] = []

  for (const beat of beatTimes) {
    let bestIndex = -1
    let bestDistance = Infinity
    for (let i = 0; i < tapTimes.length; i++) {
      if (usedTapIndices.has(i)) continue
      const distance = Math.abs(tapTimes[i] - beat)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
    }
    if (bestIndex !== -1 && bestDistance <= maxMatchWindowSeconds) {
      usedTapIndices.add(bestIndex)
      offsets.push(tapTimes[bestIndex] - beat)
    }
  }

  return offsets
}

export interface CalibrationResult {
  offsetSeconds: number
  matchedBeats: number
  totalBeats: number
}

/** Median of the matched per-beat offsets -- resistant to a single mistimed tap the same way
 * `smoothMidi` resists a single-frame pitch outlier (see `game/pitch.ts`). Returns null when fewer
 * than `MIN_MATCHED_BEATS` beats matched, too little data for a trustworthy calibration. */
export function computeCalibrationResult(
  beatTimes: number[],
  tapTimes: number[],
): CalibrationResult | null {
  const offsets = pairBeatsToTaps(beatTimes, tapTimes)
  if (offsets.length < MIN_MATCHED_BEATS) return null

  const sorted = [...offsets].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const offsetSeconds = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]

  return { offsetSeconds, matchedBeats: offsets.length, totalBeats: beatTimes.length }
}

const STORAGE_KEY = 'karaoke:calibrationOffsetSeconds'

/** The stored one-time calibration offset (seconds), or 0 (no correction) if none has been run yet
 * or storage isn't available (e.g. under test, or a browser with storage disabled). */
export function loadCalibrationOffsetSeconds(): number {
  if (typeof window === 'undefined' || !window.localStorage) return 0
  const raw = window.localStorage.getItem(STORAGE_KEY)
  if (raw === null) return 0
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : 0
}

/** Whether a real calibration result is on file, as opposed to `loadCalibrationOffsetSeconds`'s `0`
 * fallback for "never calibrated" -- callers that need to tell those two apart (e.g.
 * `useRecording`, which should let the server fall back to its own default rather than send an
 * unearned `0`) should check this first. */
export function hasCalibrationOffset(): boolean {
  if (typeof window === 'undefined' || !window.localStorage) return false
  return window.localStorage.getItem(STORAGE_KEY) !== null
}

export function saveCalibrationOffsetSeconds(offsetSeconds: number): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.setItem(STORAGE_KEY, String(offsetSeconds))
}

export function clearCalibrationOffsetSeconds(): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.removeItem(STORAGE_KEY)
}
