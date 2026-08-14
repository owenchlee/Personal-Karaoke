import { useEffect, useRef, useState } from 'react'
import type { PitchSample } from '../hooks/useMicPitch'
import { useMicPitch } from '../hooks/useMicPitch'
import { midiToNoteName } from '../game/coords'
import { clearVoiceRange, loadVoiceRange, saveVoiceRange, type VoiceRange } from '../game/voiceRange'

type CaptureTarget = 'low' | 'high'

// Polls the live pitch ref (see the module-level note on `useMicPitch` re: why this needs a ref,
// not React state) at a slow, human-scale cadence -- fast enough to catch a few seconds of a held
// note, far too slow to matter for anything performance-sensitive here.
const CAPTURE_POLL_MS = 100

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

/** Lets a player sing their lowest and highest comfortable note once, saved locally
 * (`game/voiceRange.ts`) and later used by `GameScreen` to auto-transpose songs whose sung range
 * doesn't fit -- same one-time "measure something about this player/setup and save it" shape as
 * `CalibrationScreen`'s mic-latency measurement.
 *
 * Reuses `useMicPitch` directly rather than re-implementing pitch detection: that hook only needs
 * an `<audio>` element ref to timestamp samples against the song timeline, which this screen has no
 * use for (there's no song playing here) -- passing a ref that's never attached to a real element
 * just means every sample's `time`/`capturedAt` fields read as 0, which this screen never reads;
 * only `.midi`/`.clarity` (already gated by the hook's own clarity/RMS thresholds) matter here. */
function VoiceRangeScreen() {
  const unattachedAudioRef = useRef<HTMLAudioElement | null>(null)
  const { status: micStatus, start: startMic, latestSampleRef } = useMicPitch(unattachedAudioRef)

  const [capturing, setCapturing] = useState<CaptureTarget | null>(null)
  const [captureError, setCaptureError] = useState<string | null>(null)
  const capturedSamplesRef = useRef<number[]>([])

  const [savedRange, setSavedRange] = useState<VoiceRange | null>(loadVoiceRange)
  const [lowMidi, setLowMidi] = useState<number | null>(savedRange?.lowMidi ?? null)
  const [highMidi, setHighMidi] = useState<number | null>(savedRange?.highMidi ?? null)
  const [justSaved, setJustSaved] = useState(false)

  useEffect(() => {
    if (!capturing) return

    capturedSamplesRef.current = []
    // Dedupes against stale reads: `latestSampleRef` only gets a *new* object when the hook's
    // detection loop finds a genuinely fresh valid pitch that frame (see useMicPitch.ts) -- it
    // isn't cleared when the singer stops/pauses, so polling it on a plain timer without this
    // identity check would keep re-recording whatever pitch was last heard (e.g. still holding
    // the lowest note's value for a moment after starting to capture the highest note).
    let lastSeenSample: PitchSample | null = null
    const intervalId = window.setInterval(() => {
      const sample = latestSampleRef.current
      if (sample && sample !== lastSeenSample) {
        lastSeenSample = sample
        capturedSamplesRef.current.push(sample.midi)
      }
    }, CAPTURE_POLL_MS)

    return () => window.clearInterval(intervalId)
  }, [capturing, latestSampleRef])

  const beginCapture = async (target: CaptureTarget) => {
    setCaptureError(null)
    setJustSaved(false)
    if (micStatus !== 'active') await startMic()
    setCapturing(target)
  }

  const finishCapture = () => {
    const samples = capturedSamplesRef.current
    if (samples.length === 0) {
      setCaptureError('No clear pitch detected -- try again, singing a bit louder and closer to the mic.')
      setCapturing(null)
      return
    }
    const midi = Math.round(median(samples))
    if (capturing === 'low') setLowMidi(midi)
    else if (capturing === 'high') setHighMidi(midi)
    setCapturing(null)
  }

  const rangeIsValid = lowMidi !== null && highMidi !== null && highMidi > lowMidi
  const canSave = rangeIsValid && !capturing

  const handleSave = () => {
    if (lowMidi === null || highMidi === null || !rangeIsValid) return
    const range = { lowMidi, highMidi }
    saveVoiceRange(range)
    setSavedRange(range)
    setJustSaved(true)
  }

  const handleClear = () => {
    clearVoiceRange()
    setSavedRange(null)
    setLowMidi(null)
    setHighMidi(null)
    setJustSaved(false)
  }

  return (
    <main className="game-screen">
      <p className="status-message">
        Sing your lowest and highest comfortable note once, and songs whose range doesn&rsquo;t fit
        your voice will automatically be transposed to a key that does -- the instrumental and the
        note targets shift together, at the same tempo.
      </p>

      <section className="panel">
        <h2>Voice range</h2>

        {savedRange ? (
          <p className="muted">
            Currently saved range: <strong>{midiToNoteName(savedRange.lowMidi)}</strong> to{' '}
            <strong>{midiToNoteName(savedRange.highMidi)}</strong>
          </p>
        ) : (
          <p className="muted">No range saved yet -- capture both notes below.</p>
        )}

        {micStatus === 'denied' && (
          <p className="form-error" role="alert">
            Mic permission denied or unavailable.
          </p>
        )}
        {micStatus === 'requesting' && <p className="muted">Requesting mic&hellip;</p>}

        <div className="voice-range-capture-row">
          <div className="voice-range-capture">
            <h3>Lowest note</h3>
            <p className="voice-range-value">{lowMidi !== null ? midiToNoteName(lowMidi) : '–'}</p>
            {capturing === 'low' ? (
              <button
                type="button"
                className="btn btn-secondary recording-active"
                onClick={finishCapture}
              >
                <span className="recording-dot" aria-hidden="true" />
                Stop &mdash; I&rsquo;m done
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-secondary"
                disabled={capturing !== null}
                onClick={() => void beginCapture('low')}
              >
                Sing my lowest note
              </button>
            )}
          </div>

          <div className="voice-range-capture">
            <h3>Highest note</h3>
            <p className="voice-range-value">{highMidi !== null ? midiToNoteName(highMidi) : '–'}</p>
            {capturing === 'high' ? (
              <button
                type="button"
                className="btn btn-secondary recording-active"
                onClick={finishCapture}
              >
                <span className="recording-dot" aria-hidden="true" />
                Stop &mdash; I&rsquo;m done
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-secondary"
                disabled={capturing !== null}
                onClick={() => void beginCapture('high')}
              >
                Sing my highest note
              </button>
            )}
          </div>
        </div>

        {captureError && (
          <p className="form-error" role="alert">
            {captureError}
          </p>
        )}
        {lowMidi !== null && highMidi !== null && !rangeIsValid && (
          <p className="form-error" role="alert">
            Your highest note should be higher than your lowest -- try recapturing one of them.
          </p>
        )}

        <div className="button-row calibration-actions">
          <button type="button" className="btn btn-primary" disabled={!canSave} onClick={handleSave}>
            {justSaved ? 'Saved' : 'Save my range'}
          </button>
          {savedRange && (
            <button type="button" className="btn btn-secondary" onClick={handleClear}>
              Clear saved range
            </button>
          )}
        </div>
      </section>
    </main>
  )
}

export default VoiceRangeScreen
