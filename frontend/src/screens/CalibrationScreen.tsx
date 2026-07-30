import { useState } from 'react'
import { useCalibration } from '../hooks/useCalibration'
import {
  clearCalibrationOffsetSeconds,
  loadCalibrationOffsetSeconds,
  saveCalibrationOffsetSeconds,
} from '../game/calibration'

function formatMs(offsetSeconds: number): string {
  return `${Math.round(offsetSeconds * 1000)}ms`
}

function CalibrationScreen() {
  const { status, beatsScheduled, totalBeats, tapsDetected, result, start, cancel } = useCalibration()
  const [savedOffsetSeconds, setSavedOffsetSeconds] = useState(loadCalibrationOffsetSeconds)
  const [justSaved, setJustSaved] = useState(false)

  const handleSave = () => {
    if (!result) return
    saveCalibrationOffsetSeconds(result.offsetSeconds)
    setSavedOffsetSeconds(result.offsetSeconds)
    setJustSaved(true)
  }

  const handleReset = () => {
    clearCalibrationOffsetSeconds()
    setSavedOffsetSeconds(0)
    setJustSaved(false)
  }

  return (
    <main className="game-screen">
      <p className="status-message">
        Measures the combined delay between when a beat is scheduled and when your clap actually
        reaches the mic -- speaker output latency, your reaction time, and mic input latency all
        stack up, and this one-time correction gets applied to every pitch reading during a song so
        the mic timing lines up with the reference notes.
      </p>

      <section className="panel">
        <h2>Mic delay calibration</h2>
        <p className="muted">
          Currently saved correction: <strong>{formatMs(savedOffsetSeconds)}</strong>
          {savedOffsetSeconds === 0 && ' (none yet -- run the calibration below)'}
        </p>

        {status === 'idle' && (
          <div className="button-row calibration-actions">
            <button type="button" className="btn btn-primary" onClick={() => void start()}>
              Start calibration
            </button>
            {savedOffsetSeconds !== 0 && (
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Clear saved correction
              </button>
            )}
          </div>
        )}

        {status === 'requesting-mic' && <p className="muted">Requesting mic&hellip;</p>}

        {status === 'error' && (
          <p className="form-error" role="alert">
            Mic permission denied or unavailable.
          </p>
        )}

        {status === 'running' && (
          <div className="calibration-running">
            <p>
              Clap along with the clicks, in time with the beat -- {beatsScheduled}/{totalBeats}{' '}
              played, {tapsDetected} claps detected.
            </p>
            <div className="progress-bar">
              <div
                className="progress-bar-fill"
                style={{ width: `${(beatsScheduled / totalBeats) * 100}%` }}
              />
            </div>
            <button type="button" className="btn btn-secondary calibration-actions" onClick={cancel}>
              Cancel
            </button>
          </div>
        )}

        {status === 'done' && (
          <div className="calibration-result">
            {result ? (
              <>
                <p>
                  Measured offset: <strong>{formatMs(result.offsetSeconds)}</strong> (from{' '}
                  {result.matchedBeats}/{result.totalBeats} matched claps).
                </p>
                <div className="button-row">
                  <button type="button" className="btn btn-primary" onClick={handleSave}>
                    {justSaved ? 'Saved' : 'Save this correction'}
                  </button>
                  <button type="button" className="btn btn-secondary" onClick={() => void start()}>
                    Retry
                  </button>
                </div>
              </>
            ) : (
              <>
                <p className="form-error" role="alert">
                  Not enough claps were detected in time with the beat to trust a result -- try
                  clapping louder and closer to the mic, or check the browser has mic permission.
                </p>
                <button type="button" className="btn btn-secondary" onClick={() => void start()}>
                  Retry
                </button>
              </>
            )}
          </div>
        )}
      </section>
    </main>
  )
}

export default CalibrationScreen
