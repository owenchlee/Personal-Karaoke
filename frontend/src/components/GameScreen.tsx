import { useCallback, useEffect, useRef, useState } from 'react'
import type { NoteEvent } from '../types/note'
import type { LyricWord } from '../types/lyrics'
import LyricsDisplay from './LyricsDisplay'
import NoteHighway from './NoteHighway'
import SongLibrary from './SongLibrary'
import { midiToNoteName } from '../game/coords'
import { accuracyFraction } from '../game/scoring'
import { computeBestShift, hasVoiceRange, loadVoiceRange } from '../game/voiceRange'
import { useLivePitchIndicator } from '../hooks/useLivePitchIndicator'
import { useMicPitch } from '../hooks/useMicPitch'
import { useRecording } from '../hooks/useRecording'
import { useScoring } from '../hooks/useScoring'
import type { ScoreSubmissionResult } from '../types/score'
import type { Song } from '../types/song'
import { BASE_URL, CACHE_BASE, DEMO_MODE, SONGS_URL } from '../config'

// One octave either way -- must match the job server's own `_MAX_TRANSPOSE_SEMITONES`
// (scripts/server.py), which rejects anything outside this range with a 422.
const MAX_TRANSPOSE_SEMITONES = 12

function getSongId(): string | null {
  return new URLSearchParams(window.location.search).get('song')
}

function selectSong(slug: string) {
  // Hard navigation so the whole screen (and every hook keyed on songId) starts fresh for the
  // newly chosen song, rather than trying to hot-swap state mid-render.
  window.location.href = `${BASE_URL}?song=${encodeURIComponent(slug)}`
}

type LoadState = 'loading' | 'loaded' | 'error'

function fetchJson<T>(url: string): Promise<T> {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`)
    return response.json() as Promise<T>
  })
}

interface RecentNoteStat {
  onset: number
  noteName: string
  hitPercent: number | null
}

// How often the debug panel's "recent notes" list re-derives from the scoring accuracy refs --
// those are refs (not React state, deliberately, so 60fps scoring doesn't force re-renders), so
// something has to poll them on a slow cadence to show them as text. Only runs while the debug
// panel is actually open.
const RECENT_NOTES_POLL_MS = 250
const RECENT_NOTES_COUNT = 3

function GameScreen() {
  const [songId] = useState(getSongId)
  const [songTitle, setSongTitle] = useState<string | null>(null)
  const [notes, setNotes] = useState<NoteEvent[]>([])
  const [lyrics, setLyrics] = useState<LyricWord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [finalScore, setFinalScore] = useState<number | null>(null)
  const [scoreResult, setScoreResult] = useState<ScoreSubmissionResult | null>(null)
  const [debug, setDebug] = useState(false)
  const [recentNotes, setRecentNotes] = useState<RecentNoteStat[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // The "key change" feature (see game/voiceRange.ts): `baseNotesRef` always holds the song's
  // original, untransposed notes (so `computeBestShift` and a "reset to original key" both have
  // something stable to work from), while `notes`/`instrumentalUrl` hold whatever's actually being
  // played right now -- the original, or a key-shifted variant fetched from the job server.
  const baseNotesRef = useRef<NoteEvent[]>([])
  const [semitoneShift, setSemitoneShift] = useState(0)
  const [keyChangeStatus, setKeyChangeStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [instrumentalUrl, setInstrumentalUrl] = useState<string | null>(null)
  const transposeRequestIdRef = useRef(0)
  const [voiceRangeSaved] = useState(hasVoiceRange)

  const {
    status: micStatus,
    start: startMic,
    latestSampleRef,
    streamRef: micStreamRef,
    latencySecondsRef,
  } = useMicPitch(audioRef)
  const { runningScore, getFinalScore, reset: resetScoring, accuraciesRef } = useScoring({
    audioRef,
    notes,
    latestSampleRef,
    active: micStatus === 'active',
  })
  const { stateRef: indicatorRef, display: indicatorDisplay } = useLivePitchIndicator({
    audioRef,
    notes,
    latestSampleRef,
  })
  const recording = useRecording({ audioRef, micStreamRef, songId: songId ?? '' })

  useEffect(() => {
    if (songId === null) return

    let cancelled = false
    setLoadState('loading')

    Promise.all([
      fetchJson<NoteEvent[]>(`${CACHE_BASE}/${songId}/notes.json`),
      fetchJson<LyricWord[]>(`${CACHE_BASE}/${songId}/lyrics.json`),
    ])
      .then(([notesData, lyricsData]) => {
        if (cancelled) return
        baseNotesRef.current = notesData
        setSemitoneShift(0)
        setInstrumentalUrl(null)
        setNotes(notesData)
        setLyrics(lyricsData)
        setLoadState('loaded')
      })
      .catch(() => {
        if (cancelled) return
        setLoadState('error')
      })

    return () => {
      cancelled = true
    }
  }, [songId])

  // Ensures a pitch-shifted variant of this song exists (generating/caching it server-side on
  // first request -- see scripts/server.py's transpose endpoints) and swaps `notes`/
  // `instrumentalUrl` over to it. `shift === 0` just reverts to the original assets locally, no
  // request needed. Guards against out-of-order responses (e.g. the +/- buttons clicked faster
  // than a request can resolve) with a request-id token -- only the most recently issued request's
  // result is ever applied.
  const applyKeyShift = useCallback(
    async (shift: number) => {
      const requestId = ++transposeRequestIdRef.current

      if (shift === 0) {
        setNotes(baseNotesRef.current)
        setInstrumentalUrl(null)
        setSemitoneShift(0)
        setKeyChangeStatus('idle')
        return
      }

      setKeyChangeStatus('loading')
      try {
        const response = await fetch(`/api/songs/${songId}/transpose`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ semitones: shift }),
        })
        if (!response.ok) throw new Error(`Failed to transpose: ${response.status}`)
        const { instrumental_url, notes_url } = (await response.json()) as {
          instrumental_url: string
          notes_url: string
        }
        const shiftedNotes = await fetchJson<NoteEvent[]>(notes_url)

        if (transposeRequestIdRef.current !== requestId) return // superseded by a newer request
        setNotes(shiftedNotes)
        setInstrumentalUrl(instrumental_url)
        setSemitoneShift(shift)
        setKeyChangeStatus('idle')
      } catch {
        if (transposeRequestIdRef.current !== requestId) return
        setKeyChangeStatus('error')
      }
    },
    [songId],
  )

  // Auto-fits the key the moment a song's notes are available, if the player has synced a voice
  // range (game/voiceRange.ts) -- runs once per song load, not on every render, since `loadState`
  // only transitions to 'loaded' once per songId. A no-op (stays in the original key) when no
  // range is saved, on the demo build (no job server to transpose against), or when the song
  // already fits.
  useEffect(() => {
    if (loadState !== 'loaded' || DEMO_MODE) return
    const voiceRange = loadVoiceRange()
    if (!voiceRange) return
    const bestShift = computeBestShift(baseNotesRef.current, voiceRange)
    if (bestShift !== 0) void applyKeyShift(bestShift)
  }, [loadState, applyKeyShift])

  useEffect(() => {
    if (songId === null) return

    let cancelled = false
    fetchJson<{ songs: Song[] }>(SONGS_URL)
      .then(({ songs }) => {
        if (cancelled) return
        const match = songs.find((song) => song.slug === songId)
        if (match) setSongTitle(match.title)
      })
      .catch(() => {
        // Falls back to the raw slug in the heading below -- not worth surfacing an error for.
      })

    return () => {
      cancelled = true
    }
  }, [songId])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || loadState !== 'loaded') return

    const handlePlay = () => {
      // Only treat this as a fresh attempt (reset the scorecard) when playback is starting from
      // the beginning, not on every resume-after-pause mid-song.
      if (audio.currentTime < 0.5) {
        resetScoring()
        setFinalScore(null)
        setScoreResult(null)
      }
    }
    const handleEnded = () => {
      const score = getFinalScore()
      setFinalScore(score)

      // No job server on the static demo build to persist scores against.
      if (DEMO_MODE) return

      fetch('/api/scores', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_id: songId, score }),
      })
        .then((response) => {
          if (!response.ok) throw new Error(`Failed to submit score: ${response.status}`)
          return response.json() as Promise<ScoreSubmissionResult>
        })
        .then(setScoreResult)
        .catch(() => setScoreResult(null))
    }

    audio.addEventListener('play', handlePlay)
    audio.addEventListener('ended', handleEnded)
    return () => {
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [loadState, songId, resetScoring, getFinalScore])

  useEffect(() => {
    if (!debug) return

    const poll = () => {
      const currentTime = audioRef.current?.currentTime ?? 0
      const completed = notes
        .map((note, index) => ({ note, index }))
        .filter(({ note }) => note.offset <= currentTime)
        .slice(-RECENT_NOTES_COUNT)

      setRecentNotes(
        completed.map(({ note, index }) => {
          const accuracy = accuraciesRef.current[index]
          return {
            onset: note.onset,
            noteName: midiToNoteName(note.pitch_midi),
            hitPercent: accuracy && accuracy.totalCount > 0 ? Math.round(accuracyFraction(accuracy) * 100) : null,
          }
        }),
      )
    }

    poll()
    const intervalId = window.setInterval(poll, RECENT_NOTES_POLL_MS)
    return () => window.clearInterval(intervalId)
  }, [debug, notes, accuraciesRef])

  if (songId === null) {
    return (
      <main className="game-screen">
        <section className="panel">
          <h2>Choose a song</h2>
          <SongLibrary onSelect={selectSong} />
        </section>
      </main>
    )
  }

  return (
    <main className={`game-screen${loadState === 'loaded' ? ' game-screen--player' : ''}`}>
      {loadState === 'loading' && <p className="status-message">Loading song&hellip;</p>}
      {loadState === 'error' && (
        <p className="status-message status-message--error">
          Couldn&rsquo;t load cached song assets for <code>{songId}</code>. Open the menu to load
          a song, or run <code>scripts/publish_song.py {songId}</code> first.
        </p>
      )}
      {loadState === 'loaded' && (
        <section className="panel player-panel">
          <div className="player-head">
            <div className="player-head-row">
              <h2>{songTitle ?? songId}</h2>
              <div className="mic-controls">
                {micStatus === 'idle' && (
                  <button type="button" className="btn btn-secondary" onClick={() => void startMic()}>
                    Enable Mic
                  </button>
                )}
                {micStatus === 'requesting' && <span className="muted">Requesting mic&hellip;</span>}
                {micStatus === 'denied' && (
                  <span className="form-error">Mic permission denied or unavailable.</span>
                )}
                {micStatus === 'active' && (
                  <>
                    <span className="score-badge">Score: {runningScore}%</span>
                    {recording.status === 'idle' && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void recording.start()}
                      >
                        Start Recording
                      </button>
                    )}
                    {recording.status === 'recording' && (
                      <button
                        type="button"
                        className="btn btn-secondary recording-active"
                        onClick={recording.stop}
                      >
                        <span className="recording-dot" aria-hidden="true" />
                        Stop Recording
                      </button>
                    )}
                    {recording.status === 'processing' && <span className="muted">Rendering MP3&hellip;</span>}
                    {recording.status === 'done' && recording.downloadUrl && (
                      <>
                        <a
                          className="btn btn-primary"
                          href={recording.downloadUrl}
                          download={recording.downloadFilename ?? `${songId}-recording.mp3`}
                        >
                          Download MP3
                        </a>
                        <button type="button" className="btn btn-secondary" onClick={recording.reset}>
                          Record again
                        </button>
                      </>
                    )}
                    {recording.status === 'error' && recording.errorMessage && (
                      <span className="form-error">{recording.errorMessage}</span>
                    )}
                  </>
                )}
                <label className="debug-toggle">
                  <input
                    type="checkbox"
                    checked={debug}
                    onChange={(event) => setDebug(event.target.checked)}
                  />
                  Debug
                </label>
              </div>
            </div>
            <audio
              ref={audioRef}
              src={instrumentalUrl ?? `${CACHE_BASE}/${songId}/instrumental.wav`}
              controls
            />
            {!DEMO_MODE && voiceRangeSaved && (
              <div className="key-change-row">
                <span className="pitch-readout-label">Key</span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={keyChangeStatus === 'loading' || semitoneShift <= -MAX_TRANSPOSE_SEMITONES}
                  onClick={() => void applyKeyShift(Math.max(semitoneShift - 1, -MAX_TRANSPOSE_SEMITONES))}
                  aria-label="Shift key down one semitone"
                >
                  &minus;
                </button>
                <span className="key-change-value">
                  {semitoneShift === 0 ? 'Original' : semitoneShift > 0 ? `+${semitoneShift}` : semitoneShift}
                </span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={keyChangeStatus === 'loading' || semitoneShift >= MAX_TRANSPOSE_SEMITONES}
                  onClick={() => void applyKeyShift(Math.min(semitoneShift + 1, MAX_TRANSPOSE_SEMITONES))}
                  aria-label="Shift key up one semitone"
                >
                  +
                </button>
                {semitoneShift !== 0 && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={keyChangeStatus === 'loading'}
                    onClick={() => void applyKeyShift(0)}
                  >
                    Reset
                  </button>
                )}
                {keyChangeStatus === 'loading' && <span className="muted">Adjusting to your key&hellip;</span>}
                {keyChangeStatus === 'error' && (
                  <span className="form-error">Key change failed -- still playing the last working key.</span>
                )}
              </div>
            )}
            {micStatus === 'active' && (
              <div className="pitch-readout">
                <span className="pitch-readout-item">
                  <span className="pitch-readout-label">Target</span>
                  <span className="pitch-readout-value">{indicatorDisplay.targetNoteName ?? '-'}</span>
                </span>
                <span className="pitch-readout-item">
                  <span className="pitch-readout-label">You</span>
                  <span className="pitch-readout-value">{indicatorDisplay.detectedNoteName ?? '-'}</span>
                </span>
                {indicatorDisplay.centsOff !== null && (
                  <span
                    className={
                      'pitch-readout-cents' +
                      (Math.abs(indicatorDisplay.centsOff) <= 50 ? ' pitch-readout-cents--good' : '')
                    }
                  >
                    {indicatorDisplay.centsOff > 0 ? '+' : ''}
                    {indicatorDisplay.centsOff}c
                  </span>
                )}
              </div>
            )}
            {finalScore !== null && (
              <div className="final-score-banner">
                <strong>Final score: {finalScore}%</strong>
                {scoreResult?.is_new_best && (
                  <div className="final-score-highlight">
                    🏆 New high score!
                    {scoreResult.previous_best !== null ? ` (was ${scoreResult.previous_best}%)` : ''}
                  </div>
                )}
                {scoreResult && !scoreResult.is_new_best && (
                  <div className="final-score-subtext">Best: {scoreResult.best_score}%</div>
                )}
              </div>
            )}
          </div>
          <LyricsDisplay audioRef={audioRef} words={lyrics} />
          <div className="stage">
            <NoteHighway
              audioRef={audioRef}
              notes={notes}
              indicatorRef={indicatorRef}
              accuraciesRef={accuraciesRef}
              latencySecondsRef={latencySecondsRef}
            />
          </div>
          {debug && (
            <div className="debug-panel">
              <div className="debug-panel-row">
                <span>raw</span>
                <span>
                  {indicatorDisplay.rawHz !== null ? `${indicatorDisplay.rawHz}Hz` : '-'}
                  {indicatorDisplay.rawClarity !== null ? ` · clarity ${indicatorDisplay.rawClarity}` : ''}
                  {indicatorDisplay.sampleAge !== null ? ` · age ${indicatorDisplay.sampleAge}s` : ''}
                </span>
              </div>
              {recentNotes.map((note) => (
                <div className="debug-panel-row" key={note.onset}>
                  <span>
                    {note.onset.toFixed(1)}s {note.noteName}
                  </span>
                  <span>{note.hitPercent !== null ? `${note.hitPercent}% hit` : 'not sung'}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  )
}

export default GameScreen
