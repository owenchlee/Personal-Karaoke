import { useEffect, useRef, useState } from 'react'
import type { NoteEvent } from '../types/note'
import type { LyricWord } from '../types/lyrics'
import LyricsDisplay from './LyricsDisplay'
import NoteHighway from './NoteHighway'
import { midiToNoteName } from '../game/coords'
import { accuracyFraction } from '../game/scoring'
import { useLivePitchIndicator } from '../hooks/useLivePitchIndicator'
import { useMicPitch } from '../hooks/useMicPitch'
import { useScoring } from '../hooks/useScoring'

const DEFAULT_SONG_ID = 'test-song'

function getSongId(): string {
  return new URLSearchParams(window.location.search).get('song') ?? DEFAULT_SONG_ID
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
  const [notes, setNotes] = useState<NoteEvent[]>([])
  const [lyrics, setLyrics] = useState<LyricWord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [finalScore, setFinalScore] = useState<number | null>(null)
  const [debug, setDebug] = useState(false)
  const [recentNotes, setRecentNotes] = useState<RecentNoteStat[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const { status: micStatus, start: startMic, latestSampleRef } = useMicPitch(audioRef)
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

  useEffect(() => {
    let cancelled = false
    setLoadState('loading')

    Promise.all([
      fetchJson<NoteEvent[]>(`/cache/${songId}/notes.json`),
      fetchJson<LyricWord[]>(`/cache/${songId}/lyrics.json`),
    ])
      .then(([notesData, lyricsData]) => {
        if (cancelled) return
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

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || loadState !== 'loaded') return

    const handlePlay = () => {
      // Only treat this as a fresh attempt (reset the scorecard) when playback is starting from
      // the beginning, not on every resume-after-pause mid-song.
      if (audio.currentTime < 0.5) {
        resetScoring()
        setFinalScore(null)
      }
    }
    const handleEnded = () => setFinalScore(getFinalScore())

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

  return (
    <main className="game-screen">
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
              <h2>{songId}</h2>
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
                {micStatus === 'active' && <span className="score-badge">Score: {runningScore}%</span>}
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
            <audio ref={audioRef} src={`/cache/${songId}/instrumental.wav`} controls />
            {micStatus === 'active' && (
              <div className="pitch-readout">
                <span className="pitch-readout-item">
                  <span className="pitch-readout-label">Target</span>
                  <span className="pitch-readout-value">{indicatorDisplay.targetNoteName ?? '—'}</span>
                </span>
                <span className="pitch-readout-item">
                  <span className="pitch-readout-label">You</span>
                  <span className="pitch-readout-value">{indicatorDisplay.detectedNoteName ?? '—'}</span>
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
            />
          </div>
          {debug && (
            <div className="debug-panel">
              <div className="debug-panel-row">
                <span>raw</span>
                <span>
                  {indicatorDisplay.rawHz !== null ? `${indicatorDisplay.rawHz}Hz` : '—'}
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
