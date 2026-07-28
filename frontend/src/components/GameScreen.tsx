import { useEffect, useRef, useState } from 'react'
import type { NoteEvent } from '../types/note'
import type { LyricWord } from '../types/lyrics'
import LyricsDisplay from './LyricsDisplay'
import NoteHighway from './NoteHighway'

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

function GameScreen() {
  const [songId] = useState(getSongId)
  const [notes, setNotes] = useState<NoteEvent[]>([])
  const [lyrics, setLyrics] = useState<LyricWord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    let cancelled = false

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

  return (
    <main style={{ fontFamily: 'sans-serif', maxWidth: 960, margin: '2rem auto' }}>
      <h1>Phase 2: note highway</h1>

      {loadState === 'loading' && <p>Loading song&hellip;</p>}
      {loadState === 'error' && (
        <p>
          Couldn&rsquo;t load cached song assets for <code>{songId}</code>. Run{' '}
          <code>scripts/publish_song.py {songId}</code> first.
        </p>
      )}
      {loadState === 'loaded' && (
        <>
          <audio ref={audioRef} src={`/cache/${songId}/instrumental.wav`} controls />
          <LyricsDisplay audioRef={audioRef} words={lyrics} />
          <div style={{ marginTop: '1.5rem' }}>
            <NoteHighway audioRef={audioRef} notes={notes} />
          </div>
        </>
      )}
    </main>
  )
}

export default GameScreen
