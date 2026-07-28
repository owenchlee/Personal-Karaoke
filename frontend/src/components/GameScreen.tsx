import { useEffect, useRef, useState } from 'react'
import type { NoteEvent } from '../types/note'
import NoteHighway from './NoteHighway'

const DEFAULT_SONG_ID = 'test-song'

function getSongId(): string {
  return new URLSearchParams(window.location.search).get('song') ?? DEFAULT_SONG_ID
}

type LoadState = 'loading' | 'loaded' | 'error'

function GameScreen() {
  const [songId] = useState(getSongId)
  const [notes, setNotes] = useState<NoteEvent[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    let cancelled = false

    fetch(`/cache/${songId}/notes.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load notes.json: ${response.status}`)
        return response.json() as Promise<NoteEvent[]>
      })
      .then((data) => {
        if (cancelled) return
        setNotes(data)
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
          Couldn&rsquo;t load <code>/cache/{songId}/notes.json</code>. Run{' '}
          <code>scripts/publish_song.py {songId}</code> first.
        </p>
      )}
      {loadState === 'loaded' && (
        <>
          <audio ref={audioRef} src={`/cache/${songId}/instrumental.wav`} controls />
          <NoteHighway audioRef={audioRef} notes={notes} />
        </>
      )}
    </main>
  )
}

export default GameScreen
