import { useEffect, useState } from 'react'
import { formatProcessedAt, sortSongsByRecency } from '../game/songLibrary'
import type { Song } from '../types/song'

interface SongLibraryProps {
  onSelect: (slug: string) => void
  refreshKey?: number
}

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 7h16M9 7V4h6v3m-1 0v13a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1V7h6ZM10 11v6M14 11v6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function SongLibrary({ onSelect, refreshKey }: SongLibraryProps) {
  const [songs, setSongs] = useState<Song[]>([])
  const [deletingSlug, setDeletingSlug] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    fetch('/api/songs')
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load songs: ${response.status}`)
        return response.json() as Promise<{ songs: Song[] }>
      })
      .then(({ songs: fetched }) => {
        if (!cancelled) setSongs(fetched)
      })
      .catch(() => {
        if (!cancelled) setSongs([])
      })

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  const handleDelete = async (song: Song) => {
    if (!window.confirm(`Delete "${song.title}" from your cached songs? This can't be undone.`)) {
      return
    }

    setDeleteError(null)
    setDeletingSlug(song.slug)
    try {
      const response = await fetch(`/api/songs/${encodeURIComponent(song.slug)}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error(`Failed to delete song: ${response.status}`)
      setSongs((current) => current.filter((entry) => entry.slug !== song.slug))
    } catch {
      setDeleteError(`Couldn't delete "${song.title}". Try again.`)
    } finally {
      setDeletingSlug(null)
    }
  }

  if (songs.length === 0) return <p className="muted">No cached songs yet.</p>

  return (
    <div className="song-library">
      {deleteError && <p className="form-error">{deleteError}</p>}
      <ul className="song-library-list">
        {sortSongsByRecency(songs).map((song) => {
          const date = formatProcessedAt(song.processed_at)
          return (
            <li key={song.slug} className="song-library-row">
              <button
                type="button"
                className="song-library-select"
                onClick={() => onSelect(song.slug)}
              >
                <span className="song-library-title">{song.title}</span>
                {date && <span className="song-library-date">Added {date}</span>}
              </button>
              <button
                type="button"
                className="song-library-delete"
                onClick={() => handleDelete(song)}
                disabled={deletingSlug === song.slug}
                aria-label={`Delete ${song.title}`}
                title={`Delete ${song.title}`}
              >
                <TrashIcon />
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default SongLibrary
