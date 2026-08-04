import { useEffect, useState } from 'react'
import { formatProcessedAt, sortSongLibrary } from '../game/songLibrary'
import { findBestScore } from '../game/scores'
import { CACHE_BASE, DEMO_MODE } from '../config'
import type { Song } from '../types/song'
import type { ScoreRecord } from '../types/score'
import { StarIcon, TrashIcon } from './icons'

// The static demo build has no job server behind it, so the library is read from a manifest
// baked into the build (see scripts/generate_demo_manifest.py) instead of the live API, and
// star/delete -- which mutate state on a server that doesn't exist here -- are hidden.
const SONGS_URL = DEMO_MODE ? `${CACHE_BASE}/manifest.json` : '/api/songs'

interface SongLibraryProps {
  onSelect: (slug: string) => void
  refreshKey?: number
}

function SongLibrary({ onSelect, refreshKey }: SongLibraryProps) {
  const [songs, setSongs] = useState<Song[]>([])
  const [scores, setScores] = useState<ScoreRecord[]>([])
  const [deletingSlug, setDeletingSlug] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [togglingStarSlug, setTogglingStarSlug] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    fetch(SONGS_URL)
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

  // Best-effort: a failed scores fetch just means no "Best: X%" badges show,
  // the song list itself still renders fine without it.
  useEffect(() => {
    let cancelled = false

    fetch('/api/scores')
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load scores: ${response.status}`)
        return response.json() as Promise<{ scores: ScoreRecord[] }>
      })
      .then(({ scores: fetched }) => {
        if (!cancelled) setScores(fetched)
      })
      .catch(() => {
        if (!cancelled) setScores([])
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

  const handleToggleStar = async (song: Song) => {
    const starred = !song.starred
    setTogglingStarSlug(song.slug)
    // Optimistic update -- reordering to the top on star should feel instant.
    setSongs((current) =>
      current.map((entry) => (entry.slug === song.slug ? { ...entry, starred } : entry)),
    )
    try {
      const response = await fetch(`/api/songs/${encodeURIComponent(song.slug)}/starred`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ starred }),
      })
      if (!response.ok) throw new Error(`Failed to update star: ${response.status}`)
    } catch {
      // Revert on failure.
      setSongs((current) =>
        current.map((entry) => (entry.slug === song.slug ? { ...entry, starred: !starred } : entry)),
      )
    } finally {
      setTogglingStarSlug(null)
    }
  }

  if (songs.length === 0) return <p className="muted">No cached songs yet.</p>

  return (
    <div className="song-library">
      {deleteError && <p className="form-error">{deleteError}</p>}
      <ul className="song-library-list">
        {sortSongLibrary(songs).map((song) => {
          const date = formatProcessedAt(song.processed_at)
          const best = findBestScore(scores, song.slug)
          return (
            <li key={song.slug} className="song-library-row">
              {!DEMO_MODE && (
                <button
                  type="button"
                  className={`song-library-star${song.starred ? ' song-library-star--active' : ''}`}
                  onClick={() => handleToggleStar(song)}
                  disabled={togglingStarSlug === song.slug}
                  aria-label={song.starred ? `Unstar ${song.title}` : `Star ${song.title}`}
                  title={song.starred ? `Unstar ${song.title}` : `Star ${song.title}`}
                >
                  <StarIcon size={16} filled={song.starred} />
                </button>
              )}
              <button
                type="button"
                className="song-library-select"
                onClick={() => onSelect(song.slug)}
              >
                <span className="song-library-title">{song.title}</span>
                {date && <span className="song-library-date">Added {date}</span>}
                {best !== null && <span className="song-library-score">Best: {best}%</span>}
              </button>
              {!DEMO_MODE && (
                <button
                  type="button"
                  className="song-library-delete"
                  onClick={() => handleDelete(song)}
                  disabled={deletingSlug === song.slug}
                  aria-label={`Delete ${song.title}`}
                  title={`Delete ${song.title}`}
                >
                  <TrashIcon size={16} />
                </button>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default SongLibrary
