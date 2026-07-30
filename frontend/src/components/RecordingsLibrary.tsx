import { useEffect, useState } from 'react'
import { formatRecordedAt, sortRecordingsByRecency } from '../game/recordings'
import type { Recording } from '../types/recording'
import { TrashIcon } from './icons'

function RecordingsLibrary() {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    fetch('/api/recordings')
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load recordings: ${response.status}`)
        return response.json() as Promise<{ recordings: Recording[] }>
      })
      .then(({ recordings: fetched }) => {
        if (!cancelled) setRecordings(fetched)
      })
      .catch(() => {
        if (!cancelled) setRecordings([])
      })

    return () => {
      cancelled = true
    }
  }, [])

  const handleDelete = async (recording: Recording) => {
    if (!window.confirm(`Delete this recording of "${recording.title}"? This can't be undone.`)) {
      return
    }

    setDeleteError(null)
    setDeletingFilename(recording.filename)
    try {
      const response = await fetch(`/api/recordings/${encodeURIComponent(recording.filename)}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error(`Failed to delete recording: ${response.status}`)
      setRecordings((current) => current.filter((entry) => entry.filename !== recording.filename))
    } catch {
      setDeleteError(`Couldn't delete the recording of "${recording.title}". Try again.`)
    } finally {
      setDeletingFilename(null)
    }
  }

  if (recordings.length === 0) return <p className="muted">No recordings yet.</p>

  return (
    <div className="recordings-library">
      {deleteError && <p className="form-error">{deleteError}</p>}
      <ul className="recordings-list">
        {sortRecordingsByRecency(recordings).map((rec) => {
          const date = formatRecordedAt(rec.recorded_at)
          return (
            <li key={rec.filename} className="recordings-row">
              <div className="recordings-info">
                <span className="recordings-title">{rec.title}</span>
                {date && <span className="recordings-date">{date}</span>}
              </div>
              <a
                className="btn btn-secondary recordings-action"
                href={`/api/recordings/${encodeURIComponent(rec.filename)}`}
                download
              >
                Download
              </a>
              <button
                type="button"
                className="recordings-delete"
                onClick={() => handleDelete(rec)}
                disabled={deletingFilename === rec.filename}
                aria-label={`Delete recording of ${rec.title}`}
                title={`Delete recording of ${rec.title}`}
              >
                <TrashIcon size={16} />
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default RecordingsLibrary
