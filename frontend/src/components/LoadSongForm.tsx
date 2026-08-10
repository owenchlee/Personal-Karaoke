import { useEffect, useRef, useState } from 'react'
import { getStageLabel } from '../game/jobStages'
import type { JobState } from '../types/job'
import { LinkIcon } from './icons'

const POLL_INTERVAL_MS = 2000

interface LoadSongFormProps {
  onLoaded: (slug: string) => void
}

type LanguageOption = '' | 'en' | 'yue'
type SeparationModelOption = 'htdemucs' | 'htdemucs_ft' | 'bs_roformer'

interface GpuStatus {
  available: boolean
  name: string | null
}

function LoadSongForm({ onLoaded }: LoadSongFormProps) {
  const [url, setUrl] = useState('')
  const [language, setLanguage] = useState<LanguageOption>('')
  const [separationModel, setSeparationModel] = useState<SeparationModelOption>('htdemucs')
  const [job, setJob] = useState<JobState | null>(null)
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => stopPolling, [])

  useEffect(() => {
    fetch('/api/gpu-status')
      .then((response) => (response.ok ? (response.json() as Promise<GpuStatus>) : null))
      .then((status) => setGpuStatus(status))
      .catch(() => setGpuStatus(null))
  }, [])

  const pollJob = (jobId: string) => {
    pollRef.current = setInterval(() => {
      fetch(`/api/jobs/${jobId}`)
        .then((response) => {
          if (!response.ok) throw new Error(`Job status request failed: ${response.status}`)
          return response.json() as Promise<JobState>
        })
        .then((state) => {
          setJob(state)
          if (state.status === 'done' && state.slug) {
            stopPolling()
            onLoaded(state.slug)
          } else if (state.status === 'error') {
            stopPolling()
          }
        })
        .catch((error: Error) => {
          stopPolling()
          setJob({ status: 'error', progress: 0, slug: null, error: error.message })
        })
    }, POLL_INTERVAL_MS)
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    if (!url.trim() || isBusy) return

    setJob({ status: 'queued', progress: 0, slug: null, error: null })
    fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: url.trim(),
        language: language || null,
        separation_model: separationModel,
      }),
    })
      .then((response) => {
        if (response.status === 502) {
          throw new Error(
            "Can't reach the backend job server. Start it with " +
              '`venv/Scripts/python.exe scripts/server.py` and try again.',
          )
        }
        if (!response.ok) throw new Error(`Couldn't start processing: ${response.status}`)
        return response.json() as Promise<{ job_id: string }>
      })
      .then(({ job_id }) => pollJob(job_id))
      .catch((error: Error) => {
        setJob({ status: 'error', progress: 0, slug: null, error: error.message })
      })
  }

  const isBusy = job !== null && job.status !== 'done' && job.status !== 'error'

  return (
    <form className="load-song-form" onSubmit={handleSubmit}>
      <label className="field-label" htmlFor="song-link">
        Load a song from a link
      </label>
      <div className="input-row">
        <div className="input-with-icon">
          <span className="input-icon">
            <LinkIcon size={18} />
          </span>
          <input
            id="song-link"
            className="input"
            type="url"
            inputMode="url"
            placeholder="Paste a YouTube link…"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            disabled={isBusy}
            required
          />
        </div>
        <select
          id="song-language"
          className="select-language"
          value={language}
          onChange={(event) => setLanguage(event.target.value as LanguageOption)}
          disabled={isBusy}
          aria-label="Song language"
        >
          <option value="">Auto-detect language</option>
          <option value="en">English</option>
          <option value="yue">Cantonese</option>
        </select>
        <select
          id="song-separation-model"
          className="select-language"
          value={separationModel}
          onChange={(event) => setSeparationModel(event.target.value as SeparationModelOption)}
          disabled={isBusy}
          aria-label="Vocal separation quality"
        >
          <option value="htdemucs">Fast (default)</option>
          <option value="htdemucs_ft">Better quality (slower)</option>
          <option value="bs_roformer">Best quality (slowest)</option>
        </select>
        <button type="submit" className="btn btn-primary" disabled={isBusy || !url.trim()}>
          {isBusy ? 'Processing…' : 'Load'}
        </button>
      </div>
      <p className="muted form-hint">
        Processing runs locally on this machine and isn't uploaded anywhere: use links you have the rights to.
      </p>

      {gpuStatus && (
        <p className="gpu-status muted" role="status">
          {gpuStatus.available
            ? `NVIDIA GPU detected${gpuStatus.name ? ` (${gpuStatus.name})` : ''}: processing will use it to speed things up`
            : 'No NVIDIA GPU detected: processing will run on CPU'}
        </p>
      )}

      {job && job.status !== 'error' && (
        <div className="job-progress" role="status">
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${job.progress * 100}%` }} />
          </div>
          <p className="muted">
            {getStageLabel(job.status)} {Math.round(job.progress * 100)}%
          </p>
        </div>
      )}

      {job && job.status === 'error' && (
        <p className="form-error" role="alert">
          {job.error ?? 'Something went wrong.'}
        </p>
      )}
    </form>
  )
}

export default LoadSongForm
