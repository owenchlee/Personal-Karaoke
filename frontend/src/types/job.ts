export type JobStatus =
  | 'queued'
  | 'downloading'
  | 'separating'
  | 'extracting_melody'
  | 'transcribing_lyrics'
  | 'publishing'
  | 'done'
  | 'error'

export interface JobState {
  status: JobStatus
  /** 0-1 fraction of real progress through the whole job, weighted by stage. */
  progress: number
  slug: string | null
  error: string | null
}
