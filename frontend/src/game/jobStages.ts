import type { JobStatus } from '../types/job'

const STAGE_LABELS: Record<JobStatus, string> = {
  queued: 'Queued…',
  downloading: 'Downloading audio…',
  separating: 'Separating vocals from instrumental…',
  extracting_melody: 'Extracting melody…',
  transcribing_lyrics: 'Transcribing lyrics…',
  publishing: 'Publishing…',
  done: 'Done',
  error: 'Failed',
}

export function getStageLabel(status: JobStatus): string {
  return STAGE_LABELS[status]
}
