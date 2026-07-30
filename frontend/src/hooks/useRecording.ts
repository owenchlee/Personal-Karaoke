import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { parseContentDispositionFilename } from '../game/recordings'

export type RecordingStatus = 'idle' | 'recording' | 'processing' | 'done' | 'error'

interface UseRecordingOptions {
  audioRef: RefObject<HTMLAudioElement | null>
  micStreamRef: RefObject<MediaStream | null>
  songId: string
}

export interface UseRecordingResult {
  status: RecordingStatus
  downloadUrl: string | null
  downloadFilename: string | null
  errorMessage: string | null
  start: () => Promise<void>
  stop: () => void
  reset: () => void
}

const FALLBACK_DOWNLOAD_FILENAME = 'recording.mp3'

// Browsers can't record straight to mp3 -- MediaRecorder only speaks the codecs the browser
// itself ships (webm/opus here). The actual mp3 is rendered server-side by the local job server
// via ffmpeg (already a hard dependency of this project, see NOTES.md), so no browser-side mp3
// encoder library is needed at all.
const PREFERRED_MIME_TYPE = 'audio/webm;codecs=opus'

/** Records the instrumental + live mic together as a single downloadable mp3, so a player can
 * keep their take after singing along. Mixes both sources into one `MediaStream` via the Web
 * Audio API -- the same "record what you hear" recipe any browser supports natively -- then hands
 * the recorded webm blob to the local job server to transcode. Recording is manual (start/stop
 * buttons), not tied to playback, matching how the mic itself is manually enabled. */
export function useRecording({ audioRef, micStreamRef, songId }: UseRecordingOptions): UseRecordingResult {
  const [status, setStatus] = useState<RecordingStatus>('idle')
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [downloadFilename, setDownloadFilename] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  // A media element can only ever be handed to `createMediaElementSource` once in its lifetime
  // (a second call throws) -- cached here so re-recording, or recording again after switching
  // songs, doesn't attempt that twice against the same underlying <audio> DOM node.
  const elementSourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const downloadUrlRef = useRef<string | null>(null)

  const revokeDownloadUrl = useCallback(() => {
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current)
      downloadUrlRef.current = null
    }
  }, [])

  const renderMp3 = useCallback(async () => {
    setStatus('processing')
    try {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      const response = await fetch(`/api/recordings/mp3?song_id=${encodeURIComponent(songId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/webm' },
        body: blob,
      })
      if (!response.ok) throw new Error(`Server responded ${response.status}`)

      const filename =
        parseContentDispositionFilename(response.headers.get('Content-Disposition')) ??
        FALLBACK_DOWNLOAD_FILENAME
      const mp3Blob = await response.blob()
      revokeDownloadUrl()
      const url = URL.createObjectURL(mp3Blob)
      downloadUrlRef.current = url
      setDownloadUrl(url)
      setDownloadFilename(filename)
      setStatus('done')
    } catch {
      setErrorMessage("Couldn't render the MP3 — make sure scripts/server.py is running.")
      setStatus('error')
    }
  }, [songId, revokeDownloadUrl])

  const start = useCallback(async () => {
    const audio = audioRef.current
    const micStream = micStreamRef.current
    if (!audio || !micStream) {
      setErrorMessage('Enable the mic before recording.')
      setStatus('error')
      return
    }

    setErrorMessage(null)
    revokeDownloadUrl()
    setDownloadUrl(null)
    setDownloadFilename(null)

    const context = audioContextRef.current ?? new AudioContext()
    audioContextRef.current = context
    await context.resume()

    let elementSource = elementSourceRef.current
    if (!elementSource) {
      // Routing the <audio> element through Web Audio "hijacks" its output -- it must be
      // reconnected to the context's own destination or the song goes silent for the listener.
      elementSource = context.createMediaElementSource(audio)
      elementSource.connect(context.destination)
      elementSourceRef.current = elementSource
    }

    const mixDestination = context.createMediaStreamDestination()
    elementSource.connect(mixDestination)
    context.createMediaStreamSource(micStream).connect(mixDestination)

    const mimeType = MediaRecorder.isTypeSupported(PREFERRED_MIME_TYPE) ? PREFERRED_MIME_TYPE : undefined
    const recorder = new MediaRecorder(mixDestination.stream, mimeType ? { mimeType } : undefined)
    chunksRef.current = []
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data)
    }
    recorder.onstop = () => void renderMp3()
    recorderRef.current = recorder
    recorder.start()
    setStatus('recording')
  }, [audioRef, micStreamRef, revokeDownloadUrl, renderMp3])

  const stop = useCallback(() => {
    recorderRef.current?.stop()
  }, [])

  const reset = useCallback(() => {
    revokeDownloadUrl()
    setDownloadUrl(null)
    setDownloadFilename(null)
    setErrorMessage(null)
    setStatus('idle')
  }, [revokeDownloadUrl])

  useEffect(() => revokeDownloadUrl, [revokeDownloadUrl])

  return { status, downloadUrl, downloadFilename, errorMessage, start, stop, reset }
}
