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

/** Records the instrumental and live mic as two separate takes and uploads both to the local
 * job server, which auto-balances/cleans up the vocal and mixes them down into a single
 * downloadable mp3 (see audio_pipeline/mastering.py) -- so a player's voice comes through
 * clearly against the music instead of a raw, unprocessed mix. Each source is routed through
 * its own `MediaStreamAudioDestinationNode` via the Web Audio API and recorded with its own
 * `MediaRecorder`, started back-to-back so both tracks begin at effectively the same time (see
 * docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md's "Recording start offset"
 * section for why the two tracks can still end up slightly misaligned regardless, and how that's
 * corrected server-side). Recording is manual (start/stop buttons), not tied to playback,
 * matching how the mic itself is manually enabled. */
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
  const vocalRecorderRef = useRef<MediaRecorder | null>(null)
  const instrumentalRecorderRef = useRef<MediaRecorder | null>(null)
  const vocalChunksRef = useRef<BlobPart[]>([])
  const instrumentalChunksRef = useRef<BlobPart[]>([])
  // Each MediaRecorder's `onstop` fires independently and asynchronously -- this counts how
  // many of the two have actually finished flushing their last chunk, so the upload only
  // starts once both blobs are complete.
  const stoppedCountRef = useRef(0)
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
      const vocalBlob = new Blob(vocalChunksRef.current, { type: 'audio/webm' })
      const instrumentalBlob = new Blob(instrumentalChunksRef.current, { type: 'audio/webm' })
      const formData = new FormData()
      formData.append('vocal', vocalBlob, 'vocal.webm')
      formData.append('instrumental', instrumentalBlob, 'instrumental.webm')
      formData.append('song_id', songId)

      const response = await fetch('/api/recordings/mp3', { method: 'POST', body: formData })
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

    const vocalDestination = context.createMediaStreamDestination()
    context.createMediaStreamSource(micStream).connect(vocalDestination)

    const instrumentalDestination = context.createMediaStreamDestination()
    elementSource.connect(instrumentalDestination)

    const mimeType = MediaRecorder.isTypeSupported(PREFERRED_MIME_TYPE) ? PREFERRED_MIME_TYPE : undefined
    const vocalRecorder = new MediaRecorder(vocalDestination.stream, mimeType ? { mimeType } : undefined)
    const instrumentalRecorder = new MediaRecorder(instrumentalDestination.stream, mimeType ? { mimeType } : undefined)

    vocalChunksRef.current = []
    instrumentalChunksRef.current = []
    stoppedCountRef.current = 0

    vocalRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) vocalChunksRef.current.push(event.data)
    }
    instrumentalRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) instrumentalChunksRef.current.push(event.data)
    }

    const onEitherStop = () => {
      stoppedCountRef.current += 1
      if (stoppedCountRef.current === 2) void renderMp3()
    }
    vocalRecorder.onstop = onEitherStop
    instrumentalRecorder.onstop = onEitherStop

    vocalRecorderRef.current = vocalRecorder
    instrumentalRecorderRef.current = instrumentalRecorder
    // Started back-to-back (no `await` between the two calls) so both tracks begin capturing
    // at effectively the same time.
    vocalRecorder.start()
    instrumentalRecorder.start()
    setStatus('recording')
  }, [audioRef, micStreamRef, revokeDownloadUrl, renderMp3])

  const stop = useCallback(() => {
    vocalRecorderRef.current?.stop()
    instrumentalRecorderRef.current?.stop()
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
