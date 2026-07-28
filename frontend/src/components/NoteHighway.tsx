import { useEffect, useRef } from 'react'
import type { NoteEvent } from '../types/note'
import {
  DEFAULT_PAST_BUFFER_SECONDS,
  DEFAULT_PX_PER_SECOND,
  getPitchRange,
  getVisibleNotes,
  pitchToY,
  timeToX,
} from '../game/coords'

const CANVAS_WIDTH = 900
const CANVAS_HEIGHT = 320
const PLAYHEAD_X = 150
const MARGIN_TOP = 20
const MARGIN_BOTTOM = 20
const NOTE_BAR_HEIGHT = 10
const FUTURE_WINDOW_SECONDS = (CANVAS_WIDTH - PLAYHEAD_X) / DEFAULT_PX_PER_SECOND

interface NoteHighwayProps {
  audioRef: React.RefObject<HTMLAudioElement | null>
  notes: NoteEvent[]
}

function NoteHighway({ audioRef, notes }: NoteHighwayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const pitchRange = getPitchRange(notes)

    const draw = () => {
      const currentTime = audioRef.current?.currentTime ?? 0

      ctx.fillStyle = '#111318'
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

      ctx.strokeStyle = '#f5f5f5'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(PLAYHEAD_X, 0)
      ctx.lineTo(PLAYHEAD_X, CANVAS_HEIGHT)
      ctx.stroke()

      const visibleNotes = getVisibleNotes(
        notes,
        currentTime,
        DEFAULT_PAST_BUFFER_SECONDS,
        FUTURE_WINDOW_SECONDS,
      )

      ctx.fillStyle = '#4caf50'
      for (const note of visibleNotes) {
        const startX = timeToX(note.onset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const endX = timeToX(note.offset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const y = pitchToY(note.pitch_midi, pitchRange, CANVAS_HEIGHT, MARGIN_TOP, MARGIN_BOTTOM)
        ctx.fillRect(startX, y - NOTE_BAR_HEIGHT / 2, endX - startX, NOTE_BAR_HEIGHT)
      }

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [audioRef, notes])

  return <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
}

export default NoteHighway
