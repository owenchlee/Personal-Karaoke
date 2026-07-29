import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import type { NoteEvent } from '../types/note'
import {
  DEFAULT_PAST_BUFFER_SECONDS,
  DEFAULT_PX_PER_SECOND,
  getPitchAxisTicks,
  getPitchRange,
  getVisibleNotes,
  midiToNoteName,
  pitchToY,
  timeToX,
} from '../game/coords'
import { isNoteHit, type NoteAccuracy } from '../game/scoring'
import type { LiveIndicatorState } from '../hooks/useLivePitchIndicator'

const CANVAS_WIDTH = 900
const CANVAS_HEIGHT = 320
const PLAYHEAD_X = 150
const MARGIN_TOP = 20
const MARGIN_BOTTOM = 20
const NOTE_BAR_HEIGHT = 10
const PITCH_PILL_WIDTH = 26
const PITCH_PILL_HEIGHT = 12
const FUTURE_WINDOW_SECONDS = (CANVAS_WIDTH - PLAYHEAD_X) / DEFAULT_PX_PER_SECOND

// Bars start white and only flip to a hit/missed color once a note has actually finished playing
// -- coloring it live, while the note is still in progress, made the color flicker back and forth
// as the cumulative hit fraction crossed the 50% line sample to sample, which read as broken
// rather than informative.
const NOTE_COLOR_PENDING = '#f8fafc'
const NOTE_COLOR_HIT = '#22c55e'
const NOTE_COLOR_MISSED = '#ef4444'

// The statistically-smoothed midi value from `useLivePitchIndicator` can still step frame-to-frame
// (a new sample entering/leaving its median window). Easing the on-screen position toward that
// target, rather than snapping straight to it, is what makes the pill read as a stable indicator
// instead of a jittery one. Expressed as a time constant so the motion looks the same regardless
// of frame rate. This is purely a rendering/presentation detail, so it stays local to this
// component rather than living in the shared hook.
// Was 0.1, then lowered to 0.05 to cut lag (combined with the smoothing window in
// `useLivePitchIndicator`, that pair was adding enough total lag that the pill visibly followed
// the highway rather than tracking it in real time -- see NOTES.md). Nudged back up to 0.08 for
// a smoother pill now that the lag itself is fixed -- still well under the original 0.1.
const RENDER_SMOOTHING_TAU_SECONDS = 0.08

interface NoteHighwayProps {
  audioRef: React.RefObject<HTMLAudioElement | null>
  notes: NoteEvent[]
  indicatorRef?: RefObject<LiveIndicatorState>
  accuraciesRef?: RefObject<NoteAccuracy[]>
}

function NoteHighway({ audioRef, notes, indicatorRef, accuraciesRef }: NoteHighwayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const renderedYRef = useRef<number | null>(null)
  const lastFrameTimestampRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const pitchRange = getPitchRange(notes)
    const noteIndices = new Map(notes.map((note, index) => [note, index]))

    const draw = (timestamp: number) => {
      const lastTimestamp = lastFrameTimestampRef.current
      const dt = lastTimestamp === null ? 1 / 60 : (timestamp - lastTimestamp) / 1000
      lastFrameTimestampRef.current = timestamp

      const currentTime = audioRef.current?.currentTime ?? 0

      ctx.fillStyle = '#0f0f17'
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

      // Absolute note-name gridlines -- see the mapping-strategy note on `getPitchRange` in
      // coords.ts: without these, "near the bottom of the highway" only means "the lowest note in
      // this song," which reads as an absolute "low pitch" even when it isn't one.
      ctx.font = '10px system-ui, sans-serif'
      ctx.textBaseline = 'middle'
      for (const tick of getPitchAxisTicks(pitchRange)) {
        const y = pitchToY(tick, pitchRange, CANVAS_HEIGHT, MARGIN_TOP, MARGIN_BOTTOM)
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(CANVAS_WIDTH, y)
        ctx.stroke()
        ctx.fillStyle = 'rgba(148, 163, 184, 0.7)'
        ctx.fillText(midiToNoteName(tick), 4, y)
      }

      ctx.strokeStyle = 'rgba(196, 181, 253, 0.8)'
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

      for (const note of visibleNotes) {
        const startX = timeToX(note.onset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const endX = timeToX(note.offset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const y = pitchToY(note.pitch_midi, pitchRange, CANVAS_HEIGHT, MARGIN_TOP, MARGIN_BOTTOM)
        const radius = Math.min(NOTE_BAR_HEIGHT / 2, Math.max(endX - startX, 0) / 2)

        let color = NOTE_COLOR_PENDING
        const accuracy = accuraciesRef?.current[noteIndices.get(note) ?? -1]
        if (accuracy && note.offset <= currentTime) {
          color = isNoteHit(accuracy) ? NOTE_COLOR_HIT : NOTE_COLOR_MISSED
        }

        ctx.fillStyle = color
        ctx.beginPath()
        ctx.roundRect(startX, y - NOTE_BAR_HEIGHT / 2, endX - startX, NOTE_BAR_HEIGHT, radius)
        ctx.fill()
      }

      // The fold/smoothing state machine itself lives in `useLivePitchIndicator`, shared with the
      // always-visible "Target vs You" readout and the debug panel in GameScreen -- this component
      // only owns the presentation detail of easing the pill smoothly toward that shared target.
      const indicator = indicatorRef?.current
      if (indicator && indicator.smoothedMidi !== null) {
        const targetY = pitchToY(indicator.smoothedMidi, pitchRange, CANVAS_HEIGHT, MARGIN_TOP, MARGIN_BOTTOM)

        // Snap straight to the target on the first frame after a silence (rather than easing in
        // from wherever the pill last was), so it doesn't visibly sweep across the highway when
        // singing resumes.
        if (renderedYRef.current === null) {
          renderedYRef.current = targetY
        } else {
          const easing = 1 - Math.exp(-dt / RENDER_SMOOTHING_TAU_SECONDS)
          renderedYRef.current += (targetY - renderedYRef.current) * easing
        }

        ctx.fillStyle = '#38bdf8'
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.roundRect(
          PLAYHEAD_X - PITCH_PILL_WIDTH / 2,
          renderedYRef.current - PITCH_PILL_HEIGHT / 2,
          PITCH_PILL_WIDTH,
          PITCH_PILL_HEIGHT,
          PITCH_PILL_HEIGHT / 2,
        )
        ctx.fill()
        ctx.stroke()
      } else {
        renderedYRef.current = null
      }

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [audioRef, notes, indicatorRef, accuraciesRef])

  return <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
}

export default NoteHighway
