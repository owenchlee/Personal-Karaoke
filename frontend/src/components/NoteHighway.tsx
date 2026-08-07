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

const CANVAS_WIDTH_FALLBACK = 900
const MIN_CANVAS_WIDTH = 240
const CANVAS_HEIGHT_FALLBACK = 380
const MIN_CANVAS_HEIGHT = 240
const PLAYHEAD_X = 150
const MARGIN_TOP = 20
const MARGIN_BOTTOM = 20
const NOTE_BAR_HEIGHT = 10
const PITCH_PILL_WIDTH = 26
const PITCH_PILL_HEIGHT = 12

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

interface NoteColors {
  pending: string
  hit: string
  missed: string
  gridLine: string
  gridLabel: string
  playhead: string
  pitchIndicator: string
  pitchIndicatorStroke: string
}

// Canvas can't consume CSS custom properties directly, so colors live as tokens in index.css and
// get read off the DOM once (and again on a theme change) rather than duplicated as JS literals --
// that's what lets this highway theme correctly in both light and dark mode instead of always
// rendering on a hardcoded background.
function readNoteColors(canvas: HTMLCanvasElement): NoteColors {
  const style = getComputedStyle(canvas)
  const read = (name: string) => style.getPropertyValue(name).trim()
  return {
    pending: read('--note-pending'),
    hit: read('--note-hit'),
    missed: read('--note-missed'),
    gridLine: read('--note-grid-line'),
    gridLabel: read('--note-grid-label'),
    playhead: read('--note-playhead'),
    pitchIndicator: read('--pitch-indicator'),
    pitchIndicatorStroke: read('--pitch-indicator-stroke'),
  }
}

interface NoteHighwayProps {
  audioRef: React.RefObject<HTMLAudioElement | null>
  notes: NoteEvent[]
  indicatorRef?: RefObject<LiveIndicatorState>
  accuraciesRef?: RefObject<NoteAccuracy[]>
  /** Buffer + calibration mic latency (`useMicPitch.ts`), held for the whole mic session -- see the
   * note above `currentTime` in the draw loop below for why the highway itself needs this, not
   * just scoring/the live pill. */
  latencySecondsRef?: RefObject<number>
}

function NoteHighway({ audioRef, notes, indicatorRef, accuraciesRef, latencySecondsRef }: NoteHighwayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const renderedYRef = useRef<number | null>(null)
  const lastFrameTimestampRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    const container = canvas?.parentElement
    if (!canvas || !ctx || !container) return

    const pitchRange = getPitchRange(notes)
    const noteIndices = new Map(notes.map((note, index) => [note, index]))

    let colors = readNoteColors(canvas)
    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleThemeChange = () => {
      colors = readNoteColors(canvas)
    }
    darkModeQuery.addEventListener('change', handleThemeChange)

    // Sized to the container's content box (not stretched via CSS width/height: 100%, and not a
    // fixed pixel constant) and rescaled for devicePixelRatio, so the highway stays crisp at any
    // viewport size/DPR and fills however much space its container (`.stage`, sized in CSS to a
    // large chunk of the viewport -- see index.css) actually gives it, rather than blurring when a
    // fixed-resolution bitmap gets stretched, or rendering into a small fixed-height strip.
    let cssWidth = Math.max(MIN_CANVAS_WIDTH, Math.round(container.clientWidth)) || CANVAS_WIDTH_FALLBACK
    let cssHeight =
      Math.max(MIN_CANVAS_HEIGHT, Math.round(container.clientHeight)) || CANVAS_HEIGHT_FALLBACK
    let dpr = window.devicePixelRatio || 1

    const applyCanvasSize = () => {
      canvas.width = Math.round(cssWidth * dpr)
      canvas.height = Math.round(cssHeight * dpr)
      canvas.style.width = `${cssWidth}px`
      canvas.style.height = `${cssHeight}px`
    }
    applyCanvasSize()

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const nextWidth = Math.max(MIN_CANVAS_WIDTH, Math.round(entry.contentRect.width))
      const nextHeight = Math.max(MIN_CANVAS_HEIGHT, Math.round(entry.contentRect.height))
      if (nextWidth !== cssWidth || nextHeight !== cssHeight) {
        cssWidth = nextWidth
        cssHeight = nextHeight
        applyCanvasSize()
      }
    })
    resizeObserver.observe(container)

    const draw = (timestamp: number) => {
      const lastTimestamp = lastFrameTimestampRef.current
      const dt = lastTimestamp === null ? 1 / 60 : (timestamp - lastTimestamp) / 1000
      lastFrameTimestampRef.current = timestamp

      dpr = window.devicePixelRatio || dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      // Scrolling notes off the raw audio clock would reach the playhead before the mic pipeline
      // can possibly judge them -- pitch detection only ever produces a sample after buffer +
      // calibration latency has elapsed (`useMicPitch.ts`), so a note that's already scrolled past
      // the playhead by the time that judgment lands reads as "the highway moved too early." Holding
      // the highway back by that same latency (rather than the raw audio position) keeps whatever
      // bar sits at the playhead in sync with what's actually being judged right now.
      const currentTime = (audioRef.current?.currentTime ?? 0) - (latencySecondsRef?.current ?? 0)
      const futureWindowSeconds = (cssWidth - PLAYHEAD_X) / DEFAULT_PX_PER_SECOND

      ctx.clearRect(0, 0, cssWidth, cssHeight)

      // Absolute note-name gridlines -- see the mapping-strategy note on `getPitchRange` in
      // coords.ts: without these, "near the bottom of the highway" only means "the lowest note in
      // this song," which reads as an absolute "low pitch" even when it isn't one.
      ctx.font = '10px system-ui, sans-serif'
      ctx.textBaseline = 'middle'
      for (const tick of getPitchAxisTicks(pitchRange)) {
        const y = pitchToY(tick, pitchRange, cssHeight, MARGIN_TOP, MARGIN_BOTTOM)
        ctx.strokeStyle = colors.gridLine
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(cssWidth, y)
        ctx.stroke()
        ctx.fillStyle = colors.gridLabel
        ctx.fillText(midiToNoteName(tick), 4, y)
      }

      ctx.strokeStyle = colors.playhead
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(PLAYHEAD_X, 0)
      ctx.lineTo(PLAYHEAD_X, cssHeight)
      ctx.stroke()

      const visibleNotes = getVisibleNotes(
        notes,
        currentTime,
        DEFAULT_PAST_BUFFER_SECONDS,
        futureWindowSeconds,
      )

      // Bars start white and only flip to a hit/missed color once a note has actually finished
      // playing -- coloring it live, while the note is still in progress, made the color flicker
      // back and forth as the cumulative hit fraction crossed the 50% line sample to sample, which
      // read as broken rather than informative.
      for (const note of visibleNotes) {
        const startX = timeToX(note.onset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const endX = timeToX(note.offset, currentTime, DEFAULT_PX_PER_SECOND, PLAYHEAD_X)
        const y = pitchToY(note.pitch_midi, pitchRange, cssHeight, MARGIN_TOP, MARGIN_BOTTOM)
        const radius = Math.min(NOTE_BAR_HEIGHT / 2, Math.max(endX - startX, 0) / 2)

        let color = colors.pending
        const accuracy = accuraciesRef?.current[noteIndices.get(note) ?? -1]
        if (accuracy && note.offset <= currentTime) {
          color = isNoteHit(accuracy) ? colors.hit : colors.missed
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
        const targetY = pitchToY(indicator.smoothedMidi, pitchRange, cssHeight, MARGIN_TOP, MARGIN_BOTTOM)

        // Snap straight to the target on the first frame after a silence (rather than easing in
        // from wherever the pill last was), so it doesn't visibly sweep across the highway when
        // singing resumes.
        if (renderedYRef.current === null) {
          renderedYRef.current = targetY
        } else {
          const easing = 1 - Math.exp(-dt / RENDER_SMOOTHING_TAU_SECONDS)
          renderedYRef.current += (targetY - renderedYRef.current) * easing
        }

        ctx.fillStyle = colors.pitchIndicator
        ctx.strokeStyle = colors.pitchIndicatorStroke
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
      resizeObserver.disconnect()
      darkModeQuery.removeEventListener('change', handleThemeChange)
    }
  }, [audioRef, notes, indicatorRef, accuraciesRef, latencySecondsRef])

  return <canvas ref={canvasRef} width={CANVAS_WIDTH_FALLBACK} height={CANVAS_HEIGHT_FALLBACK} />
}

export default NoteHighway
