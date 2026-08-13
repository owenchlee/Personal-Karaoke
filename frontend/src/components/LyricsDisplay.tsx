import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import {
  getCurrentLineIndex,
  getCurrentWordIndex,
  getWordProgress,
  groupWordsByLine,
  mergeSingleWordLines,
  withInstrumentalBreaks,
} from '../game/lyrics'
import type { DisplayWord } from '../game/lyrics'
import type { LyricWord } from '../types/lyrics'

interface LyricsDisplayProps {
  audioRef: RefObject<HTMLAudioElement | null>
  words: LyricWord[]
}

const ACTIVE_CLASS = 'lyric-word--active'
const SUNG_CLASS = 'lyric-word--sung'

function LyricsLine({ words, role }: { words: DisplayWord[] | undefined; role: 'prev' | 'current' | 'next' }) {
  return (
    <div className={`lyric-line lyric-line--${role}`}>
      {words
        ? words.map((word) => (
            <span
              key={word.start}
              data-word-start={word.start}
              className={`lyric-word${word.isInstrumental ? ' lyric-word--instrumental' : ''}`}
            >
              <span className="lyric-word__fill">{word.word}</span>
            </span>
          ))
        : ' '}
    </div>
  )
}

function LyricsDisplay({ audioRef, words }: LyricsDisplayProps) {
  const displayWords = useMemo(() => withInstrumentalBreaks(mergeSingleWordLines(words)), [words])
  const lines = useMemo(() => groupWordsByLine(displayWords), [displayWords])
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  // All three visible lines' spans live under this one wrapper, so looking
  // up the active word here works even in the same tick that the line
  // transitions -- the new line's spans were already in the DOM (rendered
  // as the "next" row a moment ago), just not yet re-tagged as "current".
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const activeElRef = useRef<HTMLElement | null>(null)
  const activeFillElRef = useRef<HTMLElement | null>(null)
  const activeWordIndexRef = useRef(-2)

  useEffect(() => {
    let rafId: number

    const tick = () => {
      const currentTime = audioRef.current?.currentTime ?? 0
      const wordIndex = getCurrentWordIndex(displayWords, currentTime)
      if (wordIndex !== activeWordIndexRef.current) {
        activeElRef.current = null
        activeFillElRef.current = null

        const lineIndex = getCurrentLineIndex(displayWords, currentTime)
        setCurrentLineIndex((previous) => (previous === lineIndex ? previous : lineIndex))

        const activeWord = wordIndex === -1 ? undefined : displayWords[wordIndex]
        // Re-tag every visible word rather than just the one word that
        // changed, so seeking backward (which can move the active word
        // earlier than words already marked "sung") corrects itself instead
        // of leaving stale accent-colored text ahead of the playhead.
        const elements = wrapperRef.current?.querySelectorAll<HTMLElement>('.lyric-word') ?? []
        elements.forEach((el) => {
          const start = Number(el.dataset.wordStart)
          el.classList.remove(ACTIVE_CLASS, SUNG_CLASS)
          el.querySelector<HTMLElement>('.lyric-word__fill')?.style.removeProperty('--progress')
          if (!activeWord || start < activeWord.start) {
            el.classList.add(SUNG_CLASS)
          } else if (start === activeWord.start) {
            el.classList.add(ACTIVE_CLASS)
            activeElRef.current = el
            activeFillElRef.current = el.querySelector<HTMLElement>('.lyric-word__fill')
          }
        })
        activeWordIndexRef.current = wordIndex
      }

      if (activeFillElRef.current && activeWordIndexRef.current !== -1) {
        const activeWord = displayWords[activeWordIndexRef.current]
        const progress = getWordProgress(activeWord, currentTime)
        activeFillElRef.current.style.setProperty('--progress', `${(progress * 100).toFixed(2)}%`)
      }

      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(rafId)
  }, [audioRef, displayWords])

  return (
    <div ref={wrapperRef} className="lyrics-display">
      <LyricsLine words={lines[currentLineIndex - 1]} role="prev" />
      <LyricsLine words={lines[currentLineIndex]} role="current" />
      <LyricsLine words={lines[currentLineIndex + 1]} role="next" />
    </div>
  )
}

export default LyricsDisplay
