import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { getCurrentLineIndex, getCurrentWordIndex, groupWordsByLine } from '../game/lyrics'
import type { LyricWord } from '../types/lyrics'

interface LyricsDisplayProps {
  audioRef: RefObject<HTMLAudioElement | null>
  words: LyricWord[]
}

const ACTIVE_CLASS = 'lyric-word--active'

function LyricsLine({ words, role }: { words: LyricWord[] | undefined; role: 'prev' | 'current' | 'next' }) {
  return (
    <div className={`lyric-line lyric-line--${role}`}>
      {words
        ? words.map((word) => (
            <span key={word.start} data-word-start={word.start} className="lyric-word">
              {word.word}
            </span>
          ))
        : ' '}
    </div>
  )
}

function LyricsDisplay({ audioRef, words }: LyricsDisplayProps) {
  const lines = useMemo(() => groupWordsByLine(words), [words])
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  // All three visible lines' spans live under this one wrapper, so looking
  // up the active word here works even in the same tick that the line
  // transitions -- the new line's spans were already in the DOM (rendered
  // as the "next" row a moment ago), just not yet re-tagged as "current".
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const activeElRef = useRef<HTMLElement | null>(null)
  const activeWordIndexRef = useRef(-2)

  useEffect(() => {
    let rafId: number

    const tick = () => {
      const currentTime = audioRef.current?.currentTime ?? 0
      const wordIndex = getCurrentWordIndex(words, currentTime)
      if (wordIndex !== activeWordIndexRef.current) {
        activeElRef.current?.classList.remove(ACTIVE_CLASS)
        activeElRef.current = null

        const lineIndex = getCurrentLineIndex(words, currentTime)
        setCurrentLineIndex((previous) => (previous === lineIndex ? previous : lineIndex))

        const activeWord = wordIndex === -1 ? undefined : words[wordIndex]
        if (activeWord) {
          const el = wrapperRef.current?.querySelector<HTMLElement>(
            `[data-word-start="${activeWord.start}"]`,
          )
          if (el) {
            el.classList.add(ACTIVE_CLASS)
            activeElRef.current = el
          }
        }
        activeWordIndexRef.current = wordIndex
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(rafId)
  }, [audioRef, words])

  return (
    <div ref={wrapperRef} style={{ padding: '1rem', textAlign: 'center' }}>
      <LyricsLine words={lines[currentLineIndex - 1]} role="prev" />
      <LyricsLine words={lines[currentLineIndex]} role="current" />
      <LyricsLine words={lines[currentLineIndex + 1]} role="next" />
    </div>
  )
}

export default LyricsDisplay
