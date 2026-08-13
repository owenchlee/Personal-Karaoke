import type { LyricWord } from '../types/lyrics'

export interface DisplayWord extends LyricWord {
  isInstrumental?: boolean
}

/** Minimum silence (seconds) between words before it counts as an instrumental break (intro,
 * solo, bridge, outro) rather than just a natural pause between lyrics. */
export const INSTRUMENTAL_GAP_SECONDS = 3
export const MUSIC_NOTE = '♪'

/**
 * Expands `words` into a display-ready list where instrumental gaps of
 * `INSTRUMENTAL_GAP_SECONDS` or more become a single music-note placeholder word spanning the
 * whole gap. It reuses the exact LyricWord shape, so the existing active/progress-fill
 * highlighting in LyricsDisplay fills it in over the gap's real duration just like a sung word --
 * showing the user that music is playing, and how long it lasts, with no separate rendering path.
 * Note: this doesn't trust `words[].line` boundaries when deciding where a gap falls -- a big
 * enough real-time gap always splits into its own line even if the source data (e.g. a Whisper
 * segment that silently spans dead air) called it the same line as its neighbor.
 */
export function withInstrumentalBreaks(words: LyricWord[]): DisplayWord[] {
  if (words.length === 0) return []

  const result: DisplayWord[] = []
  let line = 0
  let previousEnd = 0
  let previousOriginalLine: number | null = null

  for (const word of words) {
    if (word.start - previousEnd >= INSTRUMENTAL_GAP_SECONDS) {
      // The break gets its own line, separate from whatever real line preceded it (unless it's
      // the very first thing in the song, in which case line 0 is free for it to take).
      if (result.length > 0) line++
      result.push({ word: MUSIC_NOTE, start: previousEnd, end: word.start, line, isInstrumental: true })
      line++
    } else if (previousOriginalLine !== null && word.line !== previousOriginalLine) {
      line++
    }
    result.push({ ...word, line })
    previousEnd = word.end
    previousOriginalLine = word.line
  }

  return result
}

/** A line with just one word flashes by as the "current" line and vanishes before a singer can
 * orient to it -- see mergeSingleWordLines. Anything below this joins whichever neighboring line
 * it's acoustically closest to. */
const MIN_LINE_WORDS = 2
// Safety valve only, to stop a long run of consecutive single-word lines from all cascading into
// one unbounded line -- not a general line-length cap (real sung lines legitimately run longer).
const MAX_MERGED_LINE_WORDS = 20

/**
 * Repairs the "one line per word" pathology that local (Whisper) transcription can produce, where
 * each decoder segment -- not a real phrase boundary -- becomes its own line (see
 * audio_pipeline/lyrics_extraction.py's `_flatten_words`). Deliberately conservative: it never
 * touches a line that already has more than one word (e.g. lrclib's real, human-authored line
 * breaks), and only merges a single-word line into a neighbor when they aren't separated by an
 * actual instrumental gap -- a one-word line boxed in by music on both sides is left standing
 * alone as a legitimate short interjection rather than force-merged into something distant.
 */
export function mergeSingleWordLines(words: LyricWord[]): LyricWord[] {
  if (words.length === 0) return []

  const lines = groupWordsByLine(words).filter((line): line is LyricWord[] => line.length > 0)

  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.length >= MIN_LINE_WORDS || lines.length === 1) {
      i++
      continue
    }

    const previous = lines[i - 1]
    const next = lines[i + 1]
    const gapToPrevious = previous ? line[0].start - previous[previous.length - 1].end : Infinity
    const gapToNext = next ? next[0].start - line[line.length - 1].end : Infinity
    const previousFits =
      previous !== undefined &&
      gapToPrevious < INSTRUMENTAL_GAP_SECONDS &&
      previous.length + line.length <= MAX_MERGED_LINE_WORDS
    const nextFits =
      next !== undefined &&
      gapToNext < INSTRUMENTAL_GAP_SECONDS &&
      next.length + line.length <= MAX_MERGED_LINE_WORDS

    if (previousFits && (!nextFits || gapToPrevious <= gapToNext)) {
      lines[i - 1] = [...previous, ...line]
      lines.splice(i, 1)
    } else if (nextFits) {
      lines[i + 1] = [...line, ...next]
      lines.splice(i, 1)
    } else {
      i++ // boxed in by real gaps on both sides -- nothing sensible to merge into
    }
  }

  const result: LyricWord[] = []
  lines.forEach((lineWords, lineIndex) => {
    lineWords.forEach((word) => result.push({ ...word, line: lineIndex }))
  })
  return result
}

/**
 * Index of the word that should be highlighted right now: the word currently
 * being sung, or -- during the gap between two words -- the upcoming one.
 * Returns -1 once every word has finished (or if there are no words).
 */
export function getCurrentWordIndex(words: LyricWord[], currentTime: number): number {
  return words.findIndex((word) => word.end > currentTime)
}

/**
 * Fraction (0-1) of `word` that's been sung by `currentTime`, for the
 * karaoke-style fill animation. 0 before the word starts (e.g. it's only
 * the "upcoming" word during a gap), 1 once its own end has passed. Longer
 * words divide the same 0-1 range over more real time, so the fill
 * naturally animates slower for them with no separate speed calculation.
 */
export function getWordProgress(word: LyricWord, currentTime: number): number {
  const duration = word.end - word.start
  if (duration <= 0) return 1
  return Math.min(1, Math.max(0, (currentTime - word.start) / duration))
}

/** Groups a flat word list back into lines, indexed by each word's `line`. */
export function groupWordsByLine<T extends LyricWord>(words: T[]): T[][] {
  const lines: T[][] = []
  for (const word of words) {
    ;(lines[word.line] ??= []).push(word)
  }
  return lines
}

/**
 * Line index that should be current right now, derived from whichever word
 * `getCurrentWordIndex` says is active. Once lyrics are finished (-1), stays
 * on the last line rather than resetting.
 */
export function getCurrentLineIndex(words: LyricWord[], currentTime: number): number {
  if (words.length === 0) return -1
  const wordIndex = getCurrentWordIndex(words, currentTime)
  if (wordIndex === -1) return words[words.length - 1].line
  return words[wordIndex].line
}
