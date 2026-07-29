import type { LyricWord } from '../types/lyrics'

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
export function groupWordsByLine(words: LyricWord[]): LyricWord[][] {
  const lines: LyricWord[][] = []
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
