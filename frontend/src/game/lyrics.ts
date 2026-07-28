import type { LyricWord } from '../types/lyrics'

/**
 * Index of the word that should be highlighted right now: the word currently
 * being sung, or -- during the gap between two words -- the upcoming one.
 * Returns -1 once every word has finished (or if there are no words).
 */
export function getCurrentWordIndex(words: LyricWord[], currentTime: number): number {
  return words.findIndex((word) => word.end > currentTime)
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
