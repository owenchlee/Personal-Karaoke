import { describe, expect, it } from 'vitest'
import type { LyricWord } from '../types/lyrics'
import { getCurrentLineIndex, getCurrentWordIndex, getWordProgress, groupWordsByLine } from './lyrics'

const words: LyricWord[] = [
  { word: 'hello', start: 1.0, end: 1.5, line: 0 },
  { word: 'world', start: 2.0, end: 2.5, line: 0 },
  { word: 'again', start: 3.0, end: 3.5, line: 1 },
]

describe('getCurrentWordIndex', () => {
  it('returns the first word before any singing has started', () => {
    expect(getCurrentWordIndex(words, 0)).toBe(0)
  })

  it('returns the word currently being sung', () => {
    expect(getCurrentWordIndex(words, 1.2)).toBe(0)
    expect(getCurrentWordIndex(words, 2.2)).toBe(1)
  })

  it('returns the upcoming word during a gap between words', () => {
    expect(getCurrentWordIndex(words, 1.7)).toBe(1)
  })

  it('returns -1 after the last word has finished', () => {
    expect(getCurrentWordIndex(words, 4.0)).toBe(-1)
  })

  it('returns -1 for an empty word list', () => {
    expect(getCurrentWordIndex([], 5.0)).toBe(-1)
  })

  it('treats a word boundary (exactly at end) as finished', () => {
    expect(getCurrentWordIndex(words, 1.5)).toBe(1)
  })
})

describe('getWordProgress', () => {
  const word: LyricWord = { word: 'held', start: 10.0, end: 10.5, line: 0 }

  it('is 0 before the word starts', () => {
    expect(getWordProgress(word, 9.0)).toBe(0)
  })

  it('is 0 at the exact start', () => {
    expect(getWordProgress(word, 10.0)).toBe(0)
  })

  it('is 0.5 halfway through', () => {
    expect(getWordProgress(word, 10.25)).toBe(0.5)
  })

  it('is 1 at the exact end', () => {
    expect(getWordProgress(word, 10.5)).toBe(1)
  })

  it('clamps to 1 after the word has finished', () => {
    expect(getWordProgress(word, 20.0)).toBe(1)
  })

  it('takes longer to reach the same progress for a longer-held word', () => {
    const shortWord: LyricWord = { word: 'a', start: 0, end: 0.5, line: 0 }
    const longWord: LyricWord = { word: 'b', start: 0, end: 4.0, line: 0 }

    expect(getWordProgress(shortWord, 0.25)).toBe(0.5)
    expect(getWordProgress(longWord, 0.25)).toBeLessThan(0.5)
  })

  it('returns 1 for a zero-duration word instead of dividing by zero', () => {
    const zeroDurationWord: LyricWord = { word: 'glitch', start: 5.0, end: 5.0, line: 0 }

    expect(getWordProgress(zeroDurationWord, 5.0)).toBe(1)
  })
})

describe('groupWordsByLine', () => {
  it('groups words into arrays indexed by their line number', () => {
    expect(groupWordsByLine(words)).toEqual([
      [words[0], words[1]],
      [words[2]],
    ])
  })

  it('returns an empty array for no words', () => {
    expect(groupWordsByLine([])).toEqual([])
  })
})

describe('getCurrentLineIndex', () => {
  it('returns the line of the currently active word', () => {
    expect(getCurrentLineIndex(words, 1.2)).toBe(0)
    expect(getCurrentLineIndex(words, 3.2)).toBe(1)
  })

  it('stays on the last line once lyrics are finished', () => {
    expect(getCurrentLineIndex(words, 10.0)).toBe(1)
  })

  it('returns -1 for no words', () => {
    expect(getCurrentLineIndex([], 1.0)).toBe(-1)
  })
})
