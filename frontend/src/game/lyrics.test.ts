import { describe, expect, it } from 'vitest'
import type { LyricWord } from '../types/lyrics'
import { getCurrentLineIndex, getCurrentWordIndex, groupWordsByLine } from './lyrics'

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
