import { describe, expect, it } from 'vitest'
import type { LyricWord } from '../types/lyrics'
import {
  getCurrentLineIndex,
  getCurrentWordIndex,
  getWordProgress,
  groupWordsByLine,
  INSTRUMENTAL_GAP_SECONDS,
  mergeSingleWordLines,
  MUSIC_NOTE,
  withInstrumentalBreaks,
} from './lyrics'

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

describe('withInstrumentalBreaks', () => {
  it('returns an empty array for no words', () => {
    expect(withInstrumentalBreaks([])).toEqual([])
  })

  it('leaves words untouched (aside from line renumbering) when gaps stay below the threshold', () => {
    const result = withInstrumentalBreaks(words)
    expect(result.map((w) => w.word)).toEqual(['hello', 'world', 'again'])
    expect(result.every((w) => !w.isInstrumental)).toBe(true)
    expect(result.map((w) => w.line)).toEqual([0, 0, 1])
  })

  it('inserts a single leading instrumental note before a long intro, spanning the whole gap', () => {
    const introWords: LyricWord[] = [{ word: 'hello', start: 10.0, end: 10.5, line: 0 }]
    const result = withInstrumentalBreaks(introWords)

    const notes = result.filter((w) => w.isInstrumental)
    expect(notes).toHaveLength(1)
    expect(notes[0].word).toBe(MUSIC_NOTE)
    expect(notes[0].start).toBe(0)
    expect(notes[0].end).toBe(10.0)
    expect(notes[0].line).toBe(0)

    const realWord = result.find((w) => !w.isInstrumental)
    expect(realWord?.word).toBe('hello')
    expect(realWord?.line).toBe(1)
  })

  it('inserts a single instrumental note between lines when the gap is long enough', () => {
    const withBridge: LyricWord[] = [
      { word: 'verse', start: 1.0, end: 1.5, line: 0 },
      { word: 'chorus', start: 1.5 + INSTRUMENTAL_GAP_SECONDS, end: 2.5 + INSTRUMENTAL_GAP_SECONDS, line: 1 },
    ]
    const result = withInstrumentalBreaks(withBridge)

    expect(result.map((w) => w.isInstrumental)).toEqual([undefined, true, undefined])
    expect(result[0].line).toBe(0)
    expect(result[1].line).toBe(1)
    expect(result[1].start).toBe(1.5)
    expect(result[1].end).toBe(1.5 + INSTRUMENTAL_GAP_SECONDS)
    expect(result[2].line).toBe(2)
  })

  it('does not insert a break for a short, natural pause between words', () => {
    const result = withInstrumentalBreaks(words)
    expect(result.some((w) => w.isInstrumental)).toBe(false)
  })
})

describe('mergeSingleWordLines', () => {
  it('returns an empty array for no words', () => {
    expect(mergeSingleWordLines([])).toEqual([])
  })

  it('leaves already-multi-word lines untouched', () => {
    const multiWordOnly: LyricWord[] = [
      { word: 'hello', start: 1.0, end: 1.5, line: 0 },
      { word: 'world', start: 2.0, end: 2.5, line: 0 },
      { word: 'once', start: 3.0, end: 3.5, line: 1 },
      { word: 'again', start: 3.6, end: 4.0, line: 1 },
    ]
    expect(mergeSingleWordLines(multiWordOnly)).toEqual(multiWordOnly)
  })

  it('merges a single-word line into the closer neighbor', () => {
    const withStrayWord: LyricWord[] = [
      { word: 'hello', start: 1.0, end: 1.5, line: 0 },
      { word: 'there', start: 1.6, end: 2.0, line: 0 },
      { word: 'friend', start: 2.1, end: 2.5, line: 1 }, // stray one-word line, close to line 0
      { word: 'good', start: 5.0, end: 5.3, line: 2 },
      { word: 'morning', start: 5.4, end: 5.8, line: 2 },
    ]
    const result = mergeSingleWordLines(withStrayWord)

    expect(result.map((w) => w.word)).toEqual(['hello', 'there', 'friend', 'good', 'morning'])
    expect(result.map((w) => w.line)).toEqual([0, 0, 0, 1, 1])
  })

  it('merges into the next line when it is the closer neighbor', () => {
    const withStrayWord: LyricWord[] = [
      { word: 'good', start: 1.0, end: 1.3, line: 0 },
      { word: 'morning', start: 1.4, end: 1.8, line: 0 },
      { word: 'hey', start: 4.5, end: 4.8, line: 1 }, // stray one-word line, close to line 2
      { word: 'you', start: 5.0, end: 5.3, line: 2 },
      { word: 'there', start: 5.4, end: 5.8, line: 2 },
    ]
    const result = mergeSingleWordLines(withStrayWord)

    expect(result.map((w) => w.word)).toEqual(['good', 'morning', 'hey', 'you', 'there'])
    expect(result.map((w) => w.line)).toEqual([0, 0, 1, 1, 1])
  })

  it('leaves a single-word line standing alone when real instrumental gaps box it in on both sides', () => {
    const bigGap = INSTRUMENTAL_GAP_SECONDS + 0.5 // clearly past the threshold, not a float-precision edge case
    const isolatedWord: LyricWord[] = [
      { word: 'intro', start: 0.0, end: 0.3, line: 0 },
      { word: 'yeah', start: 0.3 + bigGap, end: 0.6 + bigGap, line: 1 },
      { word: 'outro', start: 0.6 + 2 * bigGap, end: 0.9 + 2 * bigGap, line: 2 },
    ]
    const result = mergeSingleWordLines(isolatedWord)

    expect(result.map((w) => w.word)).toEqual(['intro', 'yeah', 'outro'])
    expect(result.map((w) => w.line)).toEqual([0, 1, 2])
  })

  it('does not merge a single-word first line into a distant second line across an instrumental gap', () => {
    const bigGap = INSTRUMENTAL_GAP_SECONDS + 0.5
    const introWord: LyricWord[] = [
      { word: 'oh', start: 0.0, end: 0.3, line: 0 },
      { word: 'verse', start: 0.3 + bigGap, end: 0.6 + bigGap, line: 1 },
      { word: 'continues', start: 0.7 + bigGap, end: 1.0 + bigGap, line: 1 },
    ]
    const result = mergeSingleWordLines(introWord)

    expect(result.map((w) => w.word)).toEqual(['oh', 'verse', 'continues'])
    expect(result.map((w) => w.line)).toEqual([0, 1, 1])
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
