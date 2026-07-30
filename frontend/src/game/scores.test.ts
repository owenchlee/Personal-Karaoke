import { describe, expect, it } from 'vitest'
import { findBestScore, getBadges, sortScoresByBest } from './scores'
import type { ScoreRecord } from '../types/score'

const score = (slug: string, best_score: number | null, play_count = 1): ScoreRecord => ({
  slug,
  title: slug,
  best_score,
  best_achieved_at: best_score !== null ? '2026-01-01T00:00:00Z' : null,
  play_count,
  last_played_at: '2026-01-01T00:00:00Z',
})

describe('sortScoresByBest', () => {
  it('returns an empty array unchanged', () => {
    expect(sortScoresByBest([])).toEqual([])
  })

  it('orders highest best_score first', () => {
    const low = score('a', 40)
    const high = score('b', 95)

    expect(sortScoresByBest([low, high])).toEqual([high, low])
  })

  it('does not mutate the input array', () => {
    const input = [score('a', 40), score('b', 95)]
    const copy = [...input]

    sortScoresByBest(input)

    expect(input).toEqual(copy)
  })
})

describe('findBestScore', () => {
  it('returns null when the song has no score record', () => {
    expect(findBestScore([score('a', 80)], 'missing')).toBeNull()
  })

  it('returns the best score for a matching slug', () => {
    expect(findBestScore([score('a', 80)], 'a')).toBe(80)
  })
})

describe('getBadges', () => {
  it('earns nothing with no scores', () => {
    expect(getBadges([]).every((badge) => !badge.earned)).toBe(true)
  })

  it('earns "first-note" after a single play', () => {
    const badges = getBadges([score('a', 50)])
    expect(badges.find((badge) => badge.id === 'first-note')?.earned).toBe(true)
  })

  it('earns "perfect-pitch" only with a 100 best_score somewhere', () => {
    expect(getBadges([score('a', 99)]).find((badge) => badge.id === 'perfect-pitch')?.earned).toBe(false)
    expect(getBadges([score('a', 100)]).find((badge) => badge.id === 'perfect-pitch')?.earned).toBe(true)
  })

  it('requires 3 different songs at 90+ for "crowd-favorite"', () => {
    const two = [score('a', 90), score('b', 90)]
    const three = [...two, score('c', 90)]

    expect(getBadges(two).find((badge) => badge.id === 'crowd-favorite')?.earned).toBe(false)
    expect(getBadges(three).find((badge) => badge.id === 'crowd-favorite')?.earned).toBe(true)
  })

  it('requires 5 distinct songs for "set-list" and 10 for "full-album"', () => {
    const five = ['a', 'b', 'c', 'd', 'e'].map((slug) => score(slug, 50))
    const ten = [...five, ...['f', 'g', 'h', 'i', 'j'].map((slug) => score(slug, 50))]

    expect(getBadges(five).find((badge) => badge.id === 'set-list')?.earned).toBe(true)
    expect(getBadges(five).find((badge) => badge.id === 'full-album')?.earned).toBe(false)
    expect(getBadges(ten).find((badge) => badge.id === 'full-album')?.earned).toBe(true)
  })

  it('sums play_count across all songs for "marathon-singer"', () => {
    const under = [score('a', 50, 20), score('b', 50, 4)]
    const over = [score('a', 50, 20), score('b', 50, 5)]

    expect(getBadges(under).find((badge) => badge.id === 'marathon-singer')?.earned).toBe(false)
    expect(getBadges(over).find((badge) => badge.id === 'marathon-singer')?.earned).toBe(true)
  })
})
