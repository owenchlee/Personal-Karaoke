import type { Badge, ScoreRecord } from '../types/score'

const PERFECT_SCORE = 100
const HIGH_SCORE_THRESHOLD = 90
const HIGH_SCORE_SONG_COUNT = 3
const REPERTOIRE_SONG_COUNT = 5
const FULL_ALBUM_SONG_COUNT = 10
const MARATHON_PLAY_COUNT = 25

/** Best-first for the High Scores leaderboard. A null best_score (shouldn't
 * happen for anything GET /api/scores returns, since every scores/<slug>.json
 * entry has been played at least once, but handled defensively) sorts last. */
export function sortScoresByBest(scores: ScoreRecord[]): ScoreRecord[] {
  return [...scores].sort((a, b) => (b.best_score ?? -1) - (a.best_score ?? -1))
}

/** Looks up a single song's best score by slug, for SongLibrary's per-row
 * "Best: X%" badge. Returns null if the song has never been played. */
export function findBestScore(scores: ScoreRecord[], slug: string): number | null {
  return scores.find((entry) => entry.slug === slug)?.best_score ?? null
}

/**
 * Fixed set of achievements, derived purely from the aggregate
 * `{slug, best_score, play_count}[]` GET /api/scores returns -- no separate
 * backend badge storage/logic. Every scores/<slug>.json entry represents a
 * distinct song played at least once, so `scores.length` doubles as "number
 * of distinct songs played."
 */
export function getBadges(scores: ScoreRecord[]): Badge[] {
  const totalPlays = scores.reduce((sum, entry) => sum + entry.play_count, 0)
  const distinctSongs = scores.length
  const perfectScores = scores.filter((entry) => entry.best_score === PERFECT_SCORE).length
  const highScoreSongs = scores.filter((entry) => (entry.best_score ?? 0) >= HIGH_SCORE_THRESHOLD).length

  return [
    {
      id: 'first-note',
      label: 'First Note',
      description: 'Complete your first song',
      earned: distinctSongs >= 1,
    },
    {
      id: 'perfect-pitch',
      label: 'Perfect Pitch',
      description: 'Hit a perfect 100% score on any song',
      earned: perfectScores >= 1,
    },
    {
      id: 'crowd-favorite',
      label: 'Crowd Favorite',
      description: `Score ${HIGH_SCORE_THRESHOLD}%+ on ${HIGH_SCORE_SONG_COUNT} different songs`,
      earned: highScoreSongs >= HIGH_SCORE_SONG_COUNT,
    },
    {
      id: 'set-list',
      label: 'Set List',
      description: `Record a best score on ${REPERTOIRE_SONG_COUNT} different songs`,
      earned: distinctSongs >= REPERTOIRE_SONG_COUNT,
    },
    {
      id: 'full-album',
      label: 'Full Album',
      description: `Record a best score on ${FULL_ALBUM_SONG_COUNT} different songs`,
      earned: distinctSongs >= FULL_ALBUM_SONG_COUNT,
    },
    {
      id: 'marathon-singer',
      label: 'Marathon Singer',
      description: `Play songs ${MARATHON_PLAY_COUNT} times total, across all songs`,
      earned: totalPlays >= MARATHON_PLAY_COUNT,
    },
  ]
}
