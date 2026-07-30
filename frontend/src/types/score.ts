export interface ScoreRecord {
  slug: string
  title: string
  best_score: number | null
  best_achieved_at: string | null
  play_count: number
  last_played_at: string | null
}

/** Response shape of POST /api/scores -- a ScoreRecord plus the two fields
 * the end-of-song banner needs to render "New high score! (was X%)" without
 * a second request. */
export interface ScoreSubmissionResult extends ScoreRecord {
  is_new_best: boolean
  previous_best: number | null
}

export interface Badge {
  id: string
  label: string
  description: string
  earned: boolean
}
