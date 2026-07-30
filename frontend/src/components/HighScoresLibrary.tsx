import { useEffect, useState } from 'react'
import { getBadges, sortScoresByBest } from '../game/scores'
import type { ScoreRecord } from '../types/score'

function HighScoresLibrary() {
  const [scores, setScores] = useState<ScoreRecord[]>([])

  useEffect(() => {
    let cancelled = false

    fetch('/api/scores')
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load scores: ${response.status}`)
        return response.json() as Promise<{ scores: ScoreRecord[] }>
      })
      .then(({ scores: fetched }) => {
        if (!cancelled) setScores(fetched)
      })
      .catch(() => {
        if (!cancelled) setScores([])
      })

    return () => {
      cancelled = true
    }
  }, [])

  const badges = getBadges(scores)

  return (
    <div className="high-scores">
      <ul className="badge-list">
        {badges.map((badge) => (
          <li key={badge.id} className={`badge-row ${badge.earned ? 'badge-row--earned' : 'badge-row--locked'}`}>
            <span className="badge-label">{badge.label}</span>
            <span className="badge-description">{badge.description}</span>
          </li>
        ))}
      </ul>

      {scores.length === 0 ? (
        <p className="muted">No scores yet &ndash; play a song to get on the board.</p>
      ) : (
        <ul className="song-library-list">
          {sortScoresByBest(scores).map((entry) => (
            <li key={entry.slug} className="song-library-row">
              <div className="song-library-info">
                <span className="song-library-title">{entry.title}</span>
                <span className="song-library-date">
                  Best: {entry.best_score}% &middot; Played {entry.play_count}&times;
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default HighScoresLibrary
