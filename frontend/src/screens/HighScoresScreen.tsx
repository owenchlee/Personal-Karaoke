import HighScoresLibrary from '../components/HighScoresLibrary'

function HighScoresScreen() {
  return (
    <main className="game-screen">
      <section className="panel">
        <h2>High Scores</h2>
        <HighScoresLibrary />
      </section>
    </main>
  )
}

export default HighScoresScreen
