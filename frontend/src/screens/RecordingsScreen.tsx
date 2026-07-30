import RecordingsLibrary from '../components/RecordingsLibrary'

function RecordingsScreen() {
  return (
    <main className="game-screen">
      <section className="panel">
        <h2>My recordings</h2>
        <RecordingsLibrary />
      </section>
    </main>
  )
}

export default RecordingsScreen
