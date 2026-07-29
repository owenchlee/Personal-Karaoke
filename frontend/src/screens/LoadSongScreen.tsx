import LoadSongForm from '../components/LoadSongForm'

function LoadSongScreen() {
  const handleLoaded = (slug: string) => {
    // A hard navigation (not client-side state) since this takes us to a different page --
    // GameScreen picks the new song straight up from the `?song=` param on mount.
    window.location.href = `/?song=${encodeURIComponent(slug)}`
  }

  return (
    <main className="game-screen">
      <section className="panel">
        <h2>Load a song</h2>
        <LoadSongForm onLoaded={handleLoaded} />
      </section>
    </main>
  )
}

export default LoadSongScreen
