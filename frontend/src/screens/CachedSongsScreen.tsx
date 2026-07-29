import SongLibrary from '../components/SongLibrary'

function CachedSongsScreen() {
  const handleSelect = (slug: string) => {
    // A hard navigation (not client-side state) since this takes us to a different page --
    // GameScreen picks the new song straight up from the `?song=` param on mount.
    window.location.href = `/?song=${encodeURIComponent(slug)}`
  }

  return (
    <main className="game-screen">
      <section className="panel">
        <h2>Cached songs</h2>
        <SongLibrary onSelect={handleSelect} />
      </section>
    </main>
  )
}

export default CachedSongsScreen
