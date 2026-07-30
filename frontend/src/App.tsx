import { useState } from 'react'
import GameScreen from './components/GameScreen'
import Sidebar from './components/Sidebar'
import { MicWaveIcon, MenuIcon } from './components/icons'
import ProofScreen from './screens/ProofScreen'
import LoadSongScreen from './screens/LoadSongScreen'
import CachedSongsScreen from './screens/CachedSongsScreen'
import CalibrationScreen from './screens/CalibrationScreen'
import RecordingsScreen from './screens/RecordingsScreen'
import HighScoresScreen from './screens/HighScoresScreen'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const screen = new URLSearchParams(window.location.search).get('screen')
  const isProof = screen === 'proof'

  let content = <GameScreen />
  if (screen === 'proof') content = <ProofScreen />
  else if (screen === 'load') content = <LoadSongScreen />
  else if (screen === 'songs') content = <CachedSongsScreen />
  else if (screen === 'calibrate') content = <CalibrationScreen />
  else if (screen === 'recordings') content = <RecordingsScreen />
  else if (screen === 'highscores') content = <HighScoresScreen />

  return (
    <>
      <header className="site-header">
        <div className="site-header-left">
          <button
            type="button"
            className="menu-button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <MenuIcon />
          </button>
          <a className="brand" href="/">
            <MicWaveIcon size={22} />
            Personal Karaoke
          </a>
        </div>
        <a className="ghost-link" href={isProof ? '/' : '/?screen=proof'}>
          {isProof ? 'Back to game' : 'Diagnostics'}
        </a>
      </header>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} activeScreen={screen} />
      {content}
    </>
  )
}

export default App
