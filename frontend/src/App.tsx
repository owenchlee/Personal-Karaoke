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
import AppearanceScreen from './screens/AppearanceScreen'
import { BASE_URL, DEMO_MODE } from './config'

// Screens that need a live job server -- meaningless on the static demo build. Sidebar already
// omits their nav links, but this also blocks reaching them directly via a typed/bookmarked
// `?screen=` URL, which would otherwise render a form that fires real fetches against a backend
// that doesn't exist on GitHub Pages.
const DEMO_ONLY_SCREENS = new Set(['load', 'recordings', 'highscores'])

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  let screen = new URLSearchParams(window.location.search).get('screen')
  if (DEMO_MODE && DEMO_ONLY_SCREENS.has(screen ?? '')) screen = null
  const isProof = screen === 'proof'

  let content = <GameScreen />
  if (screen === 'proof') content = <ProofScreen />
  else if (screen === 'load') content = <LoadSongScreen />
  else if (screen === 'songs') content = <CachedSongsScreen />
  else if (screen === 'calibrate') content = <CalibrationScreen />
  else if (screen === 'recordings') content = <RecordingsScreen />
  else if (screen === 'highscores') content = <HighScoresScreen />
  else if (screen === 'appearance') content = <AppearanceScreen />

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
          <a className="brand" href={BASE_URL}>
            <MicWaveIcon size={22} />
            SingScore
          </a>
        </div>
        <a className="ghost-link" href={isProof ? BASE_URL : `${BASE_URL}?screen=proof`}>
          {isProof ? 'Back to game' : 'Diagnostics'}
        </a>
      </header>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} activeScreen={screen} />
      {content}
    </>
  )
}

export default App
