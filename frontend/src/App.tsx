import { useState } from 'react'
import GameScreen from './components/GameScreen'
import Sidebar from './components/Sidebar'
import ProofScreen from './screens/ProofScreen'
import LoadSongScreen from './screens/LoadSongScreen'
import CachedSongsScreen from './screens/CachedSongsScreen'

function MicWaveIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="2" />
      <path
        d="M5 11a7 7 0 0 0 14 0M12 18v4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 6h16M4 12h16M4 18h16"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const screen = new URLSearchParams(window.location.search).get('screen')
  const isProof = screen === 'proof'

  let content = <GameScreen />
  if (screen === 'proof') content = <ProofScreen />
  else if (screen === 'load') content = <LoadSongScreen />
  else if (screen === 'songs') content = <CachedSongsScreen />

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
            <MicWaveIcon />
            Personal Karaoke
          </a>
        </div>
        <a className="ghost-link" href={isProof ? '/' : '/?screen=proof'}>
          {isProof ? 'Back to game' : 'Diagnostics'}
        </a>
      </header>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      {content}
    </>
  )
}

export default App
