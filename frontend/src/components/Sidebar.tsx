import { useEffect } from 'react'
import { CloseIcon } from './icons'
import { BASE_URL, DEMO_MODE } from '../config'

interface SidebarProps {
  open: boolean
  onClose: () => void
  activeScreen: string | null
}

// Screens that need a live job server (loading new songs, uploading recordings, persisted
// scores) -- meaningless on the static demo build, so left out of its nav entirely.
const DEMO_ONLY_SCREENS = new Set(['load', 'recordings', 'highscores', 'voicerange'])

const ALL_NAV_LINKS: Array<{ href: string; screen: string | null; label: string }> = [
  { href: BASE_URL, screen: null, label: 'Play' },
  { href: `${BASE_URL}?screen=load`, screen: 'load', label: 'Load a song' },
  { href: `${BASE_URL}?screen=songs`, screen: 'songs', label: 'Cached songs' },
  { href: `${BASE_URL}?screen=recordings`, screen: 'recordings', label: 'My recordings' },
  { href: `${BASE_URL}?screen=highscores`, screen: 'highscores', label: 'High Scores' },
  { href: `${BASE_URL}?screen=calibrate`, screen: 'calibrate', label: 'Mic calibration' },
  { href: `${BASE_URL}?screen=voicerange`, screen: 'voicerange', label: 'Voice range' },
]

const NAV_LINKS = DEMO_MODE
  ? ALL_NAV_LINKS.filter((link) => !DEMO_ONLY_SCREENS.has(link.screen ?? ''))
  : ALL_NAV_LINKS

function Sidebar({ open, onClose, activeScreen }: SidebarProps) {
  useEffect(() => {
    if (!open) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKeyDown)

    // A modal drawer over the page shouldn't let the page behind it scroll.
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
    }
  }, [open, onClose])

  return (
    <>
      <div
        className={`sidebar-backdrop ${open ? 'sidebar-backdrop--open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`} aria-hidden={!open}>
        <div className="sidebar-header">
          <h2>Menu</h2>
          <button type="button" className="sidebar-close" onClick={onClose} aria-label="Close menu">
            <CloseIcon />
          </button>
        </div>
        <nav className="sidebar-nav">
          {NAV_LINKS.map((link) => {
            const isActive = link.screen === activeScreen
            return (
              <a
                key={link.href}
                className={`sidebar-nav-link${isActive ? ' sidebar-nav-link--active' : ''}`}
                href={link.href}
                aria-current={isActive ? 'page' : undefined}
              >
                {link.label}
              </a>
            )
          })}
        </nav>
      </aside>
    </>
  )
}

export default Sidebar
