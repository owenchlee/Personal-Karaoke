import { useEffect } from 'react'
import { CloseIcon } from './icons'

interface SidebarProps {
  open: boolean
  onClose: () => void
  activeScreen: string | null
}

const NAV_LINKS: Array<{ href: string; screen: string | null; label: string }> = [
  { href: '/', screen: null, label: 'Play' },
  { href: '/?screen=load', screen: 'load', label: 'Load a song' },
  { href: '/?screen=songs', screen: 'songs', label: 'Cached songs' },
  { href: '/?screen=recordings', screen: 'recordings', label: 'My recordings' },
  { href: '/?screen=highscores', screen: 'highscores', label: 'High Scores' },
  { href: '/?screen=calibrate', screen: 'calibrate', label: 'Mic calibration' },
]

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
