import { useEffect } from 'react'

interface SidebarProps {
  open: boolean
  onClose: () => void
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M6 6l12 12M18 6L6 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

function Sidebar({ open, onClose }: SidebarProps) {
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
          <a className="sidebar-nav-link" href="/">
            Play
          </a>
          <a className="sidebar-nav-link" href="/?screen=load">
            Load a song
          </a>
          <a className="sidebar-nav-link" href="/?screen=songs">
            Cached songs
          </a>
        </nav>
      </aside>
    </>
  )
}

export default Sidebar
