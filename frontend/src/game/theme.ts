// Accent color customization -- a single CSS custom property drives every accent-tinted surface
// (brand icon, active nav link, lyric highlight sweep, note-highway playhead, focus rings; see
// index.css, where --accent-bg/--accent-border are both derived from --accent via color-mix()
// rather than stored separately, so overriding this one value is enough to re-theme everything).
// Stored as a hex string so it round-trips cleanly through both localStorage and an
// <input type="color">.

const STORAGE_KEY = 'karaoke:accentColor'

export interface AccentPreset {
  name: string
  value: string
}

export const ACCENT_PRESETS: AccentPreset[] = [
  { name: 'Rose', value: '#db2777' },
  { name: 'Amber', value: '#d97706' },
  { name: 'Teal', value: '#0d9488' },
  { name: 'Sky', value: '#0284c7' },
  { name: 'Violet', value: '#7c3aed' },
  { name: 'Green', value: '#16a34a' },
]

/** The saved custom accent, or null if the user has never overridden the built-in default. */
export function getStoredAccent(): string | null {
  if (typeof window === 'undefined' || !window.localStorage) return null
  return window.localStorage.getItem(STORAGE_KEY)
}

export function saveAccent(hex: string): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.setItem(STORAGE_KEY, hex)
}

export function clearAccent(): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  window.localStorage.removeItem(STORAGE_KEY)
}

/** Applies (or clears) the accent override on the document root. An inline custom property on the
 * root element beats both the light and dark stylesheet rules regardless of the dark-mode media
 * query, which is what lets one stored color apply consistently across both themes instead of
 * needing a separate light/dark pair picked by hand. */
export function applyAccent(hex: string | null): void {
  if (typeof document === 'undefined') return
  if (hex) {
    document.documentElement.style.setProperty('--accent', hex)
  } else {
    document.documentElement.style.removeProperty('--accent')
  }
}

/** Call once at startup, before first paint if possible, to restore any saved override. */
export function initAccent(): void {
  applyAccent(getStoredAccent())
}
