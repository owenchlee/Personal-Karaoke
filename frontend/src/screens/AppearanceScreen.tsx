import { useState } from 'react'
import { ACCENT_PRESETS, applyAccent, clearAccent, getStoredAccent, saveAccent } from '../game/theme'

const DEFAULT_ACCENT = ACCENT_PRESETS[0].value

function AppearanceScreen() {
  const [accent, setAccent] = useState<string>(() => getStoredAccent() ?? DEFAULT_ACCENT)

  const choose = (hex: string) => {
    setAccent(hex)
    saveAccent(hex)
    applyAccent(hex)
  }

  const reset = () => {
    setAccent(DEFAULT_ACCENT)
    clearAccent()
    applyAccent(null)
  }

  const isPreset = ACCENT_PRESETS.some((preset) => preset.value === accent)

  return (
    <main className="game-screen">
      <section className="panel">
        <h2>Appearance</h2>
        <p className="muted">
          Pick an accent color &mdash; it colors the mic icon, active menu item, highlighted
          lyrics, and the note-highway playhead, in both light and dark mode.
        </p>
        <div className="accent-swatches" role="group" aria-label="Accent color">
          {ACCENT_PRESETS.map((preset) => (
            <button
              key={preset.value}
              type="button"
              className={`accent-swatch${accent === preset.value ? ' accent-swatch--active' : ''}`}
              style={{ backgroundColor: preset.value }}
              onClick={() => choose(preset.value)}
              aria-label={preset.name}
              aria-pressed={accent === preset.value}
              title={preset.name}
            />
          ))}
          <label
            className={`accent-swatch accent-swatch--custom${!isPreset ? ' accent-swatch--active' : ''}`}
            style={{ backgroundColor: accent }}
            title="Custom color"
          >
            <input
              type="color"
              value={accent}
              onChange={(event) => choose(event.target.value)}
              aria-label="Custom accent color"
            />
          </label>
        </div>
        <div className="button-row accent-reset-row">
          <button type="button" className="btn btn-secondary" onClick={reset}>
            Reset to default
          </button>
        </div>
      </section>
    </main>
  )
}

export default AppearanceScreen
