import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { ACCENT_PRESETS, clearAccent, getStoredAccent, saveAccent } from './theme'

describe('accent storage', () => {
  // Pure `game/*.ts` modules in this project are deliberately tested without a DOM environment
  // (see calibration.test.ts) -- a minimal in-memory `localStorage` stub keeps that convention
  // rather than pulling in jsdom for the whole suite just for this one module.
  class MemoryStorage {
    private store = new Map<string, string>()
    getItem(key: string) {
      return this.store.has(key) ? this.store.get(key)! : null
    }
    setItem(key: string, value: string) {
      this.store.set(key, value)
    }
    removeItem(key: string) {
      this.store.delete(key)
    }
  }

  beforeAll(() => {
    ;(globalThis as Record<string, unknown>).window = { localStorage: new MemoryStorage() }
  })
  afterAll(() => {
    delete (globalThis as Record<string, unknown>).window
  })
  afterEach(() => {
    clearAccent()
  })

  it('defaults to null when nothing has been saved', () => {
    expect(getStoredAccent()).toBeNull()
  })

  it('round-trips a saved value', () => {
    saveAccent('#0d9488')
    expect(getStoredAccent()).toBe('#0d9488')
  })

  it('clears back to null', () => {
    saveAccent('#0d9488')
    clearAccent()
    expect(getStoredAccent()).toBeNull()
  })
})

describe('ACCENT_PRESETS', () => {
  it('is non-empty and every value is a hex color', () => {
    expect(ACCENT_PRESETS.length).toBeGreaterThan(0)
    for (const preset of ACCENT_PRESETS) {
      expect(preset.value).toMatch(/^#[0-9a-f]{6}$/i)
    }
  })

  it('has no duplicate names or values', () => {
    expect(new Set(ACCENT_PRESETS.map((p) => p.name)).size).toBe(ACCENT_PRESETS.length)
    expect(new Set(ACCENT_PRESETS.map((p) => p.value)).size).toBe(ACCENT_PRESETS.length)
  })
})
