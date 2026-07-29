import { describe, expect, it } from 'vitest'
import { getStageLabel } from './jobStages'

describe('getStageLabel', () => {
  it('returns a human-readable label for every known status', () => {
    expect(getStageLabel('downloading')).toBe('Downloading audio…')
    expect(getStageLabel('done')).toBe('Done')
    expect(getStageLabel('error')).toBe('Failed')
  })
})
