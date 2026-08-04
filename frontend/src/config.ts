/** True when built with `--mode demo` (see `.env.demo`, `npm run build:demo`): a static-hosting
 * build with no job server behind it, used for the public showcase deploy. Screens and controls
 * that require a live backend (loading new songs, recordings, star/delete, score persistence)
 * are hidden rather than left to fail against a nonexistent API. */
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true'

/** Where published song assets (notes.json, lyrics.json, instrumental.wav, manifest.json) are
 * served from. The demo build reads from public/demo-cache -- a small, git-tracked set of songs
 * cleared for public redistribution -- rather than public/cache, which is .gitignored precisely
 * because it accumulates whatever a local user has processed (frequently copyrighted tracks not
 * safe to publish). See scripts/generate_demo_manifest.py. */
export const CACHE_BASE = DEMO_MODE ? '/demo-cache' : '/cache'
