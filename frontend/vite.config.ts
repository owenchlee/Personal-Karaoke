import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // The demo build is served from https://<user>.github.io/Personal-Karaoke/, a subpath, so
  // asset URLs need that prefix baked in. The normal build stays root-relative since it's
  // always served from the Vite dev server / job server at origin root.
  base: mode === 'demo' ? '/Personal-Karaoke/' : '/',
  server: {
    // Allows Vite's dev server to accept requests carrying an unfamiliar Host
    // header -- needed when fronting this dev server with a tunnel (e.g.
    // cloudflared), whose public hostname (a random *.trycloudflare.com
    // subdomain) Vite would otherwise reject as a DNS-rebinding protection.
    // Fine for occasional personal use behind a tunnel; not meant for a
    // permanently public deployment.
    allowedHosts: true,
    proxy: {
      // Forwards to scripts/server.py (run separately: venv/Scripts/python.exe
      // scripts/server.py) so the browser can hit /api/* same-origin, no CORS
      // config needed for the normal dev workflow.
      '/api': 'http://127.0.0.1:8000',
    },
  },
}))
