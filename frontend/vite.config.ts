import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forwards to scripts/server.py (run separately: venv/Scripts/python.exe
      // scripts/server.py) so the browser can hit /api/* same-origin, no CORS
      // config needed for the normal dev workflow.
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
