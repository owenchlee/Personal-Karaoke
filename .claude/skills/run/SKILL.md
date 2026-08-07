---
name: run
description: Launch and drive the SingScore (Personal-Karaoke) app locally — starts the Python job server and Vite frontend dev server, then opens it in the browser. Use for "run the app", "open karaoke", "start singscore", "launch the site".
metadata:
  author: owen
  version: "1.0.0"
---

# Run SingScore locally

The app needs **two processes** running side by side (see README.md
"Running the app"):

1. **Job server** (Python/FastAPI) — handles song processing jobs,
   serves the cached song library. Port **8000**.
2. **Frontend dev server** (Vite) — the UI. Port **5173** (Vite will
   pick the next free port like 5174 if 5173 is taken).

## Steps

1. **Check what's already running** before starting duplicates:
   ```bash
   netstat -ano | grep -E ":(8000|517[3-9])" | grep LISTENING
   ```
   If port 8000 is already listening, the job server is up — don't
   start another one. Same for the 5173-5179 range and the frontend.

2. **Start the job server** (only if not already running), from the
   repo root, in the background:
   ```bash
   venv/Scripts/python.exe scripts/server.py
   ```

3. **Start the frontend dev server** (only if not already running):
   ```bash
   cd frontend && npm run dev
   ```
   Read its output to find the actual port it bound to (it prints
   `Local: http://localhost:XXXX/`).

4. **Open it**: navigate a browser tab to that local URL and take a
   screenshot to confirm the song list / UI renders (not a blank
   page or error overlay).

## Notes

- Both servers must be started with `run_in_background: true` (Bash
  tool) — they're long-running and block otherwise.
- The Python venv at `venv/` is already set up (Python 3.11, deps
  from `requirements.txt`) — no need to recreate it.
- If `scripts/server.py` fails to start, check `server_err.log` /
  `backend_err.log` in the repo root for prior failure output before
  debugging from scratch.
- This is a local-only app (job server needs a real machine for
  Demucs/Whisper/ffmpeg) — it is not the GitHub Pages demo build.
