# Personal-Karaoke

A Rock Band/Yousician-style live karaoke pitch game. A Python backend does one-time offline
per-song processing (vocal/instrumental separation, note extraction); a browser frontend plays a
scrolling note highway synced to the instrumental track, with live client-side mic pitch
detection. See `NOTES.md` for phase-by-phase status and design decisions.

## Backend setup

Requires **Python 3.11** specifically (see `NOTES.md` for why) and system `ffmpeg`/`ffprobe` on PATH.

```bash
py -3.11 -m venv venv
venv/Scripts/python.exe -m pip install --upgrade pip
venv/Scripts/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verify the environment:

```bash
venv/Scripts/python.exe -m pytest -v
```

## Usage: process a song end-to-end

```bash
venv/Scripts/python.exe scripts/process_song.py path/to/video.mp4 --cache-dir cache
```

Extracts audio, separates vocal/instrumental stems, extracts a reference melody, and transcribes
lyrics (with word-level timestamps), caching the results under `cache/<song-slug>/`. Rerunning on
the same input reuses the cache; pass `--force` to reprocess. See `cache/<song-slug>/notes.json`
for the note-event data, `cache/<song-slug>/lyrics.json` for the transcribed lyrics, and
`cache/<song-slug>/instrumental.wav` for the backing track.

Lyrics are transcribed locally from the isolated vocal stem via `faster-whisper` (no external
lyrics source -- see `NOTES.md`), so they're a best-effort transcript, not verified-accurate
lyrics; expect some errors, especially on melisma/fast passages.

Each pipeline stage is also available standalone: `scripts/extract_audio.py`,
`scripts/separate_stems.py`, `scripts/extract_melody.py`, `scripts/extract_lyrics.py`.

To make a processed song playable in the frontend, publish its assets:

```bash
venv/Scripts/python.exe scripts/publish_song.py <song-slug>
```

Copies `notes.json` + `lyrics.json` + `instrumental.wav` from `cache/<song-slug>/` into
`frontend/public/cache/<song-slug>/`, which Vite serves directly.

## Frontend

```bash
cd frontend
npm install
npm run dev
npm run test   # unit tests for the note-highway coordinate math
```

Two screens, switched via a URL query param (no routing library):

- `/` (default) -- the note highway game screen. Loads `?song=<slug>` (default `test-song`) from
  `frontend/public/cache/<slug>/`, published via `publish_song.py` above.
- `/?screen=proof` -- the Phase 0 proof screen (confirms mic access and audio playback work in
  the browser).

No mic input or scoring yet -- Phase 2 only proves the note highway stays visually synced to the
instrumental track's own playback position. A lyrics ticker above the highway highlights the
current/next word as the song plays, synced off the same `<audio>` `currentTime`.
