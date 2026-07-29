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
venv/Scripts/python.exe -m pip install torch torchaudio
venv/Scripts/python.exe -m pip install -r requirements.txt
```

Demucs separation and Whisper lyrics transcription automatically use an NVIDIA/CUDA GPU when one
is available (`torch.cuda.is_available()`, see `audio_pipeline/device.py`) and fall back to CPU
otherwise -- no config needed either way.

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

A lyrics ticker above the highway highlights the current/next word as the song plays, synced off
the same `<audio>` `currentTime`. Clicking "Enable Mic" turns on live pitch detection (via
`pitchy`): your sung pitch is drawn as a trailing blue line on the highway (folded to whichever
octave the current reference note is in, since scoring itself ignores octave -- singing the right
note in a different register still counts), a running score badge tracks accuracy as you go, and a
final score is shown once the song ends.

## Loading a song directly from a link

The game screen has a "Load a song from a link" field so you don't have to run the CLI scripts by
hand for a new song. It needs the local job server running alongside the frontend dev server:

```bash
venv/Scripts/python.exe scripts/server.py   # in a separate terminal, keep it running
```

Pasting a link and clicking Load calls `POST /api/jobs` (proxied by Vite from `/api` to
`http://127.0.0.1:8000`, configured in `frontend/vite.config.ts`), which downloads the audio via
`yt-dlp` and runs the same pipeline as `process_song.py` + `publish_song.py` in a background
thread, reporting progress while it runs (a real song takes 1.5-7 minutes -- see the pipeline
timings above). The frontend polls `GET /api/jobs/<id>` every 2s and loads the resulting song once
it's done.
