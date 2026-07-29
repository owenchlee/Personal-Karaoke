# Personal Karaoke

A Rock Band/Yousician-style karaoke pitch game that runs entirely on your own machine. Point it
at a song (a local file or a link), and it separates vocals from instrumentals, extracts a
reference melody and lyrics, then plays them back as a scrolling note highway with live
microphone pitch scoring.

- **Python backend** — one-time, offline per-song processing: vocal/instrumental separation,
  melody extraction, and lyrics.
- **React/Vite frontend** — a scrolling note highway synced to the instrumental track, karaoke-
  style lyrics, live mic pitch detection, and a running/final score.
- **Local job server** — lets you load a new song straight from a link in the browser instead of
  running scripts by hand, plus a small library of already-processed songs.

See [`NOTES.md`](NOTES.md) for the full phase-by-phase design history and the reasoning behind
specific tuning decisions.

## How it works

1. **Extract audio** from a video/audio file (any container `ffmpeg` supports).
2. **Separate stems** with [Demucs](https://github.com/facebookresearch/demucs) into an isolated
   vocal track and an instrumental backing track.
3. **Extract melody** from the vocal stem with
   [basic-pitch](https://github.com/spotify/basic-pitch), refined against
   [`librosa.pyin`](https://librosa.org/) for pitch accuracy, monophony, and note timing, and
   silence-gated against the vocal stem's own RMS.
4. **Get lyrics**, preferring a synced-lyrics lookup from [lrclib.net](https://lrclib.net) by
   title, and falling back to local transcription via
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (word-level timestamps) when no
   match is found.
5. **Play**: the frontend renders the instrumental with a note highway, lyrics ticker, and (with
   mic access) live pitch scoring against the reference melody.

Everything after step 1 is deterministic and cached per song, so reprocessing the same input is
instant.

## Requirements

- **Python 3.11** specifically — `basic-pitch` depends on a TensorFlow version with no Python
  3.12 wheels on Windows/Linux (see `NOTES.md` for details).
- System **`ffmpeg`/`ffprobe`** on `PATH`.
- **Node.js** for the frontend (Vite + React + TypeScript).
- An NVIDIA/CUDA GPU is optional but automatically used for Demucs separation and Whisper
  transcription when available (`torch.cuda.is_available()`); everything falls back to CPU with
  no config changes needed.

## Backend setup

```bash
py -3.11 -m venv venv
venv/Scripts/python.exe -m pip install --upgrade pip
venv/Scripts/python.exe -m pip install torch torchaudio
venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verify the environment and run the test suite:

```bash
venv/Scripts/python.exe -m pytest -v
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev      # starts the Vite dev server
npm run test     # unit tests
```

## Running the app

The easiest way to use the app end-to-end is through the browser UI, which needs both the
frontend dev server and the local job server running side by side:

```bash
# Terminal 1
venv/Scripts/python.exe scripts/server.py

# Terminal 2
cd frontend
npm run dev
```

Open the printed local URL. From the sidebar menu you can:

- **Play** — the note highway game screen for the currently loaded song.
- **Load a song** — paste a link (e.g. a YouTube URL), optionally hint the language
  (English/Cantonese), and watch live progress as it downloads, separates, and extracts. A
  background job runs `download → separate → extract melody → transcribe/lookup lyrics → publish`,
  reporting progress the whole way; a full song typically takes 1.5–7 minutes depending on
  hardware.
- **Cached songs** — pick any already-processed song for an instant replay with no reprocessing.

On the game screen, click **Enable Mic** to turn on live pitch detection: your sung pitch is
matched against the reference melody (octave-tolerant — singing the right note in a different
register still counts), with a running score badge and a final score once the song ends.

### Processing a song from the command line

The same pipeline is also available without the job server, useful for scripting or debugging a
single stage:

```bash
venv/Scripts/python.exe scripts/process_song.py path/to/video.mp4 --cache-dir cache
venv/Scripts/python.exe scripts/publish_song.py <song-slug>
```

`process_song.py` extracts audio, separates stems, extracts the melody, and resolves lyrics,
caching everything under `cache/<song-slug>/`. Rerunning with the same input reuses the cache
(pass `--force` to reprocess). `publish_song.py` copies the finished assets into
`frontend/public/cache/<song-slug>/`, which Vite serves directly.

Each pipeline stage also has its own standalone CLI: `scripts/extract_audio.py`,
`scripts/separate_stems.py`, `scripts/extract_melody.py`, `scripts/extract_lyrics.py`.

## Project structure

```
audio_pipeline/   Core processing: separation, melody extraction, lyrics, download, pipeline glue
scripts/          CLI entry points + the FastAPI job server (server.py)
frontend/         React/Vite app: note highway, lyrics, scoring, song library, load-from-link UI
tests/            Python test suite (pytest)
frontend/src/**/*.test.ts   Frontend unit tests (vitest)
NOTES.md          Design history: decisions, measurements, and why things are tuned the way they are
```

## Testing

```bash
# Backend
venv/Scripts/python.exe -m pytest -v

# Frontend
cd frontend
npm run test
npx tsc --noEmit
```

## Known limitations

- Lyrics are either a licensed lookup match (when lrclib has one) or a best-effort local
  transcription — expect occasional errors, especially on melisma or fast passages.
- Melody extraction is tuned and validated against real songs but, like any automatic
  transcription, won't be perfect on every recording.
- This is a personal, local-first project — there's no deployment target or hosted instance.
