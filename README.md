# SingScore

Turn any song into a Rock Band-style karaoke game, for songs that don't have one.

*Demo video coming soon — for now, clone it and run it locally (below) to try it yourself.*

<img src="https://img.shields.io/badge/React-19-149eca?logo=react&logoColor=white" alt="React 19"> <img src="https://img.shields.io/badge/TypeScript-3178c6?logo=typescript&logoColor=white" alt="TypeScript"> <img src="https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white" alt="Python 3.11"> <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"> <img src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">

## What it does

Games like Rock Band and Yousician only cover licensed setlists. SingScore builds that
experience for *any* song: give it a link, and it separates the vocals, figures out the melody
and lyrics on its own, then turns it into a playable game: a scrolling note highway, synced
lyrics, and live pitch scoring against your own microphone as you sing.

No manual charting, no pre-made track packs. The song itself is the only input.

## Disclaimer

This is a personal, educational project, not a hosted service. Processing runs locally on
your own machine: when you paste a link, SingScore does not upload, host, or redistribute
the audio anywhere. Any separated/cached files stay on your device for your own replay and
are never sent to or shared with anyone else. You are responsible for the content you choose
to process; only use links you have the rights to use.

## How it works

1. **Separate** vocals from the instrumental with [Demucs](https://github.com/facebookresearch/demucs) (source-separation ML model).
2. **Transcribe the melody** from the isolated vocal with [basic-pitch](https://github.com/spotify/basic-pitch), refined against [`librosa.pyin`](https://librosa.org/) for note-level pitch accuracy.
3. **Resolve lyrics** via a synced-lyrics lookup ([lrclib.net](https://lrclib.net)), falling back to local speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) when no match exists.
4. **Play**: the frontend renders the note highway and lyrics in sync with the instrumental, listens to the mic in real time, and scores pitch accuracy (octave-tolerant, so singing the right note in the wrong register still counts).

Everything after step 1 is cached per song, so replaying is instant and processing never repeats.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 19, TypeScript, Vite, Web Audio API for live pitch detection |
| Backend | Python, FastAPI job server, background pipeline with progress streaming |
| ML/Audio | PyTorch, Demucs (source separation), basic-pitch + librosa (melody), faster-whisper (transcription) |
| Testing | pytest (backend), Vitest (frontend) |

## Running it locally

**Requires:** Python 3.11, Node.js, and `ffmpeg` on your `PATH`.

```bash
git clone https://github.com/owenchlee/Personal-Karaoke.git
cd Personal-Karaoke

# 1. Start the backend (leave this terminal running)
py -3.11 -m venv venv
venv/Scripts/python.exe -m pip install torch torchaudio
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe scripts/server.py

# 2. In a second terminal, start the frontend
cd frontend
npm install
npm run dev
```

Open the URL the frontend prints (usually `http://localhost:5173`), paste a song link, and it
processes end-to-end (~1.5-7 min depending on hardware) then drops you into the game. An NVIDIA
GPU is auto-detected and used if present; otherwise everything falls back to CPU. Already-processed
songs replay instantly from a local cache.

See [`NOTES.md`](NOTES.md) for the full phase-by-phase design history and the reasoning behind
specific tuning decisions.

## Testing

```bash
venv/Scripts/python.exe -m pytest -v   # backend
cd frontend && npm run test             # frontend
```
