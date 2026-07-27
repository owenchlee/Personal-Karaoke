# Phase 0 status & open decision

## Done

- `venv/` (Python 3.11 — see "Why Python 3.11" below) with pinned deps in `requirements.txt`.
- `scripts/check_env.py` + `tests/test_environment.py` — verifies every import (`ffmpeg-python`,
  `demucs`, `basic_pitch`, `librosa`, `soundfile`) and that the system `ffmpeg`/`ffprobe` binaries
  (not just the Python wrapper) are on PATH.
- `audio_pipeline/separation.py` — `separate_vocals(input_path, output_dir)` runs Demucs
  (`htdemucs` model, CPU) and saves the isolated vocal stem as a wav.
- `scripts/separate_vocals.py` — CLI: `venv/Scripts/python.exe scripts/separate_vocals.py <audio> --output-dir <dir>`.
  Prints elapsed separation time.
- `tests/test_separation.py` — smoke test proving the pipeline runs end-to-end on a synthetic
  sine-wave clip. This only proves the mechanics work, **not** that vocal isolation is musically
  accurate — a sine tone has no vocal content to isolate.

## Why Python 3.11, not 3.12

This machine only had Python 3.12 installed. `basic-pitch` depends on `tensorflow<2.15.1`, which
has **no wheels for Python 3.12** on Windows/Linux — a harder blocker than a simple version clash.
Installed Python 3.11.9 via `winget install Python.Python.3.11` and rebuilt `venv/` on it. On 3.11,
`tensorflow==2.15.0` and `torch==2.13.0+cpu` installed side by side with no conflict.

Note: `basic-pitch[onnx]` does **not** drop the TensorFlow dependency — on non-Darwin platforms
TensorFlow is a mandatory base dependency; the `[onnx]` extra only adds `onnxruntime` as an
additional backend option. This matters again in Phase 1 when basic-pitch is actually invoked.

## Open decision: Demucs vs. Spleeter fallback — NOT YET MADE

The spec requires benchmarking Demucs on this CPU and falling back to Spleeter if it's too slow.
This can't be decided from a synthetic clip (3s synthetic tone separated in ~4-9s tells us nothing
about a real 3-4 minute song). **No real song file is available yet.**

To make this decision once a sample song exists:

1. Drop a short (30s-3min) audio or video clip somewhere, e.g. `samples/test_song.mp4`.
2. If it's a video, extract audio first: `ffmpeg -i samples/test_song.mp4 -vn samples/test_song.wav`.
3. Run: `venv/Scripts/python.exe scripts/separate_vocals.py samples/test_song.wav --output-dir output`.
4. Note the printed elapsed time. Rule of thumb: if separation takes meaningfully longer than the
   song's own duration (e.g. >2-3x realtime), it's a poor fit for an interactive local tool and
   Spleeter (lighter, faster, lower quality) should be substituted.
5. Listen to `output/test_song_vocals.wav` — confirm vocals are actually audible and reasonably
   isolated from instruments. If quality is unacceptably poor, that's also a reason to reconsider.

Record the outcome (timing + quality + decision) here once done, before starting Phase 1.

## Outcome (recorded 2026-07-27)

Test source: audio extracted via `yt-dlp` from a YouTube video (id `nSDgHBxUbVQ`), saved as
`samples/test_song.wav` (274.2s, i.e. ~4m34s — a real full-length song, not a synthetic clip).
Neither the source file nor the separated output is committed (`samples/` and `output/` are
gitignored).

- **Timing**: `separate_vocals.py` completed in **79.9s** on this CPU — ~0.29x realtime (well
  under the 2-3x "poor fit" threshold from the spec).
- **Quality**: didn't do a literal listen (no audio playback available in this session).
  As a proxy, checked the output signal programmatically: vocal-stem RMS is ~45% of the original
  mix's RMS (Demucs is attenuating non-vocal energy, not passing the mix through unchanged), and
  ~21.5% of 1-second frames are near-silent — consistent with instrumental-only sections (intro/
  bridge/outro) where an actual vocal stem should go quiet. This is consistent with real
  separation happening, but a human listen is still recommended before fully trusting quality.
- **Decision**: **Demucs stays.** No need to fall back to Spleeter — separation speed is well
  within budget for an interactive local tool on this machine.

# Phase 1 design: melody extraction

## Scope

Given the isolated vocal stem from Phase 0, extract a reference vocal melody using
basic-pitch and save it as a MIDI file. No scoring against a user recording yet — that's a
later phase. This phase is extraction only.

## Design

- `audio_pipeline/melody_extraction.py::extract_melody(vocal_stem_path, output_dir) -> Path`
  - Calls `basic_pitch.inference.predict(audio_path)` on the vocal stem using its default
    ICASSP-2022 model and default thresholds. `melodia_trick=True` is basic-pitch's default
    and is well suited to monophonic vocal lines, so no threshold tuning up front.
  - Takes the `PrettyMIDI` object returned by `predict()` and writes it via `.write(...)` to
    `output_dir/<name>_melody.mid`.
  - Returns the path to the saved MIDI file.
- `scripts/extract_melody.py` — new standalone CLI mirroring `scripts/separate_vocals.py`:
  `venv/Scripts/python.exe scripts/extract_melody.py <vocal_stem.wav> --output-dir <dir>`.
  Takes an already-separated vocal stem as input (run `separate_vocals.py` first). Prints the
  saved path and elapsed time.
- `tests/test_melody_extraction.py` — smoke test mirroring `tests/test_separation.py`: a
  synthetic sine-wave clip in, assert a `.mid` file is produced and is non-empty/parseable via
  `pretty_midi`. Proves the pipeline runs end-to-end, not that the extracted melody is
  musically accurate (same caveat as the Phase 0 separation smoke test).

## Manual validation (recorded 2026-07-27)

Ran `scripts/process_song.py samples/test_song.wav --cache-dir cache` end-to-end (audio input,
not a video container -- no real video file was available this session, but ffmpeg's extraction
step is codec/container agnostic so this exercises the same code path as a video would).

- **Timing**: 88.4s for the full `extract_audio` -> `separate_stems` -> `extract_melody` chain on
  the 274.2s test song (comparable to the 79.9s Phase 0 vocals-only baseline; the extra
  instrumental-track subtraction and JSON serialization add negligible overhead).
- **Cache short-circuit**: confirmed -- an immediate rerun with identical arguments completed in
  0.0s and returned the same cached paths without reprocessing.
- **Note quality**: 670 notes extracted, covering 160.9s of the 274.2s song (~59%) -- plausible
  for a real song with instrumental intro/outro/bridge sections. 666 of 670 notes (99.6%) fall
  within a plausible sung-vocal MIDI range (36-84, i.e. C2-C6); only 3 notes are outliers below
  that range (MIDI 29-31, ~49Hz), likely residual low-frequency bleed rather than real vocal
  content. Didn't do a literal listen this session (no audio playback available) -- same caveat
  as the Phase 0 outcome above.
- **Frontend proof screen**: manually loaded in a real browser (Chrome via automation). Playback
  proof verified working for both the `<audio>` element path and the raw Web Audio API path (no
  console errors on either). Mic proof triggers a real OS/browser permission prompt that
  automation can't click through -- the code path (`getUserMedia` -> `AudioContext` ->
  `AnalyserNode` -> level meter) is implemented and reachable, but the "level bar reacts to
  speaking" check needs a human to click "Test Mic" and grant the permission manually.
- **Overall**: Phase 0 (frontend scaffold) and the revised Phase 1 (video-to-notes pipeline with
  caching) are demoable. Outstanding before fully closing this out: a human should (1) grant mic
  permission on the proof screen once to confirm the level meter reacts, and (2) do an actual
  audio listen-through of `cache/test-song/{instrumental,vocals}.wav` when convenient.
