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
- **Frontend proof screen**: manually loaded in a real browser (Chrome via automation). Correction
  to an earlier version of this entry: "no console errors" was wrongly reported as "playback
  verified working" -- it only proves no exception was thrown, not that audio was actually
  audible. Phase 2's investigation (below) discovered the automated browser tab used by this
  tooling never becomes `document.visibilityState === "visible"` even when clicked/focused, and
  Chrome suspends `<audio>`/media-element resource loading in non-visible tabs -- so audio
  playback can't be reliably verified through this automation at all, in either phase. Mic proof
  also triggers a real OS/browser permission prompt automation can't click through.
- **Overall**: Phase 0 (frontend scaffold) and the revised Phase 1 (video-to-notes pipeline with
  caching) are demoable. Outstanding before fully closing this out: a human should, in a normal
  (non-automated) browser window, (1) grant mic permission on the proof screen to confirm the
  level meter reacts, (2) confirm the playback proof buttons produce audible sound, and (3) do an
  actual audio listen-through of `cache/test-song/{instrumental,vocals}.wav` when convenient.

## Phase 2 design + manual validation (recorded 2026-07-27)

Built the note highway game screen: `frontend/src/components/GameScreen.tsx` fetches
`/cache/<slug>/notes.json` and owns an `<audio src="/cache/<slug>/instrumental.wav" controls>`;
`frontend/src/components/NoteHighway.tsx` is a canvas element with a `requestAnimationFrame` loop
that reads `audioRef.current.currentTime` directly every frame (no separate clock) to draw a
right-to-left scrolling highway (vertical = pitch via `pitchToY`, horizontal bar = duration via
`timeToX`, both in `frontend/src/game/coords.ts`). New `scripts/publish_song.py` copies
`notes.json`+`instrumental.wav` from `cache/<slug>/` into `frontend/public/cache/<slug>/` for
Vite to serve (already-ignored by the existing `cache/` gitignore rule at any depth -- verified
via `git check-ignore`). Old Phase 0 proof screen moved to `frontend/src/screens/ProofScreen.tsx`,
reachable via `?screen=proof`; the note highway is now the default screen.

- **Automated**: 12 new `vitest` unit tests for the pure coordinate-mapping functions
  (`getPitchRange`, `pitchToY`, `timeToX`, `getVisibleNotes`) all pass; `tsc --noEmit` clean;
  Python test suite (8 tests) still passes unaffected.
- **Manual -- visual sync confirmed working**: loaded the game screen in a real browser (Chrome
  via automation) with the real `test-song` data (670 notes). The highway renders correctly at
  `currentTime = 0`: playhead line visible, upcoming notes positioned to its right at plausible
  heights, matching what the unit-tested coordinate functions predict for `t=0`.
- **Manual -- audio playback verification blocked by the automation environment, not a code bug**:
  pressing play never advanced the highway. Root-caused via `systematic-debugging`, not guessed:
  - A raw `fetch()` (including byte-range requests matching what a media element sends) against
    `/cache/test-song/instrumental.wav` returns instantly with correct headers
    (`content-type: audio/wav`, `accept-ranges: bytes`, proper `content-range`/`content-length`,
    streams real bytes) -- Vite's static serving and `publish_song.py`'s copy are both correct.
  - The `<audio>` element itself, and even a bare `new Audio(url)` with no React/app code
    involved, stay stuck at `readyState: 0` / `networkState: 2` (loading, never progressing),
    with no `error` event and no `loadedmetadata` event, even after an explicit `.load()` call.
  - `document.visibilityState` reports `"hidden"` for this tab and stays `"hidden"` even after
    clicking into the page (`document.hasFocus()` becomes `true`, but `visibilityState` does not
    change) -- Chrome deprioritizes/suspends media-element resource loading in non-visible tabs,
    independent of any application code.
  - Conclusion: this is a genuine limitation of verifying `<audio>` playback through this specific
    browser-automation tooling, not a bug in `GameScreen`/`NoteHighway`/`publish_song.py`. A human
    should open `http://localhost:5175` (or whatever port `npm run dev` picks) in their own normal
    browser window and confirm: (1) the instrumental is audible on play, (2) dragging the native
    seek bar resyncs the highway instantly with no glitch (the concrete proof the "no separate
    sync logic" design works), (3) the pitch auto-fit looks reasonable for this song's real
    MIDI 36-84 vocal range.
- **Overall**: Phase 2's code is complete and the visual/data half is verified; the audio-sync
  half needs one manual human check (above) before calling Phase 2 fully closed.

## Melody extraction quality fix (recorded 2026-07-27)

**Problem reported**: the note data included notes that don't correspond to what's actually sung,
and note lengths didn't match how long a note is actually held. Also raised: music videos often
have background noise unrelated to the song (crowd noise, talk, etc.) that should be kept out of
the extracted notes.

**Root cause (measured, not guessed)**: inspected `cache/test-song/notes.json` (670 notes,
basic-pitch defaults). 51% of notes were under 200ms, many sitting right at basic-pitch's ~128ms
minimum-length floor, with 204 pairs of overlapping notes. This is the signature of a single held
(often vibrato-inflected) note getting chopped into several short fragments: `onset_threshold=0.5`
lets brief mid-note energy blips register as spurious new onsets, and `melodia_trick=True` (on by
default) actively inserts extra notes to bridge pitch-bend segments.

**Fix**: tuned `basic_pitch.inference.predict()` in `audio_pipeline/melody_extraction.py`:
`onset_threshold=0.65` (fewer spurious re-onsets mid-note), `frame_threshold=0.25` (lets a real
note bridge brief energy dips instead of dropping out), `melodia_trick=False` (stop inventing
bridge notes), `minimum_note_length=150` ms, and `minimum_frequency=65`/`maximum_frequency=1300`
Hz (bounds extraction to a plausible sung-vocal range, rejecting low-frequency rumble/bleed --
NOTES.md's earlier Phase 1 entry flagged 3 such outlier notes at ~49Hz).

Verified directly against the cached `test-song` vocal stem (no need to rerun Demucs -- tested
several parameter combinations with plain `predict()` calls before touching code): note count
670 -> 378, notes under 200ms 51% -> 25%, overlapping-note pairs 204 -> 0 same-pitch duplicates
(76 remaining overlaps are all different-pitch, ~10-100ms overlaps at natural pitch-transition/
portamento boundaries between two real notes, not noise -- confirmed by inspecting example pairs),
median note duration 0.197s -> 0.279s.

Re-ran `scripts/extract_melody.py` on the cached vocal stem and `scripts/publish_song.py` to
refresh the frontend's copy. Pitch range across the song is now MIDI 45-75, tighter than the
previous 29-94 (the low-frequency outliers are gone).

**Test fixture fix**: the existing `tests/test_melody_extraction.py` smoke test used a pure sine
wave, which has no harmonics -- out-of-distribution for basic-pitch's model (trained on real
instrument/vocal timbres) and already produced fragmented, low-confidence, wrongly-pitched
detections even under the old defaults. The stricter thresholds pushed its already-marginal
detections to zero notes, surfacing this pre-existing fixture weakness. Fixed by synthesizing a
harmonic-rich tone (fundamental + 4 decaying overtones + an attack/decay envelope) instead --
under the new tuned settings this now produces exactly one correctly-pitched note spanning
the full clip, a much better proxy for real vocal audio.

**Background noise (crowd noise, talk, etc. unrelated to the song)**: not separately validated,
since the only available test video does not contain this kind of noise (checked: the separated
`vocals.wav`'s first/last 15s are already near-silent, RMS <= 0.008, so Demucs isn't leaking
non-song noise into the vocal stem for this particular source). The fix above should help in
general -- background noise/chatter typically produces lower and less temporally-stable pitch
confidence than sustained singing, so the raised `onset_threshold`/`frame_threshold` and the
vocal-frequency bounds should suppress much of it -- but this is reasoning from how basic-pitch's
confidence scoring works, not something measured against a real noisy example. If a future video
still shows spurious notes from background noise, that should be treated as a fresh investigation
with real evidence (which section of audio, what the vocal stem sounds like there) rather than
more speculative threshold tuning.
