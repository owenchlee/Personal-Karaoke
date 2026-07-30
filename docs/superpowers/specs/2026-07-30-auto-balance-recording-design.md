# Auto-balance and clean up recorded takes

## Context

`useRecording.ts` records a player's take by mixing the instrumental (`<audio>` element) and the
live mic together, client-side, into a single `MediaStream`, then uploads the resulting webm blob
to `POST /api/recordings/mp3` (`scripts/server.py::render_recording_mp3`), which just transcodes
it to mp3 (`audio_pipeline/transcode.py::transcode_to_mp3`) with no other processing. Requested:
the singer's voice should come through loud enough and sound "professional" against the
background music, and the mic should get cleaned up if the recording quality is bad.

## Goal

Automatically balance vocal-vs-instrumental loudness and clean up the vocal recording, with no
user-facing control -- the existing "Enable Mic" / "Start Recording" / "Stop Recording" flow is
unchanged, and every resulting take is processed the same way.

## Non-goals

- No new UI (no toggle, no vocal-boost slider) -- decided during brainstorming.
- No change to live monitoring while singing (what plays through the speakers during recording is
  unaffected; only the saved take is processed).
- Not attempting music-source separation on the recorded mix -- the two sources are kept apart at
  the point they're already separate (mic stream vs. instrumental element), never merged and then
  pulled back apart.
- Not touching `list_recordings`/`download_recording`/`delete_recording` -- they operate on
  whatever mp3 already exists on disk, unaffected by how it was produced.

## Decision: server-side post-processing on two separately-recorded tracks

Considered doing this in real time via added Web Audio nodes (`GainNode`/`DynamicsCompressorNode`)
in the existing single-mix graph -- smaller change, but bounded quality (no true loudness
measurement, can't fix a bad setting after the fact) and out of step with how this codebase
already does every other quality-sensitive audio step (offline, ffmpeg/python, "measure, don't
guess" -- see NOTES.md throughout). Chose server-side instead: record the instrumental and mic as
two separate tracks (rather than one pre-mixed stream) and master them together server-side with
ffmpeg's proven loudness/dynamics tools once both are fully captured.

## Architecture / data flow

```
useRecording.ts (start)
  -> two MediaStreamAudioDestinationNodes instead of one:
       vocalDestination  <- mic source
       instrumentalDestination <- elementSource (already also -> context.destination for playback)
  -> two MediaRecorders, started back-to-back, stopped together
       (recording)
  -> two webm blobs: vocalBlob, instrumentalBlob

POST /api/recordings/mp3  (multipart form: vocal=<blob>, instrumental=<blob>, song_id=<id>)
  -> scripts/server.py: save both, transcode each webm -> wav
  -> audio_pipeline/mastering.py::master_recording(vocal_wav, instrumental_wav, output_dir)
       -> highpass vocal (~90Hz)
       -> denoise vocal (ffmpeg afftdn)
       -> compress vocal dynamics (ffmpeg acompressor)
       -> two-pass loudnorm: vocal to target A, instrumental to target B (A louder than B)
       -> mix (amix)
       -> final limiter (alimiter), guards against clipping from the loudness-boosted vocal
     -> mastered.wav
  -> transcode_to_mp3(mastered.wav, ...)   (existing, unchanged)
  -> saved under RECORDINGS_DIR, returned to the browser exactly as today
```

## Frontend changes

**`frontend/src/hooks/useRecording.ts`**
- `start()`: replace the single `mixDestination` with two
  `MediaStreamAudioDestinationNode`s -- `vocalDestination` (mic only) and
  `instrumentalDestination` (instrumental only, in addition to `elementSource`'s existing
  connection to `context.destination` for live playback). Two `MediaRecorder`s, created and
  `.start()`ed back-to-back (no `await` between the two calls) so they begin capturing at
  effectively the same time -- both stop on the same `stop()` call.
- `renderMp3` (processing step after stop): builds a `FormData` with both blobs
  (`vocal`, `instrumental`) plus the existing `song_id`, POSTs it to `/api/recordings/mp3`. Status/
  error states (`processing`/`done`/`error`) and the download-url/filename handling are unchanged.
- No other frontend files change -- `RecordingsLibrary.tsx`/`RecordingsScreen.tsx`/`recordings.ts`
  all operate on the already-produced mp3 and don't know how it was made.

## Backend changes

**New `audio_pipeline/mastering.py`**
- `master_recording(vocal_path: Path, instrumental_path: Path, output_dir: Path) -> Path`, mirroring
  this project's one-file-per-concern pattern (`separation.py`, `transcode.py`,
  `forced_alignment.py`, etc.).
- Vocal chain: highpass (~90Hz) -> `afftdn` denoise -> `acompressor` -> two-pass `loudnorm` to a
  vocal-forward target (starting point: vocal -14 LUFS, instrumental -20 LUFS -- a real recording
  needs a human listen to confirm this balance is right, same caveat this project attaches to every
  audio-quality claim; not hard-committed here).
- Instrumental chain: two-pass `loudnorm` to its own (quieter) target only -- it's already studio
  quality from the pipeline's own separation step, no denoise/compression needed.
- Mix via `amix`, then `alimiter` on the combined output so the vocal boost can't clip.
- Raises `RuntimeError` on any ffmpeg failure, same pattern as `transcode_to_mp3`.

**`scripts/server.py`**
- `render_recording_mp3` changes from reading a single raw request body to accepting two
  `UploadFile`s (`vocal`, `instrumental`) via multipart form, alongside the existing `song_id`
  field.
- After saving both to a temp dir and transcoding each to wav (reusing `transcode_to_mp3`'s ffmpeg
  pattern, or a small shared helper if that turns out cleaner), calls
  `mastering.master_recording(...)`, then `transcode_to_mp3` on the result -- same save-to-
  `RECORDINGS_DIR` / `FileResponse` tail end as today, unchanged.

## Error handling

If mastering or either transcode step fails, the endpoint raises the same
`HTTPException(status_code=500, ...)` pattern already in place -- `useRecording.ts`'s existing
`status: 'error'` / `errorMessage` path handles it with no new frontend error handling needed. No
silent fallback to an unmastered mix.

## Testing

- **`tests/test_mastering.py`** (new): real, unmocked smoke test (same convention as every other
  audio-pipeline test in this project) using synthetic tones -- a quiet+noisy synthetic "vocal"
  clip and a louder synthetic "instrumental" clip. Asserts: output file exists and is non-empty,
  output duration matches the input duration, the vocal-vs-instrumental RMS ratio moves in the
  louder-vocal direction relative to the raw inputs, and the mastered output's peak level doesn't
  clip (stays at or below the limiter's ceiling).
- **`tests/test_server.py`**: update the existing recording-endpoint test(s) for the new multipart
  (`vocal`/`instrumental` files) contract in place of the old single-body one.
- **Manual**: process a real sung take through the updated endpoint and listen -- same "needs a
  human ear" caveat this file gives every other real-audio verification (loudness targets above are
  a starting point, not a final answer, until checked this way).

## Open questions resolved during brainstorming

- Processing timing: server-side post-processing on two separately-recorded tracks, not real-time
  Web Audio nodes.
- User control: fully automatic, no new UI.
- Denoiser choice: ffmpeg's built-in `afftdn` (no extra model download), not `arnndn` (would need
  bundling an external RNNoise model) -- browser-side `noiseSuppression`/`echoCancellation`/
  `autoGainControl` are already on for the mic stream (`useMicPitch.ts`), so this is a second pass,
  not the only line of defense.
