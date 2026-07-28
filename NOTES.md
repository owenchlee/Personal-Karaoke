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
confidence than sustained singing, so the vocal-frequency bounds should suppress some of it -- but
this is reasoning from how basic-pitch's confidence scoring works, not something measured against
a real noisy example. If a future video still shows spurious notes from background noise, that
should be treated as a fresh investigation with real evidence (which section of audio, what the
vocal stem sounds like there) rather than more speculative threshold tuning.

### Correction (recorded 2026-07-27): onset_threshold=0.65 was an overcorrection

After the fix above shipped, manual play-through surfaced a new symptom: real singing with no note
bar at all in places. Investigated by comparing note coverage directly against the vocal stem's
own energy envelope (0.25s RMS buckets, `has_energy` = RMS above the 40th percentile of non-silent
buckets) rather than guessing from note-count stats alone -- fragmentation metrics alone can't
detect missed notes, only extra/short ones.

Result: raising `onset_threshold` to 0.65 pushed real, legitimately-confident onsets below the bar
along with the spurious ones. Coverage got *worse* than the pre-fix baseline: 15% of clearly-sung
0.25s buckets had no covering note at onset=0.65, vs. 8% at basic-pitch's own default onset=0.5.
Confirmed concretely on the 218-223s window (audible singing, RMS up to 0.31): baseline found 13
overlapping onset candidates there; onset=0.65 kept only 5, leaving three multi-second silent gaps
in the highway during real singing.

Swept `onset_threshold` x `frame_threshold` against both the fragmentation metrics (note count,
%under 200ms, median duration) and the new energy-coverage metric to find a setting that doesn't
trade one failure mode for the other. `minimum_frequency`/`maximum_frequency` were confirmed to
have negligible effect on coverage (51 vs 51 missed buckets with/without), so those stay.

**Corrected settings**: `onset_threshold=0.5` (basic-pitch's own default -- back off from 0.65),
`frame_threshold=0.2` (down from 0.25, lets real notes bridge slightly deeper energy dips),
`melodia_trick=False` and the frequency bounds unchanged. On the test song: 622 notes (vs. the
original 670 and the overcorrected 378), 31% under 200ms (vs. original 51%, overcorrected 25%),
median duration 0.245s (vs. original 0.197s), missed-energy coverage 8% -- same as the original
untuned baseline, not worse. This is a real trade-off, not a free win: still meaningfully less
fragmented than the original defaults, at the cost of accepting somewhat more short notes and
diff-pitch overlaps than the (coverage-broken) 0.65 attempt had.

### Second correction (recorded 2026-07-27): silence-gate notes against the vocal stem's own RMS

Next symptom reported: the instrumental-only intro still showed note bars with no singing. Root
cause, confirmed by checking each early note's onset against the vocal stem's actual RMS at that
timestamp: basic-pitch's own confidence score ("velocity") is not a reliable silence detector. On
this song's instrumental intro the vocal stem is genuinely near-total silence (RMS ~0.0000-0.0001,
~1000x below the median non-silent RMS of ~0.096), yet basic-pitch still emitted notes there with
moderate reported velocity (0.27-0.55) -- an artifact of how its front end normalizes near-zero
input, not evidence of real signal.

Fix: added `_note_rms()` in `melody_extraction.py`, which loads the vocal stem directly (via
`soundfile`, same library used elsewhere in this pipeline) and computes each note's actual RMS
over its own `[onset, offset]` window, independent of anything basic-pitch reports. Notes below
`_SILENCE_RMS_GATE = 0.01` are dropped before the JSON/MIDI are written. The MIDI file is now
rebuilt from the filtered note list directly (previously written straight from basic-pitch's
`midi_data`) so the two output artifacts stay consistent.

Checked this threshold isn't a fragile magic number: computed per-note RMS for all 622 notes from
the prior correction and found a stark, clean bimodal split with no notes anywhere in between --
24 notes measured ~0.00004 RMS (the spurious ones), then every other note measured >= 0.05. Any
gate value between ~0.001 and ~0.04 drops exactly the same 24 notes and keeps the other 598; 0.01
sits comfortably in that gap. Re-ran on the test song: 622 -> 598 notes, first note onset moved
from 1.93s to 18.13s (matching the song's real instrumental intro before singing starts).

Same caveat as before applies to future songs: this gate is only as good as Demucs's separation
quality plus the assumption that non-silence in the vocal stem means singing. A song with loud
sustained non-vocal bleed in its vocal stem (rather than near-silence) wouldn't be caught by this
gate -- that would need fresh evidence, not more speculative tuning, if it comes up.

### Third correction (recorded 2026-07-27): enforce monophony -- no simultaneous note bars

Next symptom: the highway sometimes showed two note bars at once ("parallel bars"), implying two
simultaneous voices, which is wrong for a solo vocal line. This is the same overlap issue flagged
earlier as "natural pitch-transition artifacts" and left alone -- that call was wrong once it was
actually visible in the game; overlaps are never correct for monophonic vocals and needed a real
fix, not just an explanation.

Root cause: basic-pitch is a polyphonic transcription model (its main use case is piano, which
really can have simultaneous notes); it doesn't know a solo vocal line can't. Checked the pitch
relationship of all 221 overlapping pairs in the (598-note, post-silence-gate) extraction: 58%
were exactly one octave (12 semitones) apart -- the textbook signature of the model detecting a
strong 2nd harmonic as if it were its own note alongside the true fundamental.

Fix: added `_enforce_monophony()` in `melody_extraction.py`, a strict post-process that walks
notes in onset order and never lets two stay overlapping:
- If a note's entire span sits inside the currently-kept note's span, drop it. Checked this is a
  safe rule before relying on it: of the 77 fully-contained overlap pairs in the data, the
  containing note was longer in 100% of cases (true by construction -- a shorter span can't fully
  contain a longer one), so "keep the longer, drop the nested one" isn't a coin flip, it's
  definitional.
- For partial overlaps (a real melodic transition where two detections briefly disagree), keep
  whichever note has higher basic-pitch confidence ("velocity") and trim the other's boundary back
  to remove the overlap, dropping it outright if trimming would leave a sliver under 50ms.
- Resolves overlap chains (A overlaps B overlaps C) correctly via a small while-loop, not just a
  single pairwise comparison -- verified with a dedicated unit test
  (`test_enforce_monophony_handles_a_three_note_overlap_chain`).

Re-ran on the test song: 598 -> 450 notes, overlaps 221 -> 0 (confirmed by direct check, not
sampled). Added 3 unit tests in `tests/test_melody_extraction.py` covering full containment,
partial-overlap trimming, and a 3-note overlap chain, plus a no-overlap assertion in the existing
end-to-end smoke test. 11/11 Python tests pass.

## Lyrics feature (recorded 2026-07-27)

New request: show lyrics above the note highway (highway moved down to make room), with the
current/next word highlighted and the display moving as the song progresses.

**Lyrics source decision**: no lyrics data existed anywhere in the pipeline before this. Asked the
user how to source lyrics text + word timing; ruled out fetching lyrics from any external
site/API myself (would mean reproducing copyrighted lyrics text, which isn't something I can do
regardless of personal-use framing). Chosen approach: transcribe locally from the vocal stem using
`faster-whisper` (model size `small`, CPU, `compute_type="int8"`, `word_timestamps=True`) -- a
transcript of audio the user already has, not lyrics sourced from an external database. Benchmark
on the 274s test song: 36s transcription time (~7.6x realtime), well within budget; produced
recognizable, mostly-correct lyrics (English, language-detection confidence 0.90) with per-word
timestamps and per-word confidence scores.

**Backend**: new `audio_pipeline/lyrics_extraction.py::extract_lyrics()`, wired into
`pipeline.py` as a fourth stage alongside audio extraction/separation/melody, producing
`cache/<slug>/lyrics.json` (`[{"word", "start", "end"}, ...]`). `SongAssets` and the cache
short-circuit check (`_cached_assets`) both extended to include it. `scripts/extract_lyrics.py`
added as a standalone CLI (same pattern as the other stages); `scripts/publish_song.py` now also
copies `lyrics.json`. `_flatten_words()` (the JSON-shaping logic) is unit tested directly against
constructed `faster_whisper` `Word`/`Segment` objects, without needing the model; the end-to-end
`extract_lyrics()` smoke test uses a non-speech synthetic clip (same accuracy caveat as the melody
smoke test -- proves the pipeline runs and produces valid JSON, not transcription accuracy).
14 Python tests pass.

**Frontend**: new `frontend/src/types/lyrics.ts` (`LyricWord`), `frontend/src/game/lyrics.ts`
(`getCurrentWordIndex(words, currentTime)` -- returns the word currently being sung, or during a
gap between words, the upcoming one; -1 once lyrics are finished; 6 vitest unit tests), and
`frontend/src/components/LyricsDisplay.tsx`. `LyricsDisplay` follows the same pattern established
for `NoteHighway`: a `requestAnimationFrame` loop reads `audioRef.current.currentTime` directly
(no React state, no separate clock) and only touches the DOM -- via `classList`, not React
state/re-render -- when the active word index actually changes, then calls `scrollIntoView` on the
new active word inside a horizontally-scrolling ticker (`overflow-x: auto`, `white-space: nowrap`)
so the highlighted word animates into view. `.lyric-word`/`.lyric-word--active` styles added to
`index.css` reusing the existing theme's accent variables (so it respects the light/dark toggle
like the rest of the page). `GameScreen.tsx` now fetches `notes.json` and `lyrics.json` in
parallel, and renders order is title -> audio controls -> `LyricsDisplay` -> `NoteHighway` (moved
down accordingly). 18 vitest tests pass total; `tsc --noEmit` clean.

**Manual verification**: loaded the game screen in a real browser. Confirmed (screenshot): lyrics
render above the highway with the first word already highlighted before playback starts (matches
the "index 0 = next word" design for `getCurrentWordIndex` at `currentTime=0`), no console errors.
Also visually reconfirmed the earlier "no notes during instrumental-only intro" fix: the highway
is empty under the playhead at `t=0` now, matching the real ~18s instrumental intro. Did not
verify the actual highlight-advances-and-scrolls-during-playback behavior or transcription
accuracy against the real vocals by ear -- both need the same manual human check already pending
from Phase 2 (audio playback can't be verified through this browser automation, see above).

## Lyrics follow-up: multi-line display + model accuracy (recorded 2026-07-27)

Two issues reported after the above: (1) wanted at least 3 lines of lyrics visible, not a single
horizontal ticker line; (2) the very first transcribed word was wrong.

**Multi-line display**: the flat word list had no line/phrase grouping (it was discarded when
flattening basic-pitch -- er, faster-whisper -- segments). Added a `line` index to each word in
`_flatten_words()` (`lyrics_extraction.py`): sequential per segment, but only incremented for
segments that actually produced words, so line numbers stay contiguous with no gaps (covered by
`test_flatten_words_does_not_reserve_a_line_number_for_empty_segments`). `lyrics.json` schema is
now `[{"word", "start", "end", "line"}, ...]`.

Frontend: `groupWordsByLine()` and `getCurrentLineIndex()` added to `game/lyrics.ts` (5 new vitest
tests). `LyricsDisplay.tsx` rewritten from a single horizontal ticker to three stacked lines
(prev/current/next), current line larger and full-opacity, prev/next dimmed -- classic karaoke
"teleprompter" layout. As the active word's line changes, `currentLineIndex` (React state, updated
only on line change, not per frame) shifts which three lines are shown, so the whole block visibly
moves down as the song progresses, on top of the existing per-word highlight within the current
line. One subtlety worth recording: the active-word DOM lookup is scoped to a wrapper `ref`
around all three lines, not just the "current" line's div -- on the exact frame a line transitions,
the new active word's `<span>` was rendered a moment earlier as part of the "next" row (React
hasn't re-tagged it as "current" yet, since the state update is async), so scoping the lookup to
only the current-line div would silently fail to find it right at every line boundary. Scoping to
the shared wrapper sidesteps this instead of relying on render timing.

**Wrong first word**: benchmarked `faster-whisper` model sizes `small` (already in use), `medium`,
and `large-v3` against the same real vocal stem. `small` and `medium` independently produced the
identical wrong wording ("Love and can hurt, love and can hurt sometimes" -- not grammatical
English, a sign something was actually mistranscribed, not just an unlucky homophone). `large-v3`
corrected it to "Loving can hurt, loving can hurt sometimes" -- grammatical and evidently the
right reading. This ruled out "just retune decoding params on the small model" as a fix; the
small/medium checkpoints were consistently wrong here, not just uncertain. Switched
`_MODEL_SIZE` to `large-v3` and added `vad_filter=True` (suppresses hallucinated words in
silence, same intent as the melody pipeline's RMS silence gate). Cost: 380.9s to transcribe the
274s test song (~1.4x realtime) vs. 142.8s for `medium` -- accepted since this is one-time offline
per-song processing and the accuracy gap was decisive, not marginal. Re-ran on the test song and
republished; full test suite still fast (large model load dominates on a real song, but the
short synthetic clips used in tests keep the smoke tests quick).

Same caveat as always: transcription of sung vocals is inherently imperfect even with the largest
local model, especially on melisma/fast lyrics/heavy vibrato -- expect some remaining errors, not
a fully solved problem.

**Manual verification**: regenerated `cache/test-song/lyrics.json` with `large-v3` (270s), 375
words across 52 lines, confirmed the first word is now "Loving" (matches the corrected reading
above). Republished and reloaded the game screen in a real browser: three-line layout renders
correctly (current line prominent with "Loving" pre-highlighted, next line dimmed below it, no
console errors). Did not verify line-transition/scroll behavior during actual playback or listen
to the full transcription by ear -- still blocked on the same audio-playback automation
limitation as Phase 2; needs the same pending manual human check.
