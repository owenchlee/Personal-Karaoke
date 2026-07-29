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

## Frontend redesign + load-song-from-a-link (recorded 2026-07-28)

Two requests: make the whole frontend look better (used the `ui-ux-pro-max` skill for direction),
and add a way to load a new song by pasting a link instead of running the CLI scripts by hand.

**Visual redesign**: `ui-ux-pro-max --design-system` for a "karaoke/music rhythm game" query
recommended a dark-first indigo/violet palette with a green CTA/play accent, Poppins/Righteous
typography, and glow/rounded-surface effects. Kept the existing light/dark `prefers-color-scheme`
architecture (the recommendation was dark-only, but this app already had both, and NOTES.md
already establishes that both are supported) and adapted the palette into it: violet accent
(existing brand color, refined), green as a new `--cta` token for primary actions, semantic
surface/border/radius/spacing tokens added to `index.css`. Deliberately did **not** pull in the
recommended Google Fonts (Poppins/Righteous) -- kept the system-ui stack, since this is an
offline/local personal tool and the frontend had zero non-React dependencies before this; adding a
network font fetch for a purely cosmetic upgrade didn't seem worth breaking that. Rebuilt
`App.tsx` (shared header/brand), `GameScreen.tsx`, `ProofScreen.tsx`, `NoteHighway.tsx` (canvas
colors + rounded note bars), and `LyricsDisplay.tsx` on top of the new tokens and a small set of
reusable classes (`.panel`, `.btn`/`.btn-primary`/`.btn-secondary`, `.input`, `.progress-bar`).

**Load a song from a link**: previously, getting a new song into the app meant running
`process_song.py` then `publish_song.py` by hand from a terminal. Added `scripts/server.py`, a
small FastAPI job server: `POST /api/jobs {url}` starts a background thread that downloads audio
via a new `audio_pipeline/download.py::download_audio()` (yt-dlp, `bestaudio` format -- feeds
straight into the existing `extract_audio` step, which is already codec/container-agnostic per the
Phase 1 notes above), runs it through `process_song()`, then reuses the existing
`scripts/publish_song.py::publish_song()` unchanged. `process_song()` gained an optional
`on_stage` callback (invoked before each stage: separating/extracting_melody/transcribing_lyrics)
so the job server can report progress; `GET /api/jobs/<id>` exposes it for polling. `pipeline.py`'s
`_slugify` was made public (`slugify`) since the server needs to compute the same slug the
pipeline will use. `frontend/vite.config.ts` proxies `/api` to `http://127.0.0.1:8000` so the
browser can call it same-origin, no CORS needed for the normal dev workflow. New
`LoadSongForm.tsx` posts the link, polls every 2s, shows a stage label + progress bar
(`game/jobStages.ts`, unit tested), and on completion updates the game screen's song slug (and the
URL's `?song=` param) without a page reload.

**Verification**: all 16 Python tests pass (added one covering the `on_stage` callback fires the
three expected stage names in order), all 27 frontend vitest tests pass (9 new, covering
`jobStages.ts`), `tsc --noEmit` clean. Manually loaded the redesigned game screen and diagnostics
screen in a real browser -- dark theme renders as intended, no console errors. Exercised the full
link-loading wire end-to-end with a fake URL (`https://example.com/not-a-real-video`): confirmed
the job is created, the frontend polls and displays the job's progress UI, and -- because this
sandboxed dev environment has no network access at all (even DNS resolution for `example.com`
fails) -- the download step fails and that failure correctly propagates through
`job.status = "error"` all the way to a styled error message in the form. This validates every
piece of the new wiring (routing, threading, polling, error propagation, UI) except the actual
"real YouTube link downloads and produces a playable song" happy path, which needs a real network
connection to test -- that part is not yet verified by a human.

## Phase 3: mic pitch detection + scoring (recorded 2026-07-28)

Picked back up the Phase 3 design that had been paused mid-brainstorm for the redesign/link-loading
work above. Design decisions from that brainstorm: full scoring (live overlay + running + final
score), octave-tolerant matching (compare pitch class, ignore octave -- most users won't sing in
the original recording's exact register), the live pitch trace folded to whichever octave the
current reference note is in (so a correct different-octave match visually lines up with the note
bar instead of looking like a big miss), and `pitchy` (McLeod Pitch Method) for detection since
hand-rolling pitch detection risked the same kind of iterate-until-correct debugging the melody
extraction accuracy fixes needed.

**Implementation**: `game/pitch.ts` (`hzToMidi`, `foldToNearestOctave`, `findReferenceMidi`,
`appendTracePoint` -- all pure, unit tested) and `game/scoring.ts` (pitch-class distance/match at
a 50-cent tolerance, per-note hit/total accumulation, averaged into a 0-100 score -- also pure,
unit tested). `hooks/useMicPitch.ts` wraps `getUserMedia` + `AnalyserNode` + `pitchy`, exposing the
latest sample through a ref (not state) so the 60fps detection loop doesn't force 60fps
re-renders -- same pattern already established for `NoteHighway`/`LyricsDisplay` reading
`audioRef.current.currentTime` directly. Each sample is timestamped with the *audio's*
`currentTime`, not wall-clock time, so it lines up with note onset/offset. `hooks/useScoring.ts`
runs its own rAF loop matching the latest sample against whichever note covers the current
playhead time (samples older than 200ms are treated as stale/not-currently-singing, e.g. during a
clarity dropout), accumulating per-note accuracy and exposing a running score (React state, so it
re-renders only when the rounded percentage actually changes) plus a `getFinalScore()` read on the
audio's `ended` event. Scoring resets when a new song loads, or when playback restarts from the
beginning (`currentTime < 0.5` on `play`) -- but not on a plain pause/resume, so a mid-song pause
doesn't wipe progress. `NoteHighway.tsx` draws the live pitch as a short (~1s) trailing line,
octave-folded per-point against `findReferenceMidi` at that point's time, in a distinct blue so it
doesn't get confused with the green reference note bars. `GameScreen.tsx` adds an "Enable Mic"
button (mirrors the `ProofScreen` permission-request pattern), a running score badge next to the
song title, and a final-score banner after the song ends.

**Verification**: 51 frontend vitest tests pass (24 new, covering `pitch.ts`/`scoring.ts` pure
logic), `tsc --noEmit` clean, all 16 Python tests still pass (unaffected -- this was frontend-only).
Manually loaded the game screen in a real browser and clicked "Enable Mic": it went active
immediately with no permission-prompt automation blocker this time (unlike the earlier
`ProofScreen` mic test, this browser profile already had mic permission granted for this origin),
score badge rendered "Score: 0%" with no console errors, and the note highway kept rendering
correctly with the new trace-drawing code path active. Same limitation as Phase 2's audio-sync
check applies here too, though: this automation tooling can't verify actual singing-synced
behavior (does the pitch trace track a real voice, does the score move sensibly against real
audio) -- that needs a human to actually sing along in a normal browser window and eyeball it.

### Correction (recorded 2026-07-28): trailing line was too jittery, replaced with a smoothed blob

Human testing (finally exercising the real-singing path this automation can't reach) surfaced two
problems: the live pitch indicator jumped up and down too much to be usable, and the trailing blue
line itself wasn't the wanted visual.

**Fix**: raw per-frame pitch detection is inherently noisy (vibrato, formants, occasional
misreads) -- plotting each detected sample directly was never going to look stable regardless of
color/shape. Added `smoothMidi()` to `game/pitch.ts`: takes the median of the last
`SMOOTHING_WINDOW_SECONDS` (0.25s) of samples, which resists a single-frame outlier much better
than a mean would (e.g. `[60, 60, 60, 60, 72]` -> median 60, not dragged toward the blip). The
existing `appendTracePoint`/`TracePoint` rolling-window logic is now repurposed for this
smoothing buffer rather than a line to draw. `NoteHighway.tsx` replaced the trailing multi-point
line with a single circle ("blob") drawn right on the white playhead line, at the y-position of
the current smoothed+octave-folded pitch -- simpler to read at a glance than a wandering line, and
directly addresses "I want a blob thing on the white line" from feedback.

Added 4 unit tests for `smoothMidi` (empty window, single sample, outlier resistance, even-window
averaging). 55/55 frontend tests pass, `tsc --noEmit` clean. Manually reloaded the game screen and
confirmed visually: a single blue-filled circle with a white ring now sits on the playhead line
(screenshot taken). Still can't verify through this automation whether the smoothing actually
feels right against real singing -- that's on the human tester to judge next.

## Live-indicator octave anchoring fix + debug overlay (recorded 2026-07-28)

Further real-singing feedback surfaced three more issues: the pill still didn't line up with what
was being sung, a note sometimes turned green (hit) even though the pill visibly wasn't on it, and
no way existed to check whether the mic or the reference melody was actually at fault. A screenshot
at test-song 1:22 made the second issue concrete: the pill sat at F#3 while the just-scored note
was F#4 -- exactly one octave apart.

**Root cause**: `NoteHighway.tsx`'s octave-fold anchor (`octaveAnchorRef`) folded each sample toward
*its own previous folded value*, never toward the actual note being judged. This was a deliberate
earlier fix (see the Phase 3 correction above) to stop the pill snapping to a future note during a
pause, but it over-corrected: once the detector locked onto the wrong octave of the true pitch (a
known failure mode of autocorrelation-based detectors when a strong harmonic outweighs the
fundamental), nothing ever pulled it back, since the anchor only ever looked at itself. Meanwhile
`useScoring.ts` scores octave-agnostically by design, so a same-pitch-class-different-octave sample
is a correct, intended hit -- but the pill never folded up to show that on-screen.

**Fix**: `NoteHighway.tsx` now anchors the fold to `findActiveNoteIndex(notes, currentTime)`'s note
(the exact same note `useScoring.ts` judges this sample against) whenever one is active, falling
back to the old self-anchoring only during silence gaps/instrumental breaks -- preserving the
original pause-fix behavior where there's no ground truth to anchor to. The existing
`isPlausiblePitchJump` gate (previously compared the fold result to the *previous* anchor) now
naturally compares it to *whichever anchor is used this frame*, which turns out to be exactly the
right semantics either way (per-frame "did folding land close to the anchor's pitch class," not a
cross-frame drift check) -- no change needed to `pitch.ts` itself.

**Debug overlay**: added a "Debug" checkbox (`GameScreen.tsx`) that draws a small readout directly
on the canvas (`NoteHighway.tsx`, `debug` prop): raw detected Hz/note/clarity/age, the active
reference note and cents distance to it, the folded/smoothed value actually driving the pill, and
hit% for the last few completed notes -- a concrete tool for checking detection against real
singing without reading code.

**Verification**: `tsc --noEmit` clean, all 84 frontend tests pass (no new pure-function logic was
extracted, so no new unit tests -- the composition change lives in `NoteHighway`'s draw loop, which
isn't unit-tested, consistent with the existing convention for that file). Verified end-to-end in a
real browser against `test-song`: patched `navigator.mediaDevices.getUserMedia` to return a
synthetic oscillator tone (185.0Hz, F#3 -- exactly one octave below the then-active F#4 reference
note) feeding the *real* `pitchy` detection pipeline, not a mock. Confirmed via the new debug
overlay: raw sample read F#3 correctly, but the pill now folds up to sit exactly on the F#4 bar
(`pill F#4 (midi 66.00)`, `class-dist 0c`), and the running score updates accordingly. Also
confirmed the gap-fallback path still works correctly: moving to a stretch with no active note
correctly fell back to showing the raw (unfolded) pitch rather than snapping to a wrong note, with
no regression of the original pause-fix behavior.

## Sequential octave-jump blips in the reference melody (recorded 2026-07-28)

Separately from the mic-side fix above, real singing feedback on the same screenshot flagged that
"the singer is not changing pitches so much, so the bars are not actually mapping to the singer
correctly" -- i.e. the *reference* note bars themselves looked erratic, not just the live indicator.

**Measured** directly against `cache/test-song/notes.json` (450 notes, post-`_enforce_monophony`):
51% of notes under 250ms, 124/449 (28%) of adjacent-note transitions jumping >=7 semitones, 49
exact-octave jumps. Narrowing further to notes `B` sandwiched with (near-)zero gap between two
neighbors `A`/`C` within a semitone of each other, jumping away by >=7 semitones and back: exactly
8 clean cases (durations 0.10-0.61s). Checked velocity as a possible discriminator and found none --
one case had `B`'s velocity *higher* than both neighbors -- confirming, same as this file's own
RMS-silence-gate finding, that basic-pitch's confidence isn't reliable here either; the
near-zero-gap condition (basic-pitch itself detecting no silence between the three) is the actual
signal, not duration or velocity. This is the same octave-confusion mechanism `_enforce_monophony`
already fixes for *overlapping* notes, just showing up sequentially instead.

**Fix**: added `_collapse_octave_blips()` to `audio_pipeline/melody_extraction.py`, run right after
`_enforce_monophony`. Drops a sandwiched blip and extends the preceding note to cover its span; if
the two neighbors are then *exactly* the same pitch (5 of the 8 real cases; the rest were one
semitone apart, likely a genuine re-attack) they're merged into one continuous note rather than
left as two separate touching notes of the same pitch. 7 new unit tests in
`tests/test_melody_extraction.py` covering the absorb case, a real-silence-gap non-match, a
too-far-neighbors non-match, a too-small-jump non-match, the exact-vs-tolerance merge distinction,
and an end-of-list boundary case. All 78 Python tests pass.

**Re-measured** (not assumed) after re-running `scripts/extract_melody.py` on the cached
`test-song` vocal stem: 450 -> 438 notes, big-jump transitions 124 -> 110, exact-octave jumps 49 ->
39 -- consistent with removing the measured 8 sandwich cases (some merges eliminate two jump-
transitions at once). Overall fragmentation barely moved (51.1% -> 50.5% under 250ms) since this
pass only targets the specific sandwich signature, not general fragmentation -- as expected; the
plan deliberately scoped this narrowly rather than re-opening the already-settled onset/frame
threshold tuning from the "Melody extraction quality fix" section above. Republished via
`scripts/publish_song.py test-song` (this also surfaced and fixed an unrelated stale gap: this
song's `lyrics.json` was already missing from the cache before this session touched anything,
apparently left over from an earlier session's work on lyrics zero-duration-word cleanup;
regenerated it via `extract_lyrics` and confirmed the first word is still "Loving," matching the
earlier corrected transcription).

## Live-indicator latency: blocked on the same automation limitation as audio/mic testing

Attempted to measure the live pill's actual step-response latency (does `SMOOTHING_WINDOW_SECONDS`
0.25s + `RENDER_SMOOTHING_TAU_SECONDS` 0.1s add up to noticeable lag) objectively -- via a synthetic
oscillator frequency step fed through the real detection pipeline, sampling the pill's canvas
pixel position on a `requestAnimationFrame` loop, so no live human singer would be needed. This
does not work in this automation environment: `document.visibilityState` reports `"hidden"` for
this browser tab (confirmed directly) even with focus, and Chrome throttles/suspends
`requestAnimationFrame` callbacks in non-visible tabs -- the exact same limitation this file already
documented for `<audio>` playback and mic-permission testing, just hitting `NoteHighway`'s own
render loop instead this time. A recording loop armed for 1.5s of internal (page-clock) time
captured exactly one sample before rAF stopped firing.

**Conclusion**: per the original plan, latency constants should only be retuned against real
evidence, not guessed -- and that evidence now requires a human, the same conclusion this file has
reached for every other real-time audio/mic behavior. The debug overlay added above already shows
both the raw and the smoothed/pill values live, so a human tester can judge directly (open the game
screen in a normal browser window, enable mic + debug, sing a quick pitch change, and watch whether
the "pill" line visibly lags the "raw" line) whether retuning is actually warranted before any
constants are touched.

## pYIN-corrected reference pitch + offset, and a shared live-indicator hook (recorded 2026-07-29)

Real singing (finally exercising the paths this automation can't) surfaced that the fixes above
weren't enough: the reference bars were still wrong often enough to make singing along frustrating,
notes routinely ended before the singer actually stopped, and there was still no easy way to tell
*which* side (mic detection vs. reference data) was at fault without reading code.

**Measured, not guessed, the scale of the reference-data problem**: cross-checked every note in
`cache/test-song/notes.json` against `librosa.pyin` -- an independent, purpose-built monophonic
pitch tracker, not basic-pitch's general-purpose polyphonic model -- computed once over the same
vocal stem. 98/438 notes (22%) disagreed with pYIN's median pitch for that note's own time span by
>=1.5 semitones, and 86 of those were *exact* octave disagreements -- basic-pitch's octave confusion
isn't limited to the touching-blip signature `_collapse_octave_blips` targets; it also shows up as a
single wrong-octave note with no matching neighbor to catch it against. Separately, of 185 notes
with a real gap before the next note, 67 (36%) still had the same pitch class voiced by pYIN in the
window right after the note's reported offset -- real singing continuing past where basic-pitch cut
the note off.

**Fix**: added `_refine_with_pyin()` to `audio_pipeline/melody_extraction.py`, run last (after
`_enforce_monophony`/`_collapse_octave_blips`). Keeps basic-pitch for note segmentation (timing is
its actual job) but corrects each note's pitch to pYIN's median (when enough of the note's frames
are voiced -- `_PYIN_MIN_VOICED_FRACTION`) and extends each note's offset while pYIN keeps reporting
the same pitch class with real RMS energy (reusing the existing `_SILENCE_RMS_GATE`), capped at the
next note's onset so monophony is never violated.

**First version of the offset-extension loop barely worked -- caught by measurement, not assumed
correct**: initially stopped scanning forward the instant a single ~12ms step didn't match (either
unvoiced or below the RMS gate). Re-measuring after implementing it showed the early-cutoff rate
barely moved (36.2% -> 35.2%), which shouldn't have been possible given the pitch-correction win was
huge. Debugged concretely (not guessed) by instrumenting the loop step-by-step on real flagged
notes: found cases where the very first micro-frame right after a note's offset was unvoiced by
pYIN even though the singer was clearly still holding the note a moment later (confirmed both by
the aggregate 60%-of-window metric and by ear from the surrounding data) -- a single noisy frame
was killing the entire extension. Rewrote the loop to tolerate brief gaps before giving up
(`_OFFSET_EXTEND_GAP_TOLERANCE_SECONDS`, same "brief interruptions shouldn't reset tracking"
philosophy as the frontend's own `HOLD_SECONDS`), only ever committing the offset up to the *last*
step that actually matched. Also separately confirmed (checking RMS on genuinely-still-flagged
cases) that a meaningful chunk of the "still flagged" cases after this fix are pYIN itself
hallucinating a persistent pitch over real near-silence (RMS ~0.001-0.002, an order of magnitude
below the gate) -- the same class of bug this file already found in basic-pitch's own confidence
score -- so the *measurement* script's 60%-pitch-match-only metric is itself an overcount of real
bugs, not just the fix falling short.

**Re-measured on the real test song** (before -> after this whole pYIN pass): pitch disagreement
with pYIN 22.4% -> 1.8%, wrong-pitch-class 2.7% -> 0.7%, notes under 250ms 50.5% -> 41.8%, median
duration 0.244s -> 0.279s, early-cutoff-flagged 36.2% -> 31.1% (with the caveat above that this
last metric likely still overcounts). 6 new unit tests in `tests/test_melody_extraction.py`
(pure helper tests for `_pitch_class_distance`/`_pyin_window`, plus `_refine_with_pyin` tests using
an in-memory synthetic harmonic tone: corrects a deliberately-wrong octave and extends a
deliberately-cut-short offset, doesn't extend past the next note's onset, and tolerates a brief
mid-extension dropout). 103 Python tests pass total. Adds real cost (pYIN itself takes ~28-30s on
this song, on top of basic-pitch's own time) but this is one-time offline per-song processing, same
tradeoff already accepted for the `large-v3` lyrics transcription. Re-ran extraction on the cached
vocal stem and republished; a spot-check of the game screen shows the reference melody now moving
in small, plausible steps instead of the erratic register-jumping visible in earlier screenshots.

**Frontend: "hard to debug when I don't know what note it's supposed to be"** -- the canvas-only
debug text from the previous round wasn't legible enough to be useful while actually singing.
Extracted the fold/smoothing state machine that used to live entirely inside `NoteHighway.tsx`'s
draw loop into a new shared hook, `hooks/useLivePitchIndicator.ts`, so every consumer of "what note
is being sung right now" reads the exact same computed value instead of risking two independent
implementations drifting apart. It exposes a per-frame ref (`stateRef`, for `NoteHighway`'s
pixel-perfect pill, which now just eases toward that shared target rather than computing the fold
itself) and throttled (100ms) React state (`display`, for text -- text doesn't need 60fps churn).
`GameScreen.tsx` now renders an always-visible "TARGET F#4  YOU F#4  0c" readout (DOM, large text,
color-coded green when within 50 cents) whenever the mic is active -- not just behind the debug
toggle -- plus a proper DOM debug panel (behind the toggle) with raw Hz/clarity/age and per-note
hit% for the last few completed notes, replacing the old tiny canvas text. New `signedCentsOff()`
pure helper in `game/pitch.ts` (3 unit tests). 91 frontend tests pass, `tsc --noEmit` clean. Also
fixed an unrelated `--text-muted` CSS variable typo (should have been the existing `--muted` token)
found while touching this file, which had been silently rendering the debug toggle label with no
color since it was introduced.

**Verified end-to-end in a real browser** against the republished `test-song`: patched
`navigator.mediaDevices.getUserMedia` to feed a synthetic F#3 tone (one octave below an active F#4
reference note) through the real `pitchy` detection pipeline. Confirmed the pill folds up onto the
F#4 bar, the readout shows "TARGET F#4 YOU F#4 0c" in green, and the debug panel shows the raw
185Hz/F#3 reading alongside it -- the exact "which side is wrong" distinction the debug UX was
built to make visible. Separately confirmed visually that the republished reference melody itself
now moves in small, stepwise motion in a stretch that previously jumped registers erratically.
Still not independently verified against a real human voice (same automation limitation as
everything else real-time in this file) -- that's the next check for a human tester, now with much
better tooling to diagnose whatever's still wrong.

## Lyrics drifting out of sync with the song ("song is faster than the lyrics") (recorded 2026-07-29)

Reported: lyrics increasingly out of sync with the song as it plays, plus a separate report of some
reference notes looking wrong ("short jumps to super high pitch").

**Frontend ruled out first**: `LyricsDisplay` and `NoteHighway` both read `audioRef.current.currentTime`
from the exact same `<audio>` element every frame, no separate clock -- there is no code path where
the frontend could let lyrics and the note highway drift apart from each other or from the song.
`notes.json` timing is always derived locally from this song's own vocal stem (basic-pitch + pYIN,
sharing the identical timeline as the instrumental track Demucs produces alongside it), so it can't
drift either. That leaves lyrics sourced from the online lrclib lookup (`lyrics_lookup.py`,
triggered whenever a song is loaded via a title, e.g. through `scripts/server.py`'s link-loading
flow) as the only place a real desync could originate: those timestamps belong to whatever release
lrclib matched by title, not this specific video.

**Root cause**: `_align_synced_lyrics_to_audio` already corrected a *constant* intro-length mismatch
(re-anchoring the first word to this audio's own measured vocal onset), but assumed lrclib's line
timing otherwise ran at the exact same pace as this video throughout. A genuine tempo/pacing
difference between releases (a different edit, a slightly different mix speed) can't be fixed by a
single offset -- it shows up as a gap that grows across the song, exactly the reported symptom, and
the existing code had no way to detect or correct it.

**Fix**: added `_find_last_vocal_activity` (the mirror of the existing `_find_first_vocal_onset`,
scanning from near the reported last word's end to the end of the audio) and `_fit_time_correction`,
which fits a two-point affine map (`true_time = scale * reported_time + offset`) between
(first word start -> first detected onset) and (last word end -> last detected activity) instead of
assuming `scale=1`. Falls back to the previous pure-offset behavior whenever the fit isn't
trustworthy -- last-activity not found, the two reported anchors too close together for a stable
slope, or the fitted scale outside a generous-but-sane `[0.9, 1.1]` band (a bad end-of-song
detection, e.g. trailing applause, is more likely than a >10% tempo difference between two releases
of the same song). `_align_synced_lyrics_to_audio`'s existing single-offset tests all still pass
unchanged, since their short synthetic fixtures (words a fraction of a second apart) always produce
an implausible fitted scale and correctly fall back to the exact same offset-only path already
covered. Added 4 new unit tests: `_fit_time_correction` in isolation (good two-anchor fit, missing
last anchor, implausible scale, anchors too close), plus one full-integration test with two
independent synthetic singing bursts (~80s apart) confirming the correction actually differs between
the start and end of the song -- the concrete proof this isn't just another constant shift.

**Verification**: installed the actual project dependencies into the local venv (numpy, soundfile,
pytest, torch-cpu, faster-whisper, requests, librosa -- turns out this sandboxed environment *does*
have real network access via pip, contradicting earlier sessions' "no network access at all" finding
for the browser-automation tool specifically; that limitation was scoped to that tool, not this
shell) and actually ran `tests/test_lyrics_extraction.py`: all 33 tests pass (one new test's expected
value had an arithmetic slip caught on the first run and fixed, then re-verified passing).

**Investigated the pitch-spike report against real cached data, not guessed**: wrote a standalone
script reusing the same pYIN-tracking approach as `melody_extraction._refine_with_pyin`, run against
every real song already cached (`ed-sheeran-photograph-lyrics`, `huang-jin-shi-dai`,
`james-arthur-car-s-outside-lyrics`, `jamie-miller-with-salem-ilese-here-s-your-perfect-lyrics`,
`sam-smith-i-m-not-the-only-one-lyric-video`, `song`, `test-song`). On `test-song`: 46/437 notes
(10.5%) match the "short duration + big jump from a neighbor" shape the existing
`_drop_unconfirmed_pitch_spikes` filter targets, but all 46 are independently confirmed by pYIN (the
note's own reported pitch agrees with pYIN's own median for that exact window) -- meaning both
trackers agree these are real fast pitch movements, not artifacts, and zero "slip through"
unconfirmed. This validates the existing filter's design (already covered by
`test_drop_unconfirmed_pitch_spikes_keeps_a_well_confirmed_short_big_jump_note`'s reasoning) rather
than exposing a gap in it. Results across the other cached real songs recorded once that run
finishes -- if the pattern holds everywhere, the "super high pitch" complaint is more likely a real
(if surprising) vocal run the extraction correctly caught, and the actionable next step is a
timestamp from the reporter so the *specific* flagged instance can be checked against the actual
audio by ear, rather than further threshold tuning with no measured evidence of an actual miss.

## Live-indicator lag fix + difficulty reduction (recorded 2026-07-29)

Real singing (the human check this file's "Live-indicator latency" section above was blocked on)
confirmed the suspicion recorded there: the pill visibly lags real singing, following the shape of
the reference highway only *after* a note bar has already crossed the playhead rather than tracking
it live -- exactly the "does `SMOOTHING_WINDOW_SECONDS` + `RENDER_SMOOTHING_TAU_SECONDS` add up to
noticeable lag" question that section left for a human to judge. Also requested separately: reduce
overall difficulty.

**Lag fix**: both constants stack directly (median-filter lag from the smoothing window, then
further easing lag from the render-side tau), so trimmed both rather than either alone.
`useLivePitchIndicator.ts`'s `SMOOTHING_WINDOW_SECONDS` 0.25 -> 0.15 (still ~9 samples at the
~60fps detection rate, enough to resist single-frame outliers, but median lag from a pitch change
drops from up to ~125ms to ~75ms). `NoteHighway.tsx`'s `RENDER_SMOOTHING_TAU_SECONDS` 0.1 -> 0.05
(halves the additional easing lag on top). Not independently measurable through this automation
environment (same `requestAnimationFrame`-throttling-in-hidden-tabs limitation already documented
above) -- a human tester should confirm the pill now tracks more in real time before further
tuning.

**Difficulty reduction**: `game/coords.ts`'s `DEFAULT_PX_PER_SECOND` 200 -> 160 (slower highway
scroll, more reaction time before a note reaches the playhead -- doesn't touch note timing itself,
which stays driven straight off `audioRef.current.currentTime`). `game/scoring.ts`'s
`CENTS_TOLERANCE` 50 -> 70 (wider pitch-match window) and `NOTE_HIT_FRACTION_THRESHOLD` 0.5 -> 0.4
(a note now counts as hit with a smaller fraction of matched samples).

All 91 frontend vitest tests pass unaffected (none assert these specific constant values), `tsc
--noEmit` clean. Not yet re-verified by ear/by singing against the real test song -- same pending
human check as above.
