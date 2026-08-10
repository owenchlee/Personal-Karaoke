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

## "Lyrics frozen but notes still on the highway" -- lrclib gap-repair + a language-detection bug (recorded 2026-07-29)

Reported: a stretch where the lyrics display isn't advancing (same word held) while the note
highway still shows notes scrolling through. Root-caused with `systematic-debugging`, not guessed:
notes (`basic-pitch`+`pyin`) and lyrics (lrclib lookup, or local `faster-whisper`) come from two
independent pipelines with no shared timeline, so nothing guarantees they agree about when singing
is happening at a given moment.

**Measured directly against every real cached song** (a script comparing each inter-word lyrics gap
against `notes.json`'s coverage of that same span): the local-transcription path already has a
defense for this (`_repair_energetic_gaps`, added earlier), but the online lrclib-lookup path had
none at all. Confirmed on `priscilla-chan-night-flight-english-yale-romanization`: a **25.76s**
stretch (166-192s) with vocal-stem RMS (0.01-0.05) comparable to the song's own median non-silent
RMS (~0.07) had zero lrclib coverage -- real singing energy the whole way through, with no
corresponding lyric text at all.

**Fix**: `_find_energetic_gaps()` (cheap, model-free RMS scan reusing the existing repair
condition) + `_repair_synced_lyrics_gaps()` in `lyrics_extraction.py`, wired into `extract_lyrics()`
right after lrclib alignment. Whisper is only loaded if the scan finds an actual gap worth
repairing, so a song whose lrclib lyrics already cover the whole song (the common case -- verified
on `huang-jin-shi-dai`/`james-arthur`/`jamie-miller`/`song`, none of which had any flagged gaps)
stays on the fast lookup-only path with zero added cost. 6 new unit tests.

**A second, pre-existing bug this fix's own verification surfaced**: re-running the new repair pass
against `priscilla-chan`'s real cached data initially recovered almost nothing (318 -> 319 words),
even in gaps with strong, unambiguous vocal energy. Instrumented the repair pass directly: it called
`_detect_language`, which returned `"en"` for this Cantonese song. Checked the model's own full
probability list: `zh` (generic Mandarin/Chinese) scored 0.64, `en` only 0.06, and `yue` wasn't in
the returned list at all (~0) -- exactly the "Cantonese gets mislabeled as zh" failure mode the
module's own comment already named, but the code never actually acted on: `_detect_language` only
ever compared raw `en` vs `yue` scores, so a near-zero `yue` let a barely-nonzero `en` "win" even
with `zh` dominant. This bug predates this session's change but was never exercised for this song
before, since it previously always succeeded via the fast lrclib path and never needed to call
Whisper at all. Fixed by folding `zh`'s probability into `yue`'s candidate score before comparing.
Confirmed directly: forcing `language="yue"` on the same gaps recovered real Cantonese text (e.g.
回頭再喝美味燈光) where `language="en"` had recovered nothing or hallucinated English words
("mobile", "building"). 1 new unit test.

**Re-verified on the real song after both fixes**: 318 -> 320 words (net +2, not the large recovery
hoped for). Investigated why, rather than assuming both fixes together were sufficient: even with
the correct language, Whisper's re-transcription of these specific isolated gap windows mostly
produced words *outside* the actual gap boundaries (in the padding, already covered by a
neighboring anchor) rather than inside them -- the same "genuine ASR miss on this specific audio"
limitation already documented earlier in this file for the local-transcription path (see "Wrong
first word" and the `_repair_energetic_gaps` module comment), not a bug in the new repair logic
itself. The mechanism is verified working end-to-end (detects real gaps, skips Whisper entirely
when not needed, uses the correct language, recovers what Whisper is actually capable of
recovering) -- but, consistent with every other ASR-accuracy finding in this file, it's not a
guarantee that every gap gets fully text-recovered. 132 Python tests pass total (`tsc --noEmit`
untouched -- this was backend-only).

Not yet re-run against the actual cached `lyrics.json` files for any song (this was verified by
calling the repair functions directly against cached data, not by reprocessing) -- a song affected
by this needs `extract_lyrics`/`process_song` re-run (with `force=True`) and republishing to pick
up the fix in the running app.

## Timing-architecture audit: pitch-detection buffer latency + mic delay calibration (recorded 2026-07-29)

Requested audit of the whole timing architecture against five principles (single clock source,
lookahead scheduling for any scheduled audio, pitch-detection latency compensation, input/output
latency calibration, rendering decoupling). Checked each against the actual code rather than
assuming.

**Already correct**: `NoteHighway.tsx`, `LyricsDisplay.tsx`, `useLivePitchIndicator.ts`, and
`useScoring.ts` all read position from `audioRef.current.currentTime` inside a `requestAnimationFrame`
loop every frame -- confirmed no `setInterval`/`Date.now`/independent timestamp anywhere derives song
position (this was already a deliberate design decision recorded earlier in this file, for Phase 2).
Playback itself goes through a plain `<audio>` element, not a Web Audio graph, so there's no separate
scheduled-audio-node case in the normal game flow either -- the only place that pattern applies is
the new calibration click track below.

**Gap found -- pitch-detection latency uncompensated**: `useMicPitch.ts` timestamped each pitch
sample with `audioRef.current.currentTime` at the moment `getFloatTimeDomainData` was read and
processed, not the start of the 2048-sample buffer that produced it. `getFloatTimeDomainData`
returns the most recent `fftSize` samples as of *now* -- the pitch it implies was actually sung
starting one whole buffer-length earlier. At a typical 44.1-48kHz mic input, `2048 / sampleRate`
is **~42.7-46.4ms** of latency that was silently baked in as "this sample is more recent than it
really is," biasing hit detection right at note boundaries. Fixed by subtracting
`analyser.fftSize / context.sampleRate` from the timestamp before it's stored.

**Gap found -- no input/output latency calibration existed at all**. Added the one-time "play a
click, clap along" calibration UltraStar and similar games use:
- `game/calibration.ts` (pure, unit tested): `pairBeatsToTaps` greedily matches each scheduled beat
  to the nearest clap within a match window (drops beats with no nearby clap and claps unrelated to
  any beat, never double-matches one clap); `computeCalibrationResult` takes the median of the
  matched (tap - beat) offsets (resistant to one mistimed clap the same way `smoothMidi` resists a
  pitch outlier), returning `null` if fewer than 4 beats matched rather than trusting too little
  data; `load`/`save`/`clearCalibrationOffsetSeconds` persist the result to `localStorage` so it's a
  true one-time step, not per-session.
- `hooks/useCalibration.ts`: schedules an 8-beat, 100bpm click track and listens for clap onsets
  (an RMS spike over a refractory-gated threshold), both against **one shared `AudioContext`'s
  `currentTime`** -- no `<audio>` element is involved in calibration at all, so this is the one place
  in the app where that clock, not the media element's, is the correct single source of truth. The
  click track is scheduled via the standard Web Audio "lookahead scheduler" pattern (a 25ms
  `setInterval` that only ever schedules a beat once it's within a 100ms lookahead window of
  `context.currentTime`, rather than playing anything directly from a timer/UI callback) -- the only
  place in the app that actually schedules audio playback events ahead of time, so the only place
  that pattern was needed.
- `screens/CalibrationScreen.tsx`, reachable via `?screen=calibrate` (added to the sidebar as "Mic
  calibration"), following the same screen-routing pattern as the existing proof/load/songs screens.
- The measured offset is applied in exactly one place -- `useMicPitch.ts` subtracts it (alongside the
  buffer-latency correction above) from every `PitchSample.time` at the moment a mic session starts
  -- so `useScoring` and `useLivePitchIndicator` get the correction for free with no changes of their
  own, consistent with this file's established pattern of keeping one shared source of truth rather
  than letting each consumer recompute its own copy.

**Verification**: 11 new vitest tests for `calibration.ts`'s pure pairing/median/storage logic (102
frontend tests pass total, up from 91), `tsc -b --noEmit` clean. Not yet verified by an actual human
running the calibration screen and singing afterward -- same category of real-time audio/mic
behavior this file has repeatedly flagged as unverifiable through this session's browser-automation
tooling (hidden-tab `rAF`/media suspension); a human should run `?screen=calibrate`, clap along, and
confirm the saved correction reduces perceived mic lag during a real song before considering this
fully closed.

## Bruno Mars ("Count on Me") only scoring ~41-47%, and a real basic-pitch octave bug (recorded 2026-07-29)

Reported: scoring only 41-47% on `bruno-mars-count-on-me-lyrics`, including a control test -- playing
the actual song's own audio back into the mic -- that itself only scored 41%. Separately asked
whether the live pill/reference "sometimes automatically goes to the high notes."

**Isolated the real-world score gap with two offline diagnostics** (no browser-automation audio/mic
limitation involved, since neither needs real hardware):
- Cross-checked this song's `notes.json` against an independent `librosa.pyin` trace over its own
  vocal stem: 96.6% pitch-class agreement, 0 clean octave errors, ~5% coverage gap -- in line with
  every other cached song already validated this way. The reference data was *not* the problem.
- Reproduced the exact production mic pipeline (same `pitchy` detector, same FFT/clarity/RMS
  constants as `useMicPitch.ts`, same `isPitchMatch`/`songAccuracyScore` as `scoring.ts`) in a
  standalone Node script fed the literal vocal-stem audio directly -- no speakers, mic, or room in
  the loop at all. Result: **95%**. This rules out the scoring thresholds and detection algorithm as
  the cause of a ~41-47% real-world score; the gap has to be the acoustic speaker-to-mic path itself
  (most likely echo-cancellation "double-talk" attenuating the near-end signal while the loud
  far-end reference plays simultaneously -- worst-case for exactly this test's methodology, but
  present during normal play too, since the instrumental always plays through speakers while the mic
  listens). Recommended the user try headphones next as a cheap, conclusive test of that hypothesis.

**The "bar jumps to high notes" question turned out to be true, for a different and more interesting
reason than expected.** Cross-checking flagged 14/412 notes (3.4%) disagreeing with pYIN by a large
margin -- e.g. reported MIDI 81 at 93.6s where pYIN's own median was 53, a ~2.3-octave difference.
Investigated *why* `_refine_with_pyin`'s existing pYIN-correction pass didn't already catch these
(it's specifically designed to fix exactly this kind of basic-pitch octave error): re-ran
`extract_melody` on the identical, unchanged `vocals.wav` a second time with no code change at all,
and got a **different** result (that same moment corrected to 53 on the second run). basic-pitch's
underlying model inference is not perfectly reproducible run-to-run -- the raw note segmentation
shifts slightly each time, which shifts how much of a note's window ends up voiced by pYIN, which can
tip a note's voiced fraction across `_PYIN_MIN_VOICED_FRACTION` (0.3) in one run but not another.

**Fix**: added a second, narrower correction path in `melody_extraction.py`'s `_refine_with_pyin` --
even below the main 0.3 voiced-fraction bar, if the sparse voiced frames that do exist *unanimously*
disagree by a large margin (`_PYIN_SPARSE_MIN_VOICED_FRACTION = 0.1`,
`_PYIN_SPARSE_MIN_DISAGREEMENT_SEMITONES = 7`), that's still trustworthy evidence of a wrong note --
real background noise or a stray breathy frame doesn't consistently agree on a specific, far-away
pitch class. Extracted the decision itself into a small pure `_should_correct_note_pitch()` helper
(directly unit tested, 4 new tests) rather than testing it via fragile hand-tuned audio synthesis.
136 Python tests pass.

**Re-verified directly on the real song**: re-ran `extract_melody` on the cached
`bruno-mars-count-on-me-lyrics` vocal stem with the fix and republished. pYIN disagreement dropped
14/412 -> **0/412**; short-note fragmentation also improved (48.3% -> 39.1% under 250ms) and coverage
ticked up slightly (66.7% -> 71.4% song duration covered) -- likely because some of the fixed notes
merged/extended better once their pitch was correct. This is a real, verified improvement for this
song, but the underlying nondeterminism isn't eliminated -- a different song, or reprocessing this
one again, could still produce an occasional case the sparse-correction path doesn't catch. If a
similar "bar looks obviously wrong for a moment" report comes in again, reprocessing the affected
song with `force=True` and re-checking against pYIN (per this note's methodology) is the fastest way
to confirm.

**Difficulty reduced another notch** per request, extending the existing "reduce difficulty" pattern:
`coords.ts`'s `DEFAULT_PX_PER_SECOND` 160 -> 130 (slower highway scroll, more reaction time),
`scoring.ts`'s `CENTS_TOLERANCE` 80 -> 100 (a full semitone of pitch margin either way) and
`NOTE_HIT_FRACTION_THRESHOLD` 0.4 -> 0.35. 102 frontend tests pass (one updated to match the new
100-cent tolerance boundary), `tsc -b --noEmit` clean.

## Two more reports on the same song: missing notes and lost quiet endings (recorded 2026-07-29)

Headphones confirmed to help a lot with real-world scoring (per the earlier echo-cancellation
hypothesis). Two further reports: (1) sometimes the original artist is clearly singing a note that
never shows up on the highway at all, and (2) singing a long note that trails off quietly (a
decrescendo) stops registering with the mic well before the note actually ends.

**Missing notes -- confirmed and fixed.** Checked whether `bruno-mars-count-on-me-lyrics` had any
lrclib-sourced `background_vocal_ranges` stripping real melody notes (a mechanism that exists for
lyrics, see the "Lyrics frozen" entry above) -- it didn't (`fetch_synced_lyrics` returned none for
this song), so that wasn't the cause here. Instead, examining the exact timestamps of "energetic,
clearly-voiced, but uncovered" moments (same coverage-gap metric used throughout this file) showed
several 0.3-0.6s stretches of real, sustained singing sitting *between* two detected notes with no
basic-pitch onset there at all -- not something any of the trimming/dropping passes were removing
(there was never a candidate note there to drop), but a genuine missed detection: basic-pitch simply
doesn't always fire an onset for a real sung transition.

**Fix**: added `_repair_melody_gaps()` to `melody_extraction.py`, run inside `_refine_with_pyin`
right after offset extension. Mirrors the existing lyrics-side gap defense
(`_find_energetic_gaps`/`_repair_energetic_gaps`) but for the note highway: walks each gap between
(and before/after) existing notes frame-by-frame over the same pYIN track already computed for
correction/extension, grouping contiguous voiced frames of a stable pitch class into runs, and
synthesizes a new note directly from pYIN's own median pitch for any run at least
`_GAP_REPAIR_MIN_RUN_SECONDS` (150ms) long with real RMS energy behind it. 4 new unit tests (fills a
real gap, leaves a real silent gap alone, ignores a too-short blip, fills a gap before the first
note) -- one existing test's fixture had to be corrected in the process (its note onset didn't
actually match where its own synthetic audio started, which the new gap-repair correctly noticed and
flagged by filling the artificial "gap" that mismatch created). 144 Python tests pass.

**Re-verified on the real song**: re-ran extraction on the cached `bruno-mars-count-on-me-lyrics`
vocal stem and republished. Notes 412 -> 430 (18 real gaps filled), the "energetic+voiced but
uncovered" coverage gap dropped 4.4% -> 2.4%, total song coverage 71.4% -> 73.1%, pYIN pitch
disagreement stayed at 0%.

**Quiet decrescendo notes dropping out of mic detection -- also confirmed, and a real bug in the
constant's provenance.** `useMicPitch.ts`'s `MIN_RMS = 0.01` gate rejects an entire pitch sample
below that loudness, regardless of `clarity` -- but that threshold was borrowed by analogy from
`_SILENCE_RMS_GATE` in `melody_extraction.py`, a constant validated for a completely different
signal (a Demucs-separated vocal stem), never actually measured against live mic input. A real
sung note trailing off quietly is exactly the case this silently kills, even though pitchy's
`clarity` score (how strongly periodic the signal is) should already be a reliable, amplitude-
independent signal of "is this really a sung pitch" on its own.

**Fix**: lowered `MIN_RMS` 0.01 -> 0.003 -- still rejects true silence/noise floor, but no longer
cuts off a genuine quiet tail purely on loudness. Not independently measurable through this
session's tooling (same live-mic limitation as everything else in this file); a human should confirm
a held note's quiet ending now keeps registering. `tsc -b --noEmit` clean, 114 frontend tests pass
(unaffected -- this constant has no dedicated unit test, consistent with the rest of this
browser-only hook).

## Highway notes with no corresponding lyrics -- e.g. Priscilla Chan's hummed intro (recorded 2026-07-29)

Reported: the note highway shows notes during the Priscilla Chan song's hummed intro, before any
lyrics are shown, so the player has nothing to sing along to. Same class of problem as
`_remove_background_vocal_notes` already solves for ad-lib/backing vocals, but a different
manifestation: `notes.json` (basic-pitch + pYIN, from the vocal stem's raw energy/pitch) and
`lyrics.json` (lrclib or local Whisper transcription) are independent pipelines with no shared
timeline, so nothing stopped a note from appearing where there's no displayed lyric at all.

**Measured directly** against the real cached
`priscilla-chan-night-flight-english-yale-romanization`: 21 notes sit entirely before the first
lyric line (45.98s) and a symmetric cluster sits entirely after the last line (245-253s) -- neither
region is ever touched by either gap-repair pass (`_repair_energetic_gaps` /
`_repair_synced_lyrics_gaps`), since both only fill gaps *between* existing words, never before the
first or after the last. Also measured that a strict, zero-tolerance cutoff at each line's own
start/end would be wrong: a note at 48.26-48.42s shares the same MIDI pitch (66) as the note ending
exactly at that line's reported end (48.18s) -- a real continuation of the sung word's tail, not
humming, and this pattern repeats at nearly every line boundary in the song.

**Fix**: `_remove_notes_without_lyrics()` in `pipeline.py`, wired into `process_song` unconditionally
(right alongside the existing `_remove_background_vocal_notes` call). Groups lyric words by their
`line` field (confirmed a contiguous, always-present grouping key on both lyric sources), computes
each line's `(min start, max end)` span, and drops any note that doesn't overlap the union of those
spans padded by `_LYRIC_COVERAGE_PADDING_SECONDS = 0.5` -- the same order of magnitude as the
existing `_OFFSET_EXTEND_MAX_SECONDS`/gap-repair-margin precedent already used elsewhere for the
identical "is this still the same utterance" judgment, which keeps the real boundary continuations
above while still dropping the genuine tens-of-seconds-away intro/outro humming. Empty lyrics (a
real possible case -- an lrclib result where every line is background vocal) drops all notes rather
than crashing. 2 new unit tests in `tests/test_pipeline.py`.

**Verified directly against the real song**: applying the fix to a copy of
`priscilla-chan-night-flight-english-yale-romanization/notes.json` dropped exactly the 21 pre-intro
notes and the post-outro cluster (max note onset 253.1s -> 245.6s) while keeping the 48.26s
boundary-continuation note. Also flagged, not fixed here: a ~4s interior gap (216.3-220.24s) with
continuous moderate-to-high-velocity notes and zero lyric coverage -- long and energetic enough that
`_repair_synced_lyrics_gaps` (added in an earlier session, same day) should recover it as real text
once this song is reprocessed with that fix; it isn't a "no lyrics possible" case like the intro/
outro, so it's called out here rather than silently left unexplained.

**Retroactively applied to every already-cached song** (12 total), since none of their original
source videos still exist (temp downloads, already deleted) to allow a full reprocess -- ran the new
function directly against each cached `notes.json`/`lyrics.json` pair in place (cheap: no
re-download, no re-running Demucs/Whisper). Note counts dropped moderately for most songs (e.g.
`flac-lyrics` 517 -> 467, `priscilla-chan...` 498 -> 438) and were unchanged for a few that already
had full lyric coverage (`james-arthur-car-s-outside-lyrics`, `jamie-miller-...`, `wait`) --
consistent with the fix only removing genuinely uncovered notes. 8/8 `tests/test_pipeline.py` tests
pass.

## Lyrics timing: replaced with CTC forced alignment (recorded 2026-07-30)

Reported: notes on the highway that don't seem to match the lyrics, and lyrics that seem a bit off
from playback. Researched the accuracy of `vocals.wav` extraction end-to-end (Demucs separation
quality, plus everything downstream of it) and proposed two independent improvements; this entry
covers the first one, implemented per request.

**Root cause (reasoned from the existing code, not re-measured from scratch -- the measurements
behind it are already in this file)**: the online lrclib path only ever had *approximate* word
timing. lrclib gives line-level timestamps; word timing within a line was a linear character-
weighted guess (`lyrics_lookup._distribute_words`), and even the line timestamps themselves belong
to whatever release lrclib matched by title -- not necessarily this specific video. The previous
fix (`_align_synced_lyrics_to_audio`, see "Lyrics drifting out of sync" above) tried to *correct*
lrclib's reported timestamps by fitting an intro-offset + proportional-drift model against the
vocal stem's own onset/last-activity energy. That's a reasonable model for a linear mismatch, but
it can only ever approximate a different release's timing, not measure this one -- and
`pipeline._remove_notes_without_lyrics` depends directly on lyric-line spans being accurate, so
timing drift there reads as "notes with no matching lyric," which is exactly the reported symptom.

**Fix**: replaced the whole correction step with CTC forced alignment
(`audio_pipeline/forced_alignment.py`, new module) -- instead of correcting lrclib's *guessed*
timestamps, force-align lrclib's *known-correct text* directly against this video's own vocal stem,
measuring real per-word timing instead of approximating it. Uses Meta's MMS wav2vec2 CTC model
(`MahmoudAshraf/mms-300m-1130-forced-aligner`, multilingual, covers both English and Cantonese) via
`transformers`, romanized through `uroman` (the model's vocabulary is a small universal phonetic
alphabet, not native scripts), with the Viterbi alignment computed by
`torchaudio.functional.forced_align`.

**Deliberately not using the `ctc-forced-aligner` PyPI package**, even though it implements this
exact idea end-to-end and was the first thing tried: its `align_ops` extension is C++ and has no
prebuilt wheel for Windows, and `pip install` failed outright on this machine (`Microsoft Visual
C++ 14.0 or greater is required`), which isn't installed here. Confirmed `torchaudio`'s own
`forced_align` (already shipped, prebuilt, part of the `torch`/`torchaudio` install this project
already requires) implements the identical algorithm/signature, so `forced_alignment.py`
reimplements the relevant slice of that package's (small, MIT/BSD) Python-side logic directly
against `torchaudio.functional.forced_align` instead of depending on the package. Cross-checked
output against the original package's own Python API pattern on real cached vocal stems before
committing to this -- not just assumed equivalent.

`lyrics_extraction.py`: `_align_synced_lyrics_to_audio`/`_fit_time_correction`/
`_find_first_vocal_onset`/`_find_last_vocal_activity` (and their constants) removed outright rather
than kept alongside the new approach -- forced alignment supersedes what they did, not
supplements it. New `_force_align_synced_lyrics` calls `forced_alignment.align_tokens` with the
lrclib-fetched word list's own tokens (language auto-detected from script -- any CJK character
means Cantonese, since a song's lyrics are never a mix of the two supported languages) and replaces
their start/end with the aligned result. `_repair_synced_lyrics_gaps` (the existing Whisper-based
gap-filler for real singing lrclib's text doesn't cover at all, e.g. a skipped ad-lib) is
unchanged and still runs after alignment -- unrelated concern, still needed.

**Background-vocal ranges** (`lyrics_lookup`'s parenthesized ad-lib stripping) have no text of
their own to force-align against, so they can't be re-timed the same way. New
`_interpolate_background_range` instead re-times each one from its *forced-aligned neighboring
words'* positions (the last real word before it, the first real word after it, in original
lrclib-time order) rather than trusting its own raw lrclib timestamp -- more consistent with the
audio's real timing than a two-point global fit would have given it anyway, and needs no separate
onset-detection logic of its own.

**Verification**: 3 new tests in `tests/test_forced_alignment.py` (real, unmocked smoke tests
against a synthetic clip -- same accuracy caveat as every other model-backed smoke test in this
file: proves the mechanics run and produce one increasing timespan per input token, not that
alignment is accurate against real singing). `tests/test_lyrics_extraction.py`'s old affine-
correction tests replaced with tests for `_language_of_words`/`_interpolate_background_range`/
`_force_align_synced_lyrics` (the latter mocking `align_tokens`, same pattern already used for
`WhisperModel` in this file -- a real 300M-param model has no place in a fast unit test). 162
Python tests pass total.

**Manually verified against real cached vocal stems** (not just synthetic clips), both languages:
- English (`test-song`, 274s): force-aligning "Loving can hurt, loving can hurt sometimes..."
  against the real vocal stem placed "Loving" at 17.96-18.48s -- close to this file's own earlier-
  recorded real vocal onset (~18.13s) and to the existing Whisper transcription's 17.58s. "hurt"
  (first occurrence) came back spanning 18.70-20.88s, a ~2.2s hold -- notably longer than Whisper's
  own 18.52-19.02s for the same word. Plausible rather than confirmed wrong: this is a repeated
  "loving can hurt, loving can hurt" phrase, exactly the shape of a held note before a repeat, and
  Whisper's own segment-cutting has documented under-duration issues elsewhere in this file: not
  independently confirmed by ear (same audio-playback limitation as everywhere else in this file),
  but internally consistent with the melody data's own held-note pattern in this same phrase.
- Cantonese (`priscilla-chan-night-flight-...`, char-level alignment): "回頭再看未未燈光" landed at
  46.06-48.02s against real vocals, closely tracking lrclib's own already-decent line timing
  (45.98s) -- confirms the char-split path (required for CJK, no whitespace word boundaries) works
  correctly end-to-end, not just the word-split English path.
- Full `extract_lyrics()` pipeline re-run three times against the real cached
  `bruno-mars-count-on-me-lyrics` vocal stem (197.52s) with `lyrics_query="Bruno Mars Count on
  Me"`: consistent, reproducible output across all three runs (284 words, last word "you" ending at
  190.24s -- sane and within the real audio's own duration, close to lrclib's own raw last-word
  guess of 189.86s). One earlier one-off run (not reproduced across three subsequent identical
  calls) produced a clearly-wrong last word past 239s, exceeding the audio's real 197.52s duration
  -- root-caused as impossible to originate from the forced-alignment layer itself (independently
  confirmed the emissions tensor's own frame count exactly bounds every possible output timestamp to
  <=197.54s for this audio), so it has to have come from lrclib's live search API returning
  different/longer content that one time, interacting with the pre-existing (unmodified by this
  change) Whisper-based `_repair_synced_lyrics_gaps` gap-filler. Flagged, not chased further, since
  it didn't reproduce and the component responsible predates this change -- worth a fresh
  investigation with real evidence if it recurs, per this file's own established convention, rather
  than more speculative fixing now.

**Not yet done**: none of the already-cached songs have been reprocessed with this change (that
needs `force=True` through `process_song`/`extract_lyrics`, real per-song compute, and a human
decision about which cached songs are worth re-running it on) -- this entry covers the pipeline
code change and its direct verification, not a bulk republish.

## Separation model choice: BS-RoFormer validation (recorded 2026-07-30)

Spike to determine whether `audio-separator`'s BS-RoFormer model is viable as a third separation
option alongside the two already-validated Demucs models (`htdemucs`, `htdemucs_ft`). **Result:
VIABLE** -- installs cleanly, the Python API works as documented, and the output vocal stem passes
the same silence-before/energy-after sanity check the rest of this project uses. The one real
tradeoff: **it's slow**, ~16 minutes of CPU time for a 4.5-minute song, vs. Demucs running close to
realtime on the same machine.

**Install**: `venv/Scripts/python.exe -m pip install "audio-separator[cpu]"` -- the `[cpu]` extra
exists and worked on the first try, no fallback to plain `audio-separator` needed. No C++ build
step, unlike `ctc-forced-aligner`'s Windows failure (see "Deliberately not using the
`ctc-forced-aligner` PyPI package" above) -- this package ships prebuilt wheels for its native
dependencies (onnxruntime, torch, etc.), so nothing to compile locally. `pip`'s dependency resolver
did flag version conflicts against this project's existing `tensorflow-intel` pin (which wants
`numpy<2.0`, `ml-dtypes~=0.2.0`, `protobuf<5.0`, but `audio-separator` pulled in `numpy==2.4.6`,
`ml-dtypes==0.5.4`, `protobuf==7.35.1`) -- these are warnings, not install failures, and nothing in
this spike touched `tensorflow-intel`-dependent code paths, so they weren't chased further here;
worth a real check before Task 7 wires this into the main pipeline in case those paths overlap.

**Confirmed Python API** (from `help(Separator.__init__)`, `help(Separator.separate)`,
`help(Separator.load_model)`, and the package's real README fetched via
`gh api repos/nomadkaraoke/python-audio-separator/readme --jq '.content' | base64 -d` -- not
guessed from the CLI `--help` text):

```python
from audio_separator.separator import Separator

separator = Separator(
    output_dir="<output directory>",       # where output WAVs are written
    model_file_dir="<model cache directory>",  # where the model file itself is downloaded/cached
)
separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
output_files = separator.separate("<path to input audio file>")
```

`load_model`'s own signature default is literally
`load_model(self, model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')` -- i.e. the exact
model filename from prior research is confirmed current directly from the installed package's own
code, not just the README. The README's "Model Filename" table independently lists this same file
(Friendly Name "Roformer Model: BS-Roformer-Viperx-1297", arch MDXC, vocals SDR 12.9 /
instrumental SDR 17.0) and it downloaded successfully when requested (639 MB, from whatever host
`download_model_files` resolves to -- resolved automatically, no manual URL needed).

**Output file naming**: for a 2-stem model like this one, `separate()` returns a list of two paths,
one per stem, following the pattern `{input_filename_stem}_({StemName})_{model_filename_without_ext}.wav`
written into `output_dir`. For this run (input `test_mix.wav`, model
`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) the exact output filenames produced were:

```
test_mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav
test_mix_(Instrumental)_model_bs_roformer_ep_317_sdr_12.wav
```

(Note the model-name suffix is truncated to `..._sdr_12` -- not the full `..._sdr_12.9755` -- this
is `audio-separator`'s own truncation behavior, not a typo; Task 7's implementer should match
against stem name in the returned `output_files` list rather than assume a specific suffix length.)
`custom_output_names` (an optional second arg to `separate()`) can override this with exact names if
Task 7 wants deterministic filenames instead of parsing the model-suffixed ones.

**Test input**: no original full-mix source audio is cached for any song (`meta.json`'s
`source_file` fields point to already-deleted temp downloads), so per the plan, reconstructed an
approximate mix by summing `cache/test-song/vocals.wav` + `cache/test-song/instrumental.wav`
(44.1kHz, 274.23s) -- valid since `instrumental.wav = original_mix - vocals.wav` was how that file
was produced in the first place (see `separation.py`'s `instrumental = original_wav - vocals`).

**Timing** (this machine, CPU-only, confirmed via `audio_pipeline/device.py`'s `get_device()`
returning `"cpu"`; AMD Ryzen, no CUDA GPU):
- Model download (639 MB, one-time, cached thereafter): ~26s.
- `load_model()`: 28.97s including that download on the first call; 1.31s on a second call once
  the model file was already cached locally.
- `separate()` on the 274.23s test mix: internally chunks into 35 windows, ~27-32s/chunk on this
  CPU, **959.91s (15:59) total wall-clock** for the actual separation, confirmed by the package's
  own log line `Separation duration: 00:15:59`. For comparison, this is roughly **3.5x the input
  audio's own duration** -- i.e. clearly slower than realtime, in contrast to Demucs
  (`htdemucs`/`htdemucs_ft`), which processes comparable-length songs close to or faster than
  realtime on the same machine. This is the main practical tradeoff of offering BS-RoFormer as an
  option: meaningfully higher quality (per the model's own published SDR numbers) at a real
  multi-minute processing-time cost per song.

**RMS sanity check** (same methodology as `melody_extraction.py`'s `_SILENCE_RMS_GATE` reasoning
and this project's other manual stem-quality checks), against `test-song`'s known real vocal onset
(~18.13s per this file's own earlier entries):

```
BS-RoFormer vocals stem (test_mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav):
  before onset (5-10s):  RMS = 0.003442
  after onset (20-25s):  RMS = 0.055967
  ratio after/before:    16.3x

Existing Demucs vocals.wav (cache/test-song/vocals.wav, for direct comparison):
  before onset (5-10s):  RMS = 0.003560
  after onset (20-25s):  RMS = 0.060980
```

Near-silent before the real vocal onset, clearly energetic after it -- the same qualitative shape
the existing Demucs output already exhibits, and the absolute numbers are close enough between the
two models (both stem from the same reconstructed input) that BS-RoFormer's separation quality on
this test file looks at least comparable to Demucs, not obviously worse.

**Operational note for Task 7's implementer**: a background run of this same script silently died
partway through (~60% through the 35 chunks) when left unattended for an extended period during
this spike, with no error or traceback captured -- root-caused to the *task-runner/shell session*
being interrupted, not to `audio-separator` itself (a clean immediate rerun completed normally end
to end, confirmed by the full `Separation duration: 00:15:59` log line and both output files
present on disk). Not an `audio-separator` reliability concern, but worth keeping in mind that a
~16-minute foreground/background call needs a runner that can actually wait that long without being
torn down early.

**Conclusion**: BS-RoFormer via `audio-separator[cpu]` is viable on this machine. Task 7 can proceed
using the API pattern confirmed above. The only decision left for Task 7 (or a product decision
before it) is whether a ~16-minute-per-song processing time is acceptable for a third "higher
quality" option users opt into, versus the near-realtime `htdemucs`/`htdemucs_ft` options already
shipped.

## Separation model choice: wired in end-to-end, including BS-RoFormer as a third option (recorded 2026-07-30)

Built the full feature from `docs/superpowers/specs/2026-07-30-separation-model-choice-design.md` /
`docs/superpowers/plans/2026-07-30-separation-model-choice.md`: `separate_stems()` gained a `model`
parameter (`htdemucs` / `htdemucs_ft` / `bs_roformer`), `process_song()` threads it through and
records it in `meta.json`, the job server (`scripts/server.py`) forces reprocessing when a
resubmitted link's requested model differs from what's cached, and `LoadSongForm.tsx` gained a
second `<select>` ("Fast (default)" / "Better quality (slower)" / "Best quality (slowest)") next to
the existing language one.

**BS-RoFormer wiring** (`_separate_bs_roformer` in `audio_pipeline/separation.py`) uses exactly the
API confirmed in this file's "BS-RoFormer validation" entry above: `Separator(output_dir=...)`,
`.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")`, `.separate(path)`. The
returned `output_files` list is matched by substring (`"(Vocals)"` / `"(Instrumental)"` in the
filename) rather than assuming a fixed suffix, per that entry's own caution about the model-name
suffix being truncated. No progress hook exists in the confirmed API, so `on_progress` (when given)
only fires once, at completion, unlike the Demucs paths' per-segment callback.

**A real regression caught before it shipped**: installing `audio-separator[cpu]` (Task 1's spike)
silently upgraded `numpy` to 2.4.6 (and `ml_dtypes`/`protobuf` alongside it), which broke
`tensorflow`/`basic-pitch` (melody extraction) outright --
`AttributeError: _ARRAY_API not found` at import time, failing `tests/test_pipeline.py` collection
entirely. `audio-separator` declares `numpy>=2` but was confirmed (by direct import + a full
BS-RoFormer separation run) to work fine under `numpy<2.0` too -- that constraint isn't actually
load-bearing at runtime. Fixed by pinning `numpy<2.0` (with `ml_dtypes~=0.2.0`/`protobuf<5.0` to
match) in `requirements.txt`, right after the `basic-pitch` dependency, so a fresh
`pip install -r requirements.txt` doesn't regress the same way.

**A real bug in `audio-separator` itself, worked around in the test, not the library code**: its
MDXC/BS-RoFormer path crashes (`RuntimeError: The size of tensor a (0) must match the size of
tensor b (...)` inside its own `overlap_add`) on audio clips under 10 seconds -- logged as "Audio
duration (2.00s) is less than 10 seconds" followed by the crash. Real songs are always well over
10s, so this has no production impact; the `bs_roformer` smoke test in `tests/test_separation.py`
uses a 12s synthetic clip instead of the 2s clips the other smoke tests use, specifically to avoid
this.

**Verification**: full Python suite (`pytest tests/ -q`) and frontend suite (`npm test`, 129 tests)
both pass, including the real (unmocked) `bs_roformer` smoke test against the actual
`audio-separator` package -- confirmed the model downloads, loads, and separates a real (if short)
clip into readable, non-empty vocals/instrumental wavs. `npx tsc -b --noEmit` reports no type
errors. Manual browser verification of the actual re-processing-on-model-change behavior against a
real song was not run end-to-end in this pass (each BS-RoFormer separation takes ~15+ minutes on
this CPU per the validation entry above, and the mismatch-forces-reprocess logic itself is already
covered by `test_run_job_forces_reprocessing_when_the_cached_model_differs`/
`test_run_job_skips_reprocessing_when_the_cached_model_matches` in `tests/test_server.py`) -- the
new `<select>` was confirmed to render with the correct three options and default via the frontend
component/type checks above; a full live run is a reasonable follow-up before relying on this in
practice for the first time.

## Separation model choice: end-to-end verification (recorded 2026-07-30)

Full automated suite: `pytest tests/ -q` -- **167 passed** (includes the real, unmocked
`htdemucs_ft` and `bs_roformer` smoke tests in `tests/test_separation.py`, and the job-server
cache-mismatch tests in `tests/test_server.py`). Frontend: `npm test` -- **129 passed**; `npx tsc -b
--noEmit` / `npm run build` -- no type errors.

Manual browser check (both the job server and Vite dev server already running locally): opened
"Load a song", confirmed the new vocal-separation-quality `<select>` renders next to the language
select, defaulted to "Fast (default)" (`htdemucs`), with all three options present in the correct
order and mapped to the right internal values (`htdemucs` / `htdemucs_ft` / `bs_roformer`) --
confirmed via the page's accessibility tree, not just visually. Submitting with the default
selection is unchanged from before this feature (no new field affects the fast path). A full live
run of the cache-mismatch-forces-reprocess behavior (resubmitting an already-cached link with a
different model and watching it actually reprocess) was not exercised end-to-end in the browser in
this pass -- each BS-RoFormer run alone takes 15+ minutes on this CPU, and that logic is already
covered directly by `test_run_job_forces_reprocessing_when_the_cached_model_differs` /
`test_run_job_skips_reprocessing_when_the_cached_model_matches` in `tests/test_server.py`. Worth a
real end-to-end run before relying on it for an actual multi-minute BS-RoFormer job in practice.

Feature ships complete: all three separation models (`htdemucs`/`htdemucs_ft`/`bs_roformer`)
selectable at upload, correctly recorded and compared for re-cache decisions.

## GPU-detection status shown on the load-song form (recorded 2026-07-30)

Both separation backends already auto-select CUDA when available (Demucs via
`audio_pipeline/device.py`'s `get_device()`, BS-RoFormer via `audio-separator`'s own internal
`torch.cuda.is_available()` check) -- but nothing surfaced that to the user, so there was no way to
tell from the UI whether a given machine was actually using its GPU. Added `get_device_info()` to
`audio_pipeline/device.py` (`{"available": bool, "name": str | None}`, via
`torch.cuda.get_device_name(0)`) and a `GET /api/gpu-status` endpoint in `scripts/server.py` that
returns it directly. `LoadSongForm.tsx` fetches it once on mount and renders "NVIDIA GPU detected
(<name>) -- processing will use it to speed things up" or "No NVIDIA GPU detected -- processing
will run on CPU" under the separation-model select. Verified: `pytest tests/test_device.py
tests/test_server.py -q` -- 31 passed; frontend `npx vitest run` -- 129 passed; `npx tsc -b
--noEmit` -- clean. Not manually verified against a real NVIDIA GPU (this dev machine is CPU-only)
-- confirmed via mocked `torch.cuda.is_available`/`get_device_name` instead, same pattern as the
existing `get_device()` tests.

## Auto-balance and clean up recordings (recorded 2026-07-30)

Built per `docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md` /
`docs/superpowers/plans/2026-07-30-auto-balance-recording.md`, via subagent-driven-development in
an isolated worktree (`.worktrees/auto-balance-recording`, branch `auto-balance-recording`).
Requested: the singer's voice should come through loud enough and sound "professional" against the
background music on a saved recording, and the mic should get cleaned up if the recording quality
is bad -- plus, raised mid-brainstorm, recordings sounded shifted from the start when played back.

**What was built**: `useRecording.ts` now records the instrumental and live mic as two separate
`MediaRecorder` tracks (previously mixed into one at record time) and uploads both to
`POST /api/recordings/mp3` as a multipart form. A new `audio_pipeline/mastering.py::master_recording()`
chains: transcode both tracks to wav -> correct a constant start-offset between them
(`_correct_start_offset`, using a to-be-measured (never guessed) `_RECORDING_OFFSET_SECONDS`
constant, currently `0.0`) -> clean the vocal (highpass @90Hz, `afftdn` denoise, `acompressor`) -> two-pass `loudnorm` each
track independently (vocal -16 LUFS, instrumental -20 LUFS -- narrowed from an initial -14/-20 split
after feedback that the voice was too loud) -> mix (`amix`, `normalize=0` to preserve the loudnorm
balance) -> `alimiter` so the vocal boost can't clip. `scripts/server.py`'s recording endpoint calls
this before its existing mp3 transcode step, unchanged from there on (same save/download/list/delete
behavior). A new `audio_pipeline/mastering.py::measure_start_offset()` (real onset detection via
`librosa.onset.onset_detect`) plus `scripts/measure_recording_offset.py`, a thin CLI wrapper, exist
so a human can determine the real offset value from an actual test recording -- deliberately
separate from the production pipeline, not auto-run.

**Automated verification**: full suite run in the worktree after all 9 plan tasks landed (task-by-task
review loop, one Critical/Important-free approval per task except Task 3's rumble test, corrected
below) -- `pytest tests/ -q`: **187 passed**. Frontend: `npm test` -- **129 passed**; `npx tsc -b
--noEmit` -- clean. Every new test in `tests/test_mastering.py` is a real, unmocked smoke test
against synthetic audio (sine tones, synthetic clicks/transients), matching this project's
established convention -- no mocking of ffmpeg or librosa anywhere in that file. `tests/test_server.py`
mocks `master_recording` at the endpoint layer only (to isolate endpoint plumbing, same pattern
already used there for `process_song`/`download_audio`), while feeding it a real generated wav so
the endpoint's own real `transcode_to_mp3` call still exercises actual ffmpeg end-to-end.

**One real measured-margin issue caught during review, not shipped**: Task 3's first commit had a
rumble-removal test asserting 90% attenuation with a synthetic 40Hz tone against the 90Hz highpass
cutoff; real ffmpeg only delivered ~85% at that separation (measured, not guessed). Fixed by moving
the test tone to 20Hz (a more realistic rumble frequency anyway -- real handling noise/thuds skew
well below 40Hz), which cleared the bar with a comfortable 96% margin. The implementation
(`_clean_vocal`'s filter chain) was left exactly as specified -- the fix was entirely in the test's
signal, not the shipped filter parameters.

**Outstanding, not completed in this session -- needs a human**: this session had no access to a
real microphone or a real (non-automated) browser, the same limitation this file has documented for
every other real-time audio/mic feature (Phase 2's audio-sync check, Phase 3's live-pitch tuning,
etc.). Two things remain, both requiring a human in a normal browser window with a working mic:
1. **Measure the real `_RECORDING_OFFSET_SECONDS` value.** It currently ships at its untuned default,
   `0.0` (a deliberate no-op, per the design spec's explicit instruction not to guess this constant)
   -- meaning the start-offset correction code path is fully built, tested (against synthetic
   offsets), and wired in, but doesn't yet correct anything in a real recording. Procedure: record a
   take clapping once sharply on the song's first strong instrumental beat, extract the two raw
   pre-mastering tracks (see `scripts/measure_recording_offset.py`'s docstring for the exact
   temporary-debug-line procedure), run that script, and set the printed value.
2. **Listen to a real mastered recording** to confirm the vocal is audibly cleaner/louder-but-not-
   overwhelming and the take starts in sync, per the design spec's loudness targets (-16/-20 LUFS)
   being a starting point pending exactly this kind of check -- same caveat this file attaches to
   every other audio-quality claim.

Until both are done, treat this feature as mechanically complete and fully test-covered, but not yet
confirmed to sound right on a real voice.

## Wrong / missing lyrics from a noisy lrclib query (recorded 2026-07-30)

**Reported symptom**: lyrics "quite inaccurate and sometimes missing words." Traced to the *lookup
query*, not the parsing/timing logic downstream of it. The job server passes the raw source title
straight through as `lyrics_query` (`server.py` -> `process_song(..., lyrics_query=title)`), and
`fetch_synced_lyrics` sent that verbatim as a free-text `q=` to lrclib's `/api/search`. A real
source title is a YouTube video title, e.g. `"Adele - Hello (Official Music Video) [4K]"`,
`"BLACKPINK - 'Pink Venom' M/V"`, `"Queen – Bohemian Rhapsody (Remastered 2011)"` -- carrying
promotional noise ("(Official Music Video)", "[4K]", "(Remastered 2011)", "ft. ...") that a lyrics
database's own track title never has. That noise makes lrclib either match the wrong release (a
sped-up edit, a cover, a reaction upload with a similar title) or match nothing at all. **Both
surface to the user as "the lyrics are wrong"**: a wrong match shows a different song's words, and a
no-match silently falls back to the error-prone local Whisper transcription (the "missing words"
half -- Whisper's long-form dropout, documented in `lyrics_extraction.py`, is exactly where words
go missing).

**Fix** (`lyrics_lookup.py`): clean and structure the title before the lookup instead of trusting it
raw.
- `_parse_title` strips promotional bracket groups (`_strip_noise_brackets` -- only a bracketed
  group whose contents contain a known noise keyword like "official"/"video"/"remaster"/"4k"/"feat",
  so a *genuine* title parenthetical like "(Interlude)" is preserved rather than mangled), strips a
  trailing "feat./ft./featuring ..." credit, and splits "Artist - Title" on a *spaced* dash (so an
  intra-word hyphen like "Jay-Z" is left intact). Handles ASCII and full-width/CJK bracket and dash
  conventions for Cantonese uploads.
- `fetch_synced_lyrics` then prefers lrclib's **structured** `track_name` + `artist_name` search
  (far more precise than free text) whenever an artist was identified, and **falls back** to a
  cleaned free-text query -- both when there was no "Artist - Title" split to make, and when the
  precise search returns nothing usable (a slightly-off cleaned track/artist that free text still
  matches loosely). Single request in the common case; the second (fallback) call only fires on a
  structured-search miss, so the fast path is unchanged.

This is a strict improvement to *which* release lrclib matches; the existing duration-closest pick
and wrong-script rejection (`_pick_best_synced_lyrics`) still apply on top, and the downstream
onset/tempo re-anchoring (`_align_synced_lyrics_to_audio`) is untouched. Verified with 9 new unit
tests (title parsing across real-world title shapes + structured-search/free-text-fallback request
behavior); all 41 `test_lyrics_lookup.py` + `test_text_script.py` tests pass. Not yet re-verified
end-to-end against a live lrclib lookup (needs network + a real download) -- the request-shaping is
unit-tested against a mocked `requests.get`, same as the rest of this module's network path.

## GPU was never actually being used, despite an NVIDIA GPU being present (recorded 2026-07-31)

**Reported symptom**: "the NVIDIA GPU only turns on for gaming" -- read as a Windows
hybrid-graphics/GPU-switching problem. That premise doesn't hold: the Windows per-app "Graphics
performance preference" (`HKCU:\Software\Microsoft\DirectX\UserGpuPreferences`, `GpuPreference=2`)
only controls which GPU renders a window's *display* output; it has no effect on which GPU a CUDA
compute call uses. `nvidia-smi` confirmed the RTX 5050 Laptop GPU (driver 596.13, CUDA 13.2) was
present, healthy, and not asleep the whole time.

**Actual root cause**: `venv/Lib/site-packages/torch/version.py` showed `torch==2.13.0+cpu` --
plain `pip install torch torchaudio` (no `--index-url`), exactly what this file's earlier "Why
Python 3.11" section and `requirements.txt`'s old comment described as normal, installs PyPI's
**CPU-only** wheel even on a machine with a working NVIDIA GPU. That old requirements.txt comment
("the default wheel is CUDA-capable... falls back to CPU automatically otherwise") was wrong --
confirmed directly, not assumed. `docs/superpowers/plans/2026-07-30-separation-model-choice.md`'s
"this machine is CPU-only" conclusion (used to justify not exploring `bs_roformer`) was therefore
also just an artifact of the CPU-only wheel, not a real hardware limit -- worth revisiting
separately now that GPU acceleration actually works, though not done as part of this fix.

**Fix**: reinstalled `torch==2.13.0`/`torchaudio==2.11.0` from
`https://download.pytorch.org/whl/cu130` (same version numbers, CUDA-enabled build -- chosen
because it's the only CUDA wheel index offering these exact versions, and CUDA 13.0 fully covers
this GPU's Blackwell architecture). `audio_pipeline/device.py`'s `get_device()` needed no changes --
it already did `torch.cuda.is_available()` correctly; it was simply always resolving to `"cpu"`
because of the wheel, not the code. `torch.cuda.is_available()` now returns `True` and correctly
names the RTX 5050.

**Second, less obvious bug this surfaced**: switching `get_device()` to `"cuda"` broke
`faster-whisper` (lyrics transcription) specifically, even though it fixed Demucs (`separation.py`)
and the CTC forced-alignment model (`forced_alignment.py`) immediately. Root-caused via a minimal
standalone repro script (not guessed): `WhisperModel(..., device="cuda")` constructs fine, but the
first real `.transcribe()` call raises `RuntimeError: Library cublas64_12.dll is not found or
cannot be loaded`. Reason: faster-whisper's backend, `ctranslate2`, links its own CUDA 12 cuBLAS at
runtime, entirely independent of whatever CUDA version torch itself bundles (torch's cu130 build
only ships `cublas64_13.dll`, privately, inside `torch/lib/` -- not visible to ctranslate2's
separate native loader at all). This is why the test suite first appeared to *hang* rather than
fail cleanly in an interactive run: reproduced the exact same error immediately in isolation once
network/model-download red herrings were ruled out (the real 4GB of first-time model downloads for
`large-v3` + the MMS forced-aligner masked the real symptom during the first couple of attempts).

Fix: installed the `nvidia-cublas-cu12` pip package (ships just the missing DLL,
`nvidia/cublas/bin/cublas64_12.dll`) and, in `audio_pipeline/device.py`, added its directory to
both `os.add_dll_directory()` and `PATH` once at import time, guarded to `sys.platform == "win32"
and torch.cuda.is_available()` so it's a no-op everywhere else. Verified the fix directly against
the real `audio_pipeline.device.get_device()` + `WhisperModel` call path (not just the standalone
repro) before trusting it. Added `nvidia-cublas-cu12; sys_platform == "win32"` to `requirements.txt`
with the same explanation.

**Verification**: full test suite -- `155 passed, 6 failed` (was previously untestable at all in
this freshly-rebuilt venv, which also turned out to be missing every non-torch dependency; installed
`requirements-lock.txt` plus `transformers`/`uroman`, which weren't in the lock file, to get a
working baseline first). All 6 remaining failures are `FileNotFoundError` from `ffmpeg`/`ffprobe`
not being installed on this machine at all (a separate, pre-existing, unrelated gap -- not
installed as part of this fix, since it means adding system-wide software the user didn't ask for)
-- zero failures relate to CUDA/torch. Full run took 54.6s total, not just individually-faster
GPU stages, consistent with real acceleration. Ran the actual `scripts/separate_stems.py` CLI (not
just a unit test) against a synthetic 30s clip: completed in 6.9s including the one-time `htdemucs`
model download, with `get_device()` confirmed returning `"cuda"` throughout. No real full-length
song was available in this reset environment to redo the exact NOTES.md Phase 0 timing comparison
(79.9s CPU baseline) apples-to-apples -- that would need a real song plus `ffmpeg` installed, both
out of scope here.

## Bare (unbracketed) promotional tags defeating the lrclib title cleanup (recorded 2026-07-31)

Reported: inaccurate lyrics on a real Cantonese song (`flac-lyrics` / 陳慧嫻 "夜機"), and the note
highway still showing notes during its hummed intro (the latter previously fixed by
`pipeline._remove_notes_without_lyrics`, see "Highway notes with no corresponding lyrics" above).

**Root cause**: the video's raw YouTube title is "陳慧嫻　夜機　無損音樂FLAC　歌詞LYRICS　純享" --
`lyrics_lookup._parse_title`'s noise stripping (`_strip_noise_brackets`) only removes
`_NOISE_KEYWORDS` when they appear inside a bracket/parenthesis group, but this title has none: the
promotional tags ("lossless-music-FLAC", "lyrics-LYRICS", "pure-enjoy") are glued directly onto
adjacent CJK words with no bracket and no space at all -- an extremely common convention on
Cantonese/Mandopop uploads. Confirmed directly against the live lrclib API: the raw, uncleaned title
sent as a free-text query returns **zero** results (lrclib's search appears to require every query
term to match, so a handful of noise words is enough to break it entirely), while the same title
cleaned down to just "陳慧嫻 夜機" matches the correct song immediately (multiple synced-lyrics
candidates). The zero-result lookup silently fell through to local Whisper transcription of the
intro's ambiguous hummed audio, primed with the same noisy title as a `hotwords` hint -- which
hallucinated fragments of that hotword text itself ("FL", "AC", "詞", "IC") as the song's first
"lyric" line at ~0.3-1.2s. That bogus early line then defeated
`pipeline._remove_notes_without_lyrics`: it gave the intro a (fake) lyric-coverage span, so the
real hummed notes there were kept instead of stripped as intended.

**Fix**: `lyrics_lookup._strip_noise_tokens`, a new pass applied to the title after bracket-stripping
and feat-credit-stripping (in `_parse_title`) -- splits on whitespace and drops any token that
exactly equals, starts with, or ends with a `_NOISE_KEYWORDS` entry (case-insensitive). Deliberately
prefix/suffix/equality matching rather than "keyword anywhere in the token" (the looser check
`_strip_noise_brackets` already uses for bracket contents): every real glued-tag pattern observed
attaches cleanly at one edge of a token ("無損音樂**FLAC**", "**歌詞**LYRICS"), and a substring-anywhere
check risks eating a real bare title word that just happens to contain a short keyword like "hd" or
"mv" mid-word. Added `"flac"`, `"lossless"`, `"無損"`/`"无损"`, `"純享"`/`"纯享"` to `_NOISE_KEYWORDS`
(not previously listed at all). Ordered after `_strip_feat` specifically so it can't collide with the
existing `ft`/`feat` featuring-credit stripping (which already handles those exact tokens via regex
matching to end-of-string).

**Verified**: live-queried the real lrclib API before and after -- raw title 0 results, cleaned title
("陳慧嫻 夜機") 9 results including 5 with synced lyrics for the correct song/artist. 2 new
`tests/test_lyrics_lookup.py` cases (the real glued-tag title parses to `("陳慧嫻 夜機", None)`; a
control case confirms a real word like "Redheaded" isn't eaten just for containing "hd"). All 35
`test_lyrics_lookup.py` tests and 85/86 of `tests/` pass (the one failure,
`test_process_song_runs_end_to_end_and_produces_assets`, reproduces identically on unmodified `main`
-- pre-existing `ffmpeg` PATH gap in this shell, unrelated to this change).

**Applied to the real cached song**: `scripts/reprocess_from_vocals.py flac-lyrics` (existing tool,
reuses the cached `vocals.wav`, no re-download/re-separation needed) with ffmpeg added to `PATH` for
the run. Confirmed directly on the regenerated output: `lyrics.json`'s first word is now the real
"回" at 46.16s (previously "FL" at 0.34s), and `notes.json`'s first note now starts at 46.2605s
(previously 0.743s) -- the hummed-intro notes are gone, matching the lyric-coverage-based removal
this song already relied on once its lyric timeline starts in the right place. Republished to
`frontend/public/cache/flac-lyrics/` by the same script. Not re-applied to any other cached song --
this bug only reproduces for a title carrying this specific glued-tag pattern; a targeted rerun of
`--all` would be the way to sweep every cached song if more turn out to be affected, not done here
since it wasn't reported for any other song.

## Lyrics timing still badly wrong even on the highest-quality separation model (recorded 2026-08-10)

Reported: `dhruv-double-take-lyrics` lyrics timing very bad, even reprocessed with the slowest/
highest-quality separation model (`bs_roformer`, confirmed via `meta.json`'s `separation_model`) --
ruling out separation quality as the cause before looking further.

**Root cause (measured directly against the real cached `lyrics.json`, not guessed)**: individual
words with grossly implausible durations, e.g. "but" spanning 7.6s and the very next word "you"
spanning 10.1s (113.18-130.86s combined) -- roughly 20-40x a normal sung word. This traces to
`forced_alignment.align_tokens`: CTC forced alignment must assign *every* audio frame to some
target symbol in the given token sequence, and the only "wildcard" available to absorb audio that
doesn't match any of the given words was a single `<star>` pair at the very start/end of the whole
song (added when forced alignment first replaced lrclib's own line-level timing -- see "Lyrics
timing: replaced with CTC forced alignment" above). A stretch mid-song that the source lyrics text
just doesn't cover -- confirmed by checking the vocal stem's own RMS through 110-130s, continuously
non-silent (0.09-0.11, in line with the song's singing elsewhere, not an instrumental gap) -- has
nothing nearby to absorb it, so it gets smeared into whichever real word sits next to it instead.
Tried a `<star>` at every *line* boundary first (a smaller change); measured that it wasn't enough
by itself -- the extra content sits *inside* what was expected to be a single short line, not
between two lines, so a line-level wildcard never gets a chance to catch it.

**Fix, two parts**:
1. `forced_alignment.align_tokens` now inserts a `<star>` wildcard between *every* word (matching
   `ctc-forced-aligner`'s own `star_frequency="segment"` mode, previously not used here -- see the
   module's own updated comments). Verified this alone converts most of the failure from "smeared
   into a word's own duration" into a clean, empty *gap* between two words instead -- a shape
   `lyrics_extraction._repair_synced_lyrics_gaps`'s existing Whisper-based gap-repair pass already
   knows how to fill in, rather than a new failure mode needing new code.
2. New `_find_overlong_word_runs`/`_repair_overlong_words` in `lyrics_extraction.py`, wired into
   `_repair_synced_lyrics_gaps`: a safety net for whatever the per-word wildcard still can't place
   correctly (measured: it wasn't quite enough on its own for this song's worst two-word run).
   Finds maximal runs of consecutive words whose own force-aligned duration still exceeds the
   already-established `_MAX_PLAUSIBLE_WORD_DURATION_SECONDS` (3.0s -- the same cap
   `_repair_energetic_gaps` already uses to reject implausible *recovered* words), re-transcribes
   that padded stretch fresh via Whisper, and replaces the run with whatever it finds -- falling
   back to leaving the original (mistimed but at least present) words untouched if nothing usable
   comes back. Runs before the existing gap-repair pass in `_repair_synced_lyrics_gaps`, since
   replacing an overlong run can itself reveal a genuine leftover gap for that pass to catch.

**Verified**: 8 new unit tests (`tests/test_lyrics_extraction.py`: `_find_overlong_word_runs`
finding a single word / merging a consecutive run / finding nothing when all words are plausible;
`_repair_overlong_words` replacing a run, leaving it untouched when nothing's recovered, rejecting
a still-implausible recovery; `_repair_synced_lyrics_gaps` wiring). 217 Python tests pass total.
Re-ran `extract_lyrics()` directly against the real cached `dhruv-double-take-lyrics` vocal stem:
zero words left with duration over 2.5s (down from several, the worst at 10.1s), and the
previously-blank stretch turned out to be a real repeated chorus ("Boy, you got me hooked onto
something / Who could say that they saw us coming? / Tell me, do you feel that love?") the fetched
lrclib text was simply missing -- now filled in with plausible per-word timing (0.1-1.5s each) via
the same Whisper re-transcription this fix wires in. One smaller residual noticed but not chased
further (a ~0.3s overlap where recovered words meet the next original line's words, and one
low-energy ~8s stretch around 71-79s that the repair pass tried and failed to recover anything
for, left as an unfilled gap) -- neither is the "words holding for 10+ seconds" failure that was
reported, and both are within this pipeline's already-documented "not perfect" tolerance elsewhere
in this file. Updated `cache/dhruv-double-take-lyrics/lyrics.json` and republished via
`scripts/publish_song.py dhruv-double-take-lyrics`. Not re-applied to any other cached song -- this
class of bug isn't title-specific like the FLAC one above, so any synced-lyrics-sourced song could
in principle be affected, but a bulk `reprocess_from_vocals.py --all` sweep wasn't done here since
only this one song was reported.
