# Separation model choice at upload

## Context

Earlier research (see NOTES.md's "Lyrics timing: replaced with CTC forced alignment" entry and
the conversation that preceded it) identified two independent levers for reducing spurious
notes/bleed in the note highway: (1) more accurate lyric timing, already shipped via CTC forced
alignment, and (2) higher-quality vocal/instrumental separation. This spec covers (2): letting the
user choose which separation model to use when loading a song, instead of always using the
current fixed `htdemucs` model.

## Goal

Add a model-choice option to the "load a song from a link" flow, threaded end-to-end from the
upload form through the job server into the pipeline's separation stage.

## Non-goals

- Not changing anything about melody extraction, lyrics, or scoring.
- Not reprocessing already-cached songs automatically (a user who wants a different model for an
  existing song re-submits the link with a new model choice; see "Caching behavior" below).
- Not building a general "pluggable separation backend" abstraction beyond what's needed for these
  three specific models.

## Model choices and labels

Three options, shown in `LoadSongForm` with plain-language labels (friendlier for a personal app,
no need to know what "htdemucs" or "BS-RoFormer" are):

| Internal id   | UI label                    | Backend              | New dependency? |
|---------------|------------------------------|-----------------------|------------------|
| `htdemucs`    | "Fast (default)"             | Demucs, existing model | No |
| `htdemucs_ft` | "Better quality (slower)"    | Demucs, fine-tuned bag-of-4-models variant | No |
| `bs_roformer` | "Best quality (slowest)"     | `audio-separator` package (BS-RoFormer model) | Yes |

Default is `htdemucs` — no behavior change for anyone who doesn't touch the new control.

`htdemucs`/`htdemucs_ft` are both already available in the `demucs` package this project already
depends on (just a different model name string to `demucs.api.Separator`). `bs_roformer` requires
a new dependency (`audio-separator`) that has not been validated on this machine yet.

## Risk: BS-RoFormer is an unvalidated dependency

Same category of risk as `ctc-forced-aligner` turning out to need a C++ compiler this Windows
machine doesn't have (see NOTES.md's forced-alignment entry). Before any UI/plumbing work, a
standalone validation spike:

1. `pip install audio-separator` (or whichever extra/variant is needed) into the venv, confirm it
   imports and its BS-RoFormer model downloads/loads.
2. Run it against a real cached vocal source (e.g. re-separating `cache/test-song`'s original
   extracted-audio-equivalent, or any available full-mix audio) and sanity-check the output vocal/
   instrumental split sounds/measures right (same kind of RMS/energy sanity check used throughout
   this project's other validations).
3. Confirm its Python API shape (the CLI form `audio-separator input.wav --model_filename ...` is
   known from research; the Python API needs direct confirmation, same as forced-alignment's
   Python API had to be confirmed against the README before trusting it).

**Fallback**: if BS-RoFormer doesn't pan out (build/install failure, bad output, API too
unstable), ship only `htdemucs`/`htdemucs_ft` as the two choices, drop `bs_roformer` from the UI
and server-side `Literal`, and leave a NOTES.md entry documenting why, exactly like every other
"tried it, didn't work, here's the evidence" entry already in that file. BS-RoFormer becomes a
follow-up rather than blocking this feature.

## Architecture / data flow

```
LoadSongForm (model <select>, default "Fast (default)")
  -> POST /api/jobs {url, language, separation_model}
    -> server.py: compare requested model against cached meta.json's
       recorded model; if it's a cache hit but the model differs, force
       reprocessing
      -> process_song(..., separation_model=..., force=<see below>)
        -> separate_stems(..., model=...)
          -> "htdemucs" / "htdemucs_ft": existing Demucs code path, model
             name is now a parameter instead of a fixed module constant
          -> "bs_roformer": new backend via `audio-separator`
```

## Backend changes

**`audio_pipeline/separation.py`**
- `separate_stems(input_path, output_dir, model: str = "htdemucs", on_progress=None)`.
- `htdemucs`/`htdemucs_ft`: unchanged Demucs call, `_MODEL_NAME` becomes the `model` argument
  instead of a fixed constant. `htdemucs_ft` is a bag of 4 models (~4x Demucs's own inference
  time) -- no code difference, just a different string handed to `demucs.api.Separator(model=...)`.
- `bs_roformer`: new function (e.g. `_separate_bs_roformer`), shaped once the validation spike
  confirms the real API. Must still return `(vocals_path, instrumental_path)` -- same contract as
  the Demucs path, since `pipeline.py`/`melody_extraction.py`/the game only care about those two
  files, not which backend produced them.
- Reject an unknown `model` value with a clear `ValueError` (mirrors `lyrics_extraction.py`'s own
  `Unsupported language` validation pattern) rather than silently falling back to a default.

**`audio_pipeline/pipeline.py`**
- `process_song()` gains `separation_model: str = "htdemucs"`, passed straight through to
  `separate_stems()`.
- `meta.json` gains a `"separation_model"` field, written alongside the existing `source_file`/
  `song_id`/`processed_at` fields, so a later request can tell what a cached song was actually
  separated with.

**`scripts/server.py`**
- `CreateJobRequest` gains `separation_model: Literal["htdemucs", "htdemucs_ft", "bs_roformer"] =
  "htdemucs"`.
- `Job` dataclass gains a matching `separation_model` field.
- `_run_job`: before the existing `is_cached(CACHE_DIR, slug)` short-circuit, also read the cached
  `meta.json`'s `separation_model` (defaulting to `"htdemucs"` for older cached songs that predate
  this field) and compare against the request. On a mismatch, treat it like `force=True` --
  reprocess and overwrite the cached separation (and everything downstream of it, since melody/
  lyrics both depend on the vocal stem).

## Caching behavior

- Re-submitting the same link with the **same** model as what's cached: unchanged fast-path
  behavior (no reprocessing).
- Re-submitting with a **different** model: reprocesses from separation onward, replacing the
  cached `vocals.wav`/`instrumental.wav`/`melody.mid`/`notes.json`/`lyrics.json` and updating
  `meta.json`'s `separation_model`. This uses the existing `process_song(..., force=True)` path --
  no new reprocessing mechanism needed, just correctly deciding when to pass `force=True`.
- Older cached songs with no `separation_model` in `meta.json` are treated as `htdemucs` (the
  model that was hardcoded before this feature existed) for comparison purposes.

## Frontend changes

**`frontend/src/components/LoadSongForm.tsx`**
- A second `<select>` next to the existing language one: "Fast (default)" / "Better quality
  (slower)" / "Best quality (slowest)", mapped to `htdemucs`/`htdemucs_ft`/`bs_roformer`
  internally. Disabled while a job is in progress, same as the existing language select.
- Included in the `POST /api/jobs` body as `separation_model`.

## Testing

- **Backend**: unit tests for `separate_stems`'s model dispatch (mocked Demucs/BS-RoFormer calls,
  one per model id, plus the unknown-model `ValueError`), `process_song`'s new parameter reaching
  `separate_stems` and `meta.json`, and `server.py`'s cached-model-mismatch-forces-reprocessing
  logic (and the same-model fast-path staying a no-op).
- **Frontend**: vitest coverage for the new select appearing in the request body, mirroring the
  existing language-select test in `LoadSongForm`'s test suite (if one exists) or `App`/
  `LoadSongForm` component tests.
- BS-RoFormer's actual separation quality is validated manually against a real song during the
  spike (per the Risk section) -- not something a fast unit test can meaningfully assert, same
  category as every other model-backed accuracy claim in this project (see NOTES.md throughout).

## Open questions resolved during brainstorming

- Default model: `htdemucs` (no behavior change by default).
- Re-cache behavior on model change: reprocess (not silently reuse stale separation).
- UI labels: plain-language, not technical model names.
- BS-RoFormer risk handling: validate first as its own step, before any UI/plumbing work.
