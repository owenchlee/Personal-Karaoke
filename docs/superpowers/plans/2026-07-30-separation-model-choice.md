# Separation Model Choice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick which vocal-separation model (htdemucs / htdemucs_ft / BS-RoFormer) is used when loading a song, threaded from the upload form through the job server into the pipeline's separation stage.

**Architecture:** `separate_stems()` gains a `model` parameter (already-installed Demucs checkpoints `htdemucs`/`htdemucs_ft` first; BS-RoFormer via a new `audio-separator` dependency once validated). `process_song()` threads it through and records it in `meta.json`. The job server compares a request's model against what a cached song was actually separated with, forcing reprocessing on a mismatch instead of silently reusing a stale separation. `LoadSongForm` gets a second `<select>` next to the existing language one.

**Tech Stack:** Python (FastAPI job server, `demucs` package, pytest), TypeScript/React (Vite, vitest).

## Global Constraints

- Default separation model is `htdemucs` — no behavior change for anyone who doesn't touch the new control (spec: "Default model").
- Re-submitting an already-cached song with a *different* model reprocesses it; same model stays on the existing fast path (spec: "Caching behavior").
- UI labels are plain-language ("Fast (default)" / "Better quality (slower)" / "Best quality (slowest)"), not technical model names (spec: "Model choices and labels").
- BS-RoFormer is validated as its own step before any UI/plumbing work touches it; if it doesn't pan out, the shipped feature is just the two Demucs options (spec: "Risk: BS-RoFormer is an unvalidated dependency").
- This project has no React component-testing setup (`@testing-library/react`/jsdom are not installed; verified no `.test.tsx` files exist anywhere in `frontend/src`, and the existing language `<select>` in `LoadSongForm.tsx` has no test coverage of its own either). Do not add new frontend test infrastructure for this feature — verify `LoadSongForm.tsx` changes manually in a real browser, matching how the existing language select was verified.
- `tests/test_separation.py`'s existing convention is real (unmocked) Demucs smoke tests against a synthetic clip, not mocks — follow that convention for `htdemucs`/`htdemucs_ft`, not a different pattern.

---

### Task 1: Validate BS-RoFormer via `audio-separator` (spike)

**Files:** none in the repo are modified by this task except `NOTES.md` (append-only, findings). All exploratory code goes in the scratchpad, not the repo.

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a written decision in `NOTES.md` — either (a) the confirmed `audio-separator` Python API (import path, constructor args, method name(s), and the exact output file-naming pattern it produces for a vocals/non-vocals split), which Task 7 consumes verbatim, or (b) a documented reason it doesn't work here, in which case Task 7 is skipped and the plan ends at Task 6.

This task is a research spike, not TDD — there is no test to write first because the thing being tested is "does this dependency and API even work on this machine," which can only be answered by trying it. Follow this exact procedure; do not guess at the Python API from memory.

- [ ] **Step 1: Install `audio-separator`**

Run: `venv/Scripts/python.exe -m pip install "audio-separator[cpu]"`

This machine is CPU-only (confirmed by `audio_pipeline/device.py`'s `get_device()`, which every other model-backed stage in this project already goes through). If the `[cpu]` extra doesn't exist or the install fails, retry with plain `audio-separator` and note in `NOTES.md` which one worked.

If this step fails outright (e.g. a C++ build error like the one `ctc-forced-aligner` hit — see `audio_pipeline/forced_alignment.py`'s module docstring and the matching `NOTES.md` entry), stop here, write the failure (exact error text) to `NOTES.md` under a new "BS-RoFormer separation: not viable on this machine" heading, and do not proceed to Task 7.

- [ ] **Step 2: Confirm the import and inspect the real API**

Run:
```
venv/Scripts/python.exe -c "from audio_separator.separator import Separator; help(Separator.__init__); help(Separator.separate)"
```

If `Separator.separate` doesn't exist under that name, run `python -c "from audio_separator.separator import Separator; print([m for m in dir(Separator) if not m.startswith('_')])"` to find the real method name, then `help()` on whatever it is. Also fetch the package's own README for the documented Python usage pattern (not just the CLI form) before writing any code against it:

```
gh api repos/nomadkaraoke/python-audio-separator/readme --jq '.content' | base64 -d
```

Write down (in a scratch note, not yet in the repo): the exact import path, constructor signature actually used, and the method call that runs separation, plus how to select the BS-RoFormer model specifically (a `model_filename` argument per the CLI form is the leading candidate from prior research, but confirm it against the real README/API, not the CLI `--help` text).

- [ ] **Step 3: Build a real audio file to separate**

There's no original full-mix source video cached anywhere in this repo (`cache/*/meta.json`'s `source_file` fields all point to already-deleted temp downloads). Reconstruct an approximate full mix from an already-cached song's own stems — since `instrumental.wav = original_mix - vocals.wav` (see `audio_pipeline/separation.py`'s existing `instrumental = original_wav - vocals` line), summing them back approximately reconstructs the original mix:

```python
# scratchpad script, e.g. <scratchpad>/build_test_mix.py
import soundfile as sf
import numpy as np

vocals, sr1 = sf.read("cache/test-song/vocals.wav")
instrumental, sr2 = sf.read("cache/test-song/instrumental.wav")
assert sr1 == sr2
n = min(len(vocals), len(instrumental))
mix = vocals[:n] + instrumental[:n]
sf.write("<scratchpad>/test_mix.wav", mix, sr1)
```

(Replace `<scratchpad>` with this session's actual scratchpad directory path.)

- [ ] **Step 4: Run BS-RoFormer separation against the reconstructed mix**

Using the confirmed API from Step 2, separate `<scratchpad>/test_mix.wav` with the BS-RoFormer model (`model_bs_roformer_ep_317_sdr_12.9755.ckpt` per prior research — confirm this filename is real and downloadable via whatever model-listing mechanism the package exposes, e.g. `Separator().list_supported_model_files()` or equivalent, rather than assuming it's still current). Note the wall-clock time and the exact output file(s) produced (names, how vocals vs. non-vocals/instrumental are distinguished in the output).

- [ ] **Step 5: Sanity-check the output against real energy, same methodology as the rest of this project**

```python
import soundfile as sf
import numpy as np

def rms(path, start_s, end_s, sr_hint=None):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seg = audio[int(start_s * sr):int(end_s * sr)]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))

# test-song's real vocal onset is ~18.13s (see NOTES.md's Phase 0/melody entries) --
# the BS-RoFormer vocal-stem output should be near-silent before that and clearly
# energetic after it, the same shape the existing Demucs vocals.wav has.
print("before onset:", rms("<output vocal path>", 5.0, 10.0))
print("after onset:", rms("<output vocal path>", 20.0, 25.0))
```

Confirm the "before" RMS is much lower than the "after" RMS (same qualitative check `melody_extraction.py`'s `_SILENCE_RMS_GATE` reasoning and this project's other manual verifications already use).

- [ ] **Step 6: Record the outcome in NOTES.md**

Append a dated entry ("## Separation model choice: BS-RoFormer validation (recorded <date>)") documenting: the exact working Python API (import, constructor, method call, model filename, output file naming), the measured timing, and the RMS sanity-check numbers from Step 5. If any step failed, document the exact failure instead and state explicitly that Task 7 is skipped and the feature ships with only `htdemucs`/`htdemucs_ft`.

- [ ] **Step 7: Commit**

```bash
git add NOTES.md
git commit -m "Record BS-RoFormer (audio-separator) validation findings"
```

---

### Task 2: `separate_stems()` gains a `model` parameter

**Files:**
- Modify: `audio_pipeline/separation.py`
- Test: `tests/test_separation.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `separate_stems(input_path, output_dir, model: str = "htdemucs", on_progress=None) -> tuple[Path, Path]`, raising `ValueError` for any `model` not in `("htdemucs", "htdemucs_ft")`. Task 3 consumes this signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_separation.py` (after the existing `test_separate_stems_produces_readable_wavs` test):

```python
def test_separate_stems_rejects_an_unknown_model(tmp_path):
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=2.0)

    with pytest.raises(ValueError, match="Unsupported separation model"):
        separate_stems(input_path, output_dir, model="not-a-real-model")


def test_separate_stems_accepts_htdemucs_ft(tmp_path):
    # htdemucs_ft is a fine-tuned bag of 4 models -- first run downloads all 4 checkpoints
    # (larger and slower than plain htdemucs's single checkpoint) and requires internet access;
    # subsequent runs use the local cache, same as the existing htdemucs smoke test above.
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    vocals_path, instrumental_path = separate_stems(input_path, output_dir, model="htdemucs_ft")

    for path in (vocals_path, instrumental_path):
        assert path.exists()
        data, samplerate = sf.read(path)
        assert samplerate > 0
        assert data.shape[0] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_separation.py::test_separate_stems_rejects_an_unknown_model tests/test_separation.py::test_separate_stems_accepts_htdemucs_ft -v`
Expected: `test_separate_stems_rejects_an_unknown_model` FAILs (no `ValueError` raised — `model` isn't a real parameter yet, so passing `model="not-a-real-model"` raises `TypeError: separate_stems() got an unexpected keyword argument 'model'` instead). `test_separate_stems_accepts_htdemucs_ft` FAILs the same way.

- [ ] **Step 3: Implement the `model` parameter**

In `audio_pipeline/separation.py`, replace:

```python
_MODEL_NAME = "htdemucs"
_VOCALS_STEM = "vocals"
```

with:

```python
_SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")
_VOCALS_STEM = "vocals"
```

Replace the `separate_stems` function signature and its first lines:

```python
def separate_stems(
    input_path: str | Path,
    output_dir: str | Path,
    model: str = "htdemucs",
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, Path]:
    """Run Demucs on ``input_path`` and save both the isolated vocal stem and
    a reconstructed instrumental track (the original mix minus vocals) as wav
    files inside ``output_dir``.

    ``model`` selects which Demucs checkpoint to run: "htdemucs" (default,
    fast, ~80s for a 274s song on this CPU -- see NOTES.md) or "htdemucs_ft"
    (a fine-tuned bag of 4 models, noticeably less bleed, ~4x slower).
    Raises ``ValueError`` for anything else.

    ``on_progress``, if given, is called repeatedly with a 0-1 fraction as
    Demucs finishes each internal segment -- lets a caller (e.g. the job
    server) show real separation progress instead of a stage that just sits
    there for the ~80s+ this normally takes.

    Returns ``(vocals_path, instrumental_path)``.
    """
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported separation model {model!r}; expected one of {_SUPPORTED_MODELS}"
        )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = demucs_api.Separator(
        model=model,
        device=get_device(),
        jobs=_DEFAULT_JOBS,
        callback=_progress_callback(on_progress) if on_progress else None,
    )
```

The rest of the function body (from `original_wav, stems = separator.separate_audio_file(input_path)` through the final `return`) is unchanged, except the error message inside the `if _VOCALS_STEM not in stems:` block, which currently reads `f"Model '{_MODEL_NAME}' did not produce..."` — change `_MODEL_NAME` to `model` there since the constant no longer exists:

```python
    if _VOCALS_STEM not in stems:
        raise RuntimeError(
            f"Model '{model}' did not produce a '{_VOCALS_STEM}' stem; "
            f"got stems: {sorted(stems)}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_separation.py -v`
Expected: all tests PASS, including the two new ones (the `htdemucs_ft` test may take several minutes on first run while its checkpoints download).

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/separation.py tests/test_separation.py
git commit -m "Add a model parameter to separate_stems (htdemucs/htdemucs_ft)"
```

---

### Task 3: `process_song()` threads `separation_model` through and records it

**Files:**
- Modify: `audio_pipeline/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `separate_stems(input_path, output_dir, model, on_progress) -> tuple[Path, Path]` from Task 2.
- Produces: `process_song(..., separation_model: str = "htdemucs")` and `reprocess_melody_and_lyrics` unaffected (it never re-runs separation); `meta.json` gains a `"separation_model"` key. Task 4 consumes the `separation_model` parameter name and the `meta.json` key name.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_process_song_passes_the_requested_separation_model_through_and_records_it(tmp_path):
    cache_dir = tmp_path / "cache"
    fake_video_path = tmp_path / "video.mp4"
    fake_video_path.write_bytes(b"fake video data")

    extracted_wav = tmp_path / "extracted.wav"
    extracted_wav.write_bytes(b"fake wav data")
    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"fake vocals")
    instrumental_path = tmp_path / "instrumental.wav"
    instrumental_path.write_bytes(b"fake instrumental")
    midi_path = tmp_path / "melody.mid"
    midi_path.write_bytes(b"fake midi")
    notes_path = tmp_path / "notes.json"
    notes_path.write_text("[]")
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text("[]")

    melody_result = type("Melody", (), {"midi_path": midi_path, "notes_path": notes_path})()
    lyrics_result = type(
        "Lyrics", (), {"lyrics_path": lyrics_path, "background_vocal_ranges": []}
    )()

    with (
        patch("audio_pipeline.pipeline.extract_audio", return_value=extracted_wav),
        patch("audio_pipeline.pipeline.separate_stems") as mock_separate_stems,
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result),
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result),
    ):
        mock_separate_stems.return_value = (vocals_path, instrumental_path)
        process_song(
            fake_video_path, cache_dir=cache_dir, song_id="my-song",
            separation_model="htdemucs_ft",
        )

    assert mock_separate_stems.call_args.kwargs["model"] == "htdemucs_ft"

    meta = json.loads((cache_dir / "my-song" / "meta.json").read_text())
    assert meta["separation_model"] == "htdemucs_ft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline.py::test_process_song_passes_the_requested_separation_model_through_and_records_it -v`
Expected: FAIL with `TypeError: process_song() got an unexpected keyword argument 'separation_model'`.

- [ ] **Step 3: Implement**

In `audio_pipeline/pipeline.py`, change the `process_song` signature:

```python
def process_song(
    video_path: str | Path,
    cache_dir: str | Path = Path("cache"),
    song_id: str | None = None,
    force: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    language: str | None = None,
    lyrics_query: str | None = None,
    separation_model: str = "htdemucs",
) -> SongAssets:
```

Update its docstring to add, after the existing `language`/`lyrics_query` paragraph:

```python
    ``separation_model`` ("htdemucs" or "htdemucs_ft") is passed straight
    through to ``separate_stems`` and recorded in ``meta.json`` so a later
    caller can tell what a cached song was actually separated with.
```

Change the `separate_stems` call:

```python
    vocals_path, instrumental_path = separate_stems(
        extracted_wav, song_cache_dir, model=separation_model,
        on_progress=(lambda fraction: report("separating", fraction)) if on_progress else None,
    )
```

Change the `meta` dict construction near the end of `process_song`:

```python
    meta = {
        "source_file": str(video_path),
        "song_id": slug,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "separation_model": separation_model,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: all tests PASS (including the pre-existing ones — `separation_model` defaults to `"htdemucs"`, so nothing else changes behavior).

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/pipeline.py tests/test_pipeline.py
git commit -m "Thread separation_model through process_song and record it in meta.json"
```

---

### Task 4: Job server — request field, cache-mismatch-forces-reprocess

**Files:**
- Modify: `scripts/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `process_song(..., separation_model: str = "htdemucs")` from Task 3.
- Produces: `POST /api/jobs` accepts an optional `separation_model` field (`"htdemucs"` or `"htdemucs_ft"`, default `"htdemucs"`); `Job.separation_model`. Task 5 (frontend) consumes the `separation_model` request-body field name and its two valid values.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py` (near the existing `test_run_job_serializes_concurrent_requests_for_the_same_song` test):

```python
def test_run_job_forces_reprocessing_when_the_cached_model_differs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    # Pre-seed a fully cached song, separated with htdemucs (the default).
    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"cached wav data")
    (song_cache_dir / "notes.json").write_text("[]")
    (song_cache_dir / "lyrics.json").write_text("[]")
    (song_cache_dir / "meta.json").write_text(
        json.dumps({
            "song_id": "my-song", "processed_at": "2026-07-28T00:00:00+00:00",
            "separation_model": "htdemucs",
        })
    )

    process_song_calls = []

    def fake_probe_title(url):
        return "My Song"

    def fake_download_audio(url, output_dir, on_progress=None):
        return Path(output_dir) / "video.mp4", "My Song"

    def fake_process_song(
        video_path, cache_dir, song_id, on_progress, language, lyrics_query, separation_model,
        force,
    ):
        process_song_calls.append({"separation_model": separation_model, "force": force})
        song_cache_dir = Path(cache_dir) / song_id
        song_cache_dir.mkdir(parents=True, exist_ok=True)
        (song_cache_dir / "instrumental.wav").write_bytes(b"new wav data")
        (song_cache_dir / "notes.json").write_text("[]")
        (song_cache_dir / "lyrics.json").write_text("[]")
        (song_cache_dir / "meta.json").write_text(
            json.dumps({
                "song_id": song_id, "processed_at": "2026-07-29T00:00:00+00:00",
                "separation_model": separation_model,
            })
        )

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", fake_download_audio)
    monkeypatch.setattr(server, "process_song", fake_process_song)

    job = server.Job(
        id="job-1", url="https://example.com/watch?v=abc", separation_model="htdemucs_ft",
    )
    server._run_job(job)

    assert process_song_calls == [{"separation_model": "htdemucs_ft", "force": True}]
    assert job.status == "done"


def test_run_job_skips_reprocessing_when_the_cached_model_matches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"cached wav data")
    (song_cache_dir / "notes.json").write_text("[]")
    (song_cache_dir / "lyrics.json").write_text("[]")
    (song_cache_dir / "meta.json").write_text(
        json.dumps({
            "song_id": "my-song", "processed_at": "2026-07-28T00:00:00+00:00",
            "separation_model": "htdemucs",
        })
    )

    def fake_probe_title(url):
        return "My Song"

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called when the cache already matches")

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", _fail_if_called)
    monkeypatch.setattr(server, "process_song", _fail_if_called)

    job = server.Job(
        id="job-1", url="https://example.com/watch?v=abc", separation_model="htdemucs",
    )
    server._run_job(job)

    assert job.status == "done"
    assert job.slug == "my-song"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_server.py::test_run_job_forces_reprocessing_when_the_cached_model_differs tests/test_server.py::test_run_job_skips_reprocessing_when_the_cached_model_matches -v`
Expected: both FAIL with `TypeError: Job.__init__() got an unexpected keyword argument 'separation_model'`.

- [ ] **Step 3: Implement**

In `scripts/server.py`, add `separation_model` to the `Job` dataclass (after `language`):

```python
@dataclass
class Job:
    id: str
    url: str
    language: str | None = None
    separation_model: str = "htdemucs"
    status: str = "queued"
    progress: float = 0.0
    slug: str | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
```

Add a helper function near `_sync_meta_title` (which already reads `meta.json`):

```python
def _cached_separation_model(song_cache_dir: Path) -> str:
    """The separation model recorded in `song_cache_dir`'s meta.json, or the
    default ("htdemucs") for a cached song processed before this field
    existed.
    """
    meta_path = song_cache_dir / "meta.json"
    if not meta_path.exists():
        return "htdemucs"
    meta = json.loads(meta_path.read_text())
    return meta.get("separation_model", "htdemucs")


def _needs_reprocessing(slug: str, requested_model: str) -> bool:
    if not is_cached(CACHE_DIR, slug):
        return True
    return _cached_separation_model(CACHE_DIR / slug) != requested_model
```

Change `_run_job`'s body (the part inside `with _lock_for_slug(slug):`) from:

```python
        with _lock_for_slug(slug):
            on_progress("downloading")

            if not is_cached(CACHE_DIR, slug):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    downloaded_path, title = download_audio(
                        job.url, Path(tmp_dir),
                        on_progress=lambda fraction: on_progress("downloading", fraction),
                    )
                    slug = slugify(title)

                    process_song(
                        downloaded_path, cache_dir=CACHE_DIR, song_id=slug, on_progress=on_progress,
                        language=job.language, lyrics_query=title,
                    )
```

to:

```python
        with _lock_for_slug(slug):
            on_progress("downloading")

            if _needs_reprocessing(slug, job.separation_model):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    downloaded_path, title = download_audio(
                        job.url, Path(tmp_dir),
                        on_progress=lambda fraction: on_progress("downloading", fraction),
                    )
                    slug = slugify(title)
                    # Re-check against the *real* (downloaded) title's slug -- probe_title's
                    # guess and the real title can differ, and each has its own cache entry.
                    force = is_cached(CACHE_DIR, slug) and (
                        _cached_separation_model(CACHE_DIR / slug) != job.separation_model
                    )

                    process_song(
                        downloaded_path, cache_dir=CACHE_DIR, song_id=slug, on_progress=on_progress,
                        language=job.language, lyrics_query=title,
                        separation_model=job.separation_model, force=force,
                    )
```

Update `CreateJobRequest`:

```python
class CreateJobRequest(BaseModel):
    url: str
    language: Literal["en", "yue"] | None = None
    separation_model: Literal["htdemucs", "htdemucs_ft"] = "htdemucs"
```

Update `create_job`:

```python
@app.post("/api/jobs")
def create_job(request: CreateJobRequest) -> dict:
    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id, url=request.url, language=request.language,
        separation_model=request.separation_model,
    )
    _JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job_id}
```

- [ ] **Step 4: Update the two existing `fake_process_song` fixtures to accept the new keyword arguments**

The two pre-existing tests `test_run_job_serializes_concurrent_requests_for_the_same_song` and `test_run_job_reports_real_progress_and_reaches_1_on_success` each define a local `fake_process_song(video_path, cache_dir, song_id, on_progress, language, lyrics_query)` that's monkeypatched in for `server.process_song`. `_run_job` now also passes `separation_model=` and `force=` as keyword arguments, which will raise `TypeError` against the old signature. Update both fixtures' signatures to:

```python
    def fake_process_song(
        video_path, cache_dir, song_id, on_progress, language, lyrics_query, separation_model,
        force,
    ):
```

(No other change needed inside either fixture body — neither uses `separation_model`/`force`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_server.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/server.py tests/test_server.py
git commit -m "Add separation_model to the job API and force reprocessing on a model mismatch"
```

---

### Task 5: `LoadSongForm.tsx` — model select

**Files:**
- Modify: `frontend/src/components/LoadSongForm.tsx`

**Interfaces:**
- Consumes: `POST /api/jobs` body field `separation_model: "htdemucs" | "htdemucs_ft"` from Task 4.
- Produces: nothing consumed by a later task (this is the last code task before manual verification).

No automated test for this step — see the Global Constraints note on this project having no React component-testing setup at all (not even for the existing language select this mirrors). Verify manually in Task 6.

- [ ] **Step 1: Add the model select**

In `frontend/src/components/LoadSongForm.tsx`, add a type alias next to the existing `LanguageOption`:

```typescript
type SeparationModelOption = 'htdemucs' | 'htdemucs_ft'
```

Add state next to the existing `language` state:

```typescript
const [separationModel, setSeparationModel] = useState<SeparationModelOption>('htdemucs')
```

Add the request body field in `handleSubmit`'s `fetch('/api/jobs', ...)` call — change:

```typescript
      body: JSON.stringify({ url: url.trim(), language: language || null }),
```

to:

```typescript
      body: JSON.stringify({
        url: url.trim(),
        language: language || null,
        separation_model: separationModel,
      }),
```

Add the `<select>` itself in the JSX, right after the existing language `<select>` (before the submit `<button>`):

```tsx
        <select
          id="song-separation-model"
          className="select-language"
          value={separationModel}
          onChange={(event) => setSeparationModel(event.target.value as SeparationModelOption)}
          disabled={isBusy}
          aria-label="Vocal separation quality"
        >
          <option value="htdemucs">Fast (default)</option>
          <option value="htdemucs_ft">Better quality (slower)</option>
        </select>
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npm run build` (or `npx tsc -b --noEmit` if that's the project's usual fast type-check command — check `frontend/package.json`'s `scripts` for the exact one already in use).
Expected: no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/LoadSongForm.tsx
git commit -m "Add a vocal-separation-quality select to the load-song form"
```

---

### Task 6: End-to-end manual verification + NOTES.md entry (two-option feature)

**Files:**
- Modify: `NOTES.md`

**Interfaces:**
- Consumes: the fully working two-option (`htdemucs`/`htdemucs_ft`) feature from Tasks 1-5.
- Produces: a NOTES.md entry documenting manual verification, following this project's established convention (see any existing entry, e.g. "Frontend redesign + load-song-from-a-link").

- [ ] **Step 1: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest tests/ -q` and `cd frontend && npm test` (or whatever `frontend/package.json`'s `test` script is — already confirmed as `vitest run` in the Global Constraints research for this plan).
Expected: all Python and frontend tests pass.

- [ ] **Step 2: Manual browser check**

Start the job server (`venv/Scripts/python.exe scripts/server.py`) and the frontend dev server (`cd frontend && npm run dev`), open the app in a real (non-automated) browser, and confirm:
- The new select appears next to the language select on the load-song form, defaulted to "Fast (default)".
- Submitting a link with the default selection behaves exactly as before (no visible change).
- Re-submitting the *same already-cached* link with "Better quality (slower)" selected actually re-runs processing (progress bar restarts from "separating") instead of instantly completing — the concrete proof the cache-mismatch-forces-reprocess logic from Task 4 works against the real server, not just its mocked test.

- [ ] **Step 3: Record the outcome in NOTES.md**

Append a dated entry summarizing: what was built (model choice at upload, htdemucs/htdemucs_ft), the manual verification steps from Step 2 and their results, and — if Task 1 concluded BS-RoFormer isn't viable on this machine — a note that the feature ships with two options and why the third was dropped (cross-reference Task 1's own NOTES.md entry rather than repeating it).

- [ ] **Step 4: Commit**

```bash
git add NOTES.md
git commit -m "Verify separation model choice end-to-end"
```

---

### Task 7: Wire in BS-RoFormer as the third option (conditional on Task 1)

**Only do this task if Task 1's NOTES.md entry confirmed a working `audio-separator` API.** If Task 1 documented that BS-RoFormer isn't viable here, skip this task entirely — the plan is complete after Task 6.

**Files:**
- Modify: `audio_pipeline/separation.py`
- Modify: `scripts/server.py`
- Modify: `frontend/src/components/LoadSongForm.tsx`
- Test: `tests/test_separation.py`, `tests/test_server.py`
- Modify: `NOTES.md`

**Interfaces:**
- Consumes: the exact `audio-separator` API confirmed in Task 1's NOTES.md entry. Do not guess or reconstruct it from general knowledge of the package — use precisely what Task 1 recorded, including the confirmed model filename and output-file-naming pattern.
- Produces: `separate_stems(..., model="bs_roformer", ...)` works the same way `"htdemucs"`/`"htdemucs_ft"` do; `separation_model: Literal["htdemucs", "htdemucs_ft", "bs_roformer"]` in `scripts/server.py`; a third `<option>` in `LoadSongForm.tsx`.

- [ ] **Step 1: Write the failing test for the new model**

Add to `tests/test_separation.py`, using the real, confirmed API from Task 1 (this step's exact code depends on Task 1's findings and cannot be written until they exist — write it following the same shape as `test_separate_stems_accepts_htdemucs_ft` from Task 2, asserting the same things: both returned paths exist, are readable wav files, and have non-empty audio):

```python
def test_separate_stems_accepts_bs_roformer(tmp_path):
    # First run downloads the BS-RoFormer checkpoint (see NOTES.md's BS-RoFormer validation
    # entry for its size/timing) and requires internet access.
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    vocals_path, instrumental_path = separate_stems(input_path, output_dir, model="bs_roformer")

    for path in (vocals_path, instrumental_path):
        assert path.exists()
        data, samplerate = sf.read(path)
        assert samplerate > 0
        assert data.shape[0] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_separation.py::test_separate_stems_accepts_bs_roformer -v`
Expected: FAIL (`ValueError: Unsupported separation model 'bs_roformer'`).

- [ ] **Step 3: Implement the BS-RoFormer backend**

In `audio_pipeline/separation.py`:
- Add `"bs_roformer"` to `_SUPPORTED_MODELS`.
- Add the `audio-separator` import at the top of the file, using the exact import path from Task 1's NOTES.md entry.
- Split `separate_stems`'s body: keep the existing Demucs code path for `model in ("htdemucs", "htdemucs_ft")`, and add a new branch (or a small private helper, e.g. `_separate_bs_roformer(input_path, output_dir, on_progress)`) for `model == "bs_roformer"` that uses the exact confirmed API from Task 1 and returns `(vocals_path, instrumental_path)` in the same shape as the Demucs path — same contract, since `pipeline.py` and everything downstream only cares about those two file paths, not which backend produced them.
- Add a docstring note for the `model` parameter covering the new option, its rough relative speed/quality versus the other two (from Task 1's measured timing), matching the style of the existing `htdemucs`/`htdemucs_ft` docstring note from Task 2.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_separation.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Extend the job server's Literal and re-verify the existing model-mismatch tests**

In `scripts/server.py`, change:

```python
    separation_model: Literal["htdemucs", "htdemucs_ft"] = "htdemucs"
```

to:

```python
    separation_model: Literal["htdemucs", "htdemucs_ft", "bs_roformer"] = "htdemucs"
```

Run: `venv/Scripts/python.exe -m pytest tests/test_server.py -v`
Expected: all tests still PASS unchanged (this Literal widening doesn't affect the existing mismatch-detection logic, which compares plain strings).

- [ ] **Step 6: Add the third UI option**

In `frontend/src/components/LoadSongForm.tsx`, widen the type and add the third `<option>`:

```typescript
type SeparationModelOption = 'htdemucs' | 'htdemucs_ft' | 'bs_roformer'
```

```tsx
          <option value="bs_roformer">Best quality (slowest)</option>
```

(Added after the existing `htdemucs_ft` option, inside the same `<select>` from Task 5.)

- [ ] **Step 7: Type-check**

Run the same frontend type-check command used in Task 5, Step 2.
Expected: no TypeScript errors.

- [ ] **Step 8: Manual browser verification**

Same procedure as Task 6, Step 2, but selecting "Best quality (slowest)" and confirming it actually runs (and takes noticeably longer than the other two options, consistent with Task 1's measured timing).

- [ ] **Step 9: Record the outcome in NOTES.md**

Append a dated entry ("## Separation model choice: BS-RoFormer wired in as the third option (recorded <date>)") documenting the manual verification from Step 8.

- [ ] **Step 10: Commit**

```bash
git add audio_pipeline/separation.py scripts/server.py frontend/src/components/LoadSongForm.tsx tests/test_separation.py tests/test_server.py NOTES.md
git commit -m "Wire BS-RoFormer in as a third separation model option"
```
