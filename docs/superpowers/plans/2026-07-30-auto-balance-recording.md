# Auto-balance and Clean Up Recordings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recordings of a player's take (instrumental + live mic) automatically come out with the vocal cleaned up (denoised, compressed) and balanced against the music (louder, but not overwhelming), and no longer sound shifted from the start — with no new UI, exactly the same "Enable Mic" / "Start Recording" / "Stop Recording" flow as today.

**Architecture:** `useRecording.ts` records the instrumental and mic as two separate `MediaRecorder` tracks instead of one pre-mixed stream, and uploads both to the job server as a multipart form. A new `audio_pipeline/mastering.py::master_recording()` — pure ffmpeg/soundfile, no ML — aligns the two tracks to the same start time, cleans and loudness-normalizes the vocal, loudness-normalizes the instrumental to a quieter target, mixes them, and limits the result against clipping. `scripts/server.py`'s existing `/api/recordings/mp3` endpoint calls it before its existing mp3 transcode step.

**Tech Stack:** Python (ffmpeg-python, soundfile, numpy, librosa, FastAPI, pytest), TypeScript/React (Web Audio API, vitest/tsc for type-checking only — no new frontend test infra).

## Global Constraints

- No new UI — fully automatic, same start/stop recording flow as today (spec: "Goal").
- Server-side mastering on two separately-recorded tracks, not real-time Web Audio nodes (spec: "Decision: server-side post-processing").
- Vocal loudness target: **-16 LUFS**. Instrumental loudness target: **-20 LUFS**. (spec: "Backend changes", narrowed after feedback that the voice was too loud.)
- Denoiser is ffmpeg's built-in `afftdn` — not `arnndn` (would need bundling an external RNNoise model). Browser-side `noiseSuppression`/`echoCancellation`/`autoGainControl` are already on for the mic stream in `useMicPitch.ts` — this is a second pass, not the only line of defense. (spec: "Open questions resolved")
- The recording-start-offset fix is a **measured** constant (`_RECORDING_OFFSET_SECONDS`), not a guessed one — ships as a wired-but-untuned no-op (`0.0`) until a human measures the real value against a real recording (spec: "Recording start offset"). Do not invent a plausible-sounding number for this constant.
- `tests/test_separation.py`/`tests/test_transcode.py`'s existing convention is real (unmocked) ffmpeg/audio smoke tests against synthetic clips, not mocks — follow that convention for all of `audio_pipeline/mastering.py`. Mocking is only used in `tests/test_server.py`, to isolate the endpoint's own plumbing from the (separately, directly tested) mastering internals — matching how that file already mocks `process_song`/`download_audio` in the job-server tests.
- This project has no browser-API mocking / hooks-testing infrastructure (confirmed: `frontend/src/hooks/` has zero `.test.ts` files, and NOTES.md repeatedly documents that `<audio>`/`MediaRecorder`/`getUserMedia` behavior can't be verified through this project's browser automation either — only by a human in a real browser). Do not add new frontend test infrastructure for `useRecording.ts` — verify manually, per Task 9.
- New Python dependency: `python-multipart` (FastAPI's multipart/form-data parser, needed for the endpoint's `File`/`Form` params — confirmed not currently installed in `venv`).

---

### Task 1: `transcode_to_wav` helper

**Files:**
- Modify: `audio_pipeline/transcode.py`
- Test: `tests/test_transcode.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `transcode_to_wav(input_path: str | Path, output_dir: str | Path) -> Path`, raising `RuntimeError` on any ffmpeg failure (same pattern as the existing `transcode_to_mp3`). Task 5 consumes this signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transcode.py`, after the existing `test_transcode_to_mp3_raises_a_clear_error_on_a_bogus_input`:

```python
def test_transcode_to_wav_produces_a_valid_wav(tmp_path):
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    wav_path = transcode_to_wav(input_path, output_dir)

    assert wav_path.exists()
    assert wav_path.suffix == ".wav"
    data, samplerate = sf.read(wav_path)
    assert samplerate == 44100
    assert data.shape[0] > 0
    output_duration_s = data.shape[0] / samplerate
    assert output_duration_s == pytest.approx(input_duration_s, abs=0.5)


def test_transcode_to_wav_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.webm"
    bogus_path.write_bytes(b"this is not a real media file")

    with pytest.raises(RuntimeError, match="ffmpeg failed to transcode"):
        transcode_to_wav(bogus_path, tmp_path / "out")
```

Change the import line at the top of `tests/test_transcode.py` from:

```python
from audio_pipeline.transcode import transcode_to_mp3
```

to:

```python
from audio_pipeline.transcode import transcode_to_mp3, transcode_to_wav
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_transcode.py -v`
Expected: the two new tests FAIL with `ImportError: cannot import name 'transcode_to_wav'`.

- [ ] **Step 3: Implement**

Append to `audio_pipeline/transcode.py`:

```python
def transcode_to_wav(input_path: str | Path, output_dir: str | Path) -> Path:
    """Transcode ``input_path`` (any container/codec ffmpeg can read, e.g. the
    webm/opus a browser's MediaRecorder produces) to a 44.1kHz wav file inside
    ``output_dir`` -- for downstream processing (audio_pipeline/mastering.py)
    that needs sample-accurate access via soundfile, which can't read
    webm/opus directly. Forces a fixed sample rate so two independently-
    transcoded tracks (e.g. a recording's separate vocal/instrumental takes)
    always line up sample-for-sample.

    Returns the path to the saved wav file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / f"{input_path.stem}.wav"

    try:
        (
            ffmpeg.input(str(input_path))
            .output(str(wav_path), ar=44100)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to transcode {input_path} to wav: {stderr}") from exc

    return wav_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_transcode.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/transcode.py tests/test_transcode.py
git commit -m "Add a transcode_to_wav helper alongside the existing mp3 transcoder"
```

---

### Task 2: `mastering.py` — start-offset correction

**Files:**
- Create: `audio_pipeline/mastering.py`
- Test: `tests/test_mastering.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: module `audio_pipeline/mastering.py` with `_RECORDING_OFFSET_SECONDS: float` (module-level constant, `0.0`) and `_correct_start_offset(vocal_path: Path, instrumental_path: Path, output_dir: Path, offset_seconds: float = _RECORDING_OFFSET_SECONDS) -> tuple[Path, Path]`. Also the test file's `_write_tone` helper, reused by Tasks 3-5's tests. Task 5 consumes `_correct_start_offset`'s signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mastering.py`:

```python
"""Smoke tests for the recording auto-balance/cleanup mastering pass.

Uses synthetic tones rather than a real sung take -- same convention as
tests/test_separation.py and tests/test_transcode.py: proves the ffmpeg/
soundfile pipeline runs end-to-end and moves loudness/timing in the right
direction, not that it sounds good on a real voice. That needs a human
listen (see NOTES.md's established convention for every other audio-quality
claim in this project) -- see Task 9 of this feature's implementation plan.
"""
import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.mastering import _correct_start_offset


def _write_tone(path, duration_s=3.0, samplerate=44100, freq_hz=220.0, amplitude=0.2):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, tone.astype(np.float32), samplerate)


def test_correct_start_offset_trims_the_lagging_vocal_track(tmp_path):
    samplerate = 44100
    lag_seconds = 0.5
    duration_s = 2.0

    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Simulate the vocal arriving `lag_seconds` late: pad silence onto its front.
    lag_samples = int(lag_seconds * samplerate)
    lagged_vocal = np.concatenate([np.zeros(lag_samples, dtype=np.float32), tone])

    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, lagged_vocal, samplerate)
    sf.write(instrumental_path, tone, samplerate)

    output_dir = tmp_path / "aligned"
    aligned_vocal_path, aligned_instrumental_path = _correct_start_offset(
        vocal_path, instrumental_path, output_dir, offset_seconds=lag_seconds
    )

    aligned_vocal, _ = sf.read(aligned_vocal_path)
    aligned_instrumental, _ = sf.read(aligned_instrumental_path)

    n = min(len(aligned_vocal), len(aligned_instrumental))
    np.testing.assert_allclose(aligned_vocal[:n], aligned_instrumental[:n], atol=1e-6)


def test_correct_start_offset_trims_the_leading_instrumental_when_offset_is_negative(tmp_path):
    samplerate = 44100
    lead_seconds = 0.3
    duration_s = 2.0

    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    lead_samples = int(lead_seconds * samplerate)
    leading_instrumental = np.concatenate([np.zeros(lead_samples, dtype=np.float32), tone])

    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, tone, samplerate)
    sf.write(instrumental_path, leading_instrumental, samplerate)

    output_dir = tmp_path / "aligned"
    aligned_vocal_path, aligned_instrumental_path = _correct_start_offset(
        vocal_path, instrumental_path, output_dir, offset_seconds=-lead_seconds
    )

    aligned_vocal, _ = sf.read(aligned_vocal_path)
    aligned_instrumental, _ = sf.read(aligned_instrumental_path)

    n = min(len(aligned_vocal), len(aligned_instrumental))
    np.testing.assert_allclose(aligned_vocal[:n], aligned_instrumental[:n], atol=1e-6)


def test_correct_start_offset_is_a_no_op_at_zero(tmp_path):
    samplerate = 44100
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, tone, samplerate)
    sf.write(instrumental_path, tone, samplerate)

    output_dir = tmp_path / "aligned"
    aligned_vocal_path, _ = _correct_start_offset(
        vocal_path, instrumental_path, output_dir, offset_seconds=0.0
    )

    aligned_vocal, _ = sf.read(aligned_vocal_path)
    original_vocal, _ = sf.read(vocal_path)
    np.testing.assert_allclose(aligned_vocal, original_vocal, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audio_pipeline.mastering'`.

- [ ] **Step 3: Implement**

Create `audio_pipeline/mastering.py`:

```python
"""Auto-balance and clean-up mastering pass for a player's recorded take.

useRecording.ts (frontend) records the instrumental and the live mic as two
separate tracks instead of one pre-mixed stream (see
docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md). This
module takes those two tracks and produces a single mastered wav: aligned to
the same start time, the vocal denoised/compressed/highpassed, both tracks
loudness-balanced against each other (vocal louder), then mixed and limited
so the result can't clip.

master_recording() (added in a later task) is the single entry point
scripts/server.py calls; the rest of this module is its implementation,
ordered in the same sequence the audio actually flows through: align ->
clean the vocal -> loudness-normalize each track -> mix -> limit.
"""
from pathlib import Path

import soundfile as sf

# Positive = the vocal (mic) track lags the instrumental in a raw two-track recording.
# The mic capture path (getUserMedia -> hardware -> driver buffering) has real warm-up
# latency the instrumental's already-decoded <audio> element source doesn't, so the
# singer's actual voice lands later in a raw recording than when they actually sang it --
# see the "Recording start offset" section of
# docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md.
#
# 0.0 (this untuned default) is a no-op. Do NOT replace this with a guessed "typical mic
# latency" number -- measure the real value with scripts/measure_recording_offset.py
# against a real test recording (a sharp clap right on the song's first strong
# instrumental beat) before relying on this correction. See that script's own docstring
# for the exact procedure.
_RECORDING_OFFSET_SECONDS = 0.0


def _correct_start_offset(
    vocal_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    offset_seconds: float = _RECORDING_OFFSET_SECONDS,
) -> tuple[Path, Path]:
    """Trim ``offset_seconds`` off whichever track leads, so both start at the
    same real moment. Positive ``offset_seconds`` means the vocal lags (the
    common case, see the module docstring) -- trims that many seconds off the
    *front* of the vocal track. Negative means the vocal leads -- trims off
    the instrumental's front instead. Zero is a no-op: both tracks are copied
    through unchanged.
    """
    vocal_data, vocal_sr = sf.read(vocal_path)
    instrumental_data, instrumental_sr = sf.read(instrumental_path)

    if offset_seconds > 0:
        trim_samples = int(round(offset_seconds * vocal_sr))
        vocal_data = vocal_data[trim_samples:]
    elif offset_seconds < 0:
        trim_samples = int(round(-offset_seconds * instrumental_sr))
        instrumental_data = instrumental_data[trim_samples:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_vocal_path = output_dir / "vocal_aligned.wav"
    aligned_instrumental_path = output_dir / "instrumental_aligned.wav"
    sf.write(aligned_vocal_path, vocal_data, vocal_sr)
    sf.write(aligned_instrumental_path, instrumental_data, instrumental_sr)
    return aligned_vocal_path, aligned_instrumental_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/mastering.py tests/test_mastering.py
git commit -m "Add start-offset correction to a new mastering module"
```

---

### Task 3: `mastering.py` — vocal cleanup (highpass, denoise, compress)

**Files:**
- Modify: `audio_pipeline/mastering.py`
- Test: `tests/test_mastering.py`

**Interfaces:**
- Consumes: nothing from other tasks (independent of Task 2's function, both are used together later by Task 5).
- Produces: `_HIGHPASS_HZ: int` constant and `_clean_vocal(input_path: Path, output_dir: Path) -> Path`, raising `RuntimeError` on ffmpeg failure. Task 5 consumes this signature.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mastering.py`:

```python
def test_clean_vocal_removes_low_frequency_rumble(tmp_path):
    duration_s = 2.0
    samplerate = 44100
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    voice = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    rumble = 0.2 * np.sin(2 * np.pi * 40.0 * t)  # well below the 90Hz highpass cutoff
    mixed = (voice + rumble).astype(np.float32)

    input_path = tmp_path / "noisy_vocal.wav"
    sf.write(input_path, mixed, samplerate)

    cleaned_path = _clean_vocal(input_path, tmp_path)
    cleaned, cleaned_sr = sf.read(cleaned_path)
    assert cleaned_sr == samplerate

    rumble_bin = int(round(40.0 * duration_s))
    original_rumble_energy = np.abs(np.fft.rfft(mixed))[rumble_bin]
    cleaned_rumble_energy = np.abs(np.fft.rfft(cleaned.astype(np.float32)))[rumble_bin]
    assert cleaned_rumble_energy < original_rumble_energy * 0.1


def test_clean_vocal_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.wav"
    bogus_path.write_bytes(b"this is not real audio")

    with pytest.raises(RuntimeError, match="ffmpeg failed to clean vocal"):
        _clean_vocal(bogus_path, tmp_path / "out")
```

Change the import line at the top of `tests/test_mastering.py` from:

```python
from audio_pipeline.mastering import _correct_start_offset
```

to:

```python
from audio_pipeline.mastering import _clean_vocal, _correct_start_offset
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: the two new tests FAIL with `ImportError: cannot import name '_clean_vocal'`.

- [ ] **Step 3: Implement**

Change the top of `audio_pipeline/mastering.py` from:

```python
from pathlib import Path

import soundfile as sf
```

to:

```python
from pathlib import Path

import ffmpeg
import soundfile as sf
```

Append to `audio_pipeline/mastering.py`:

```python
_HIGHPASS_HZ = 90


def _clean_vocal(input_path: Path, output_dir: Path) -> Path:
    """Highpass (removes rumble/handling noise below ``_HIGHPASS_HZ``), denoise
    (ffmpeg's built-in ``afftdn`` -- no extra model download, see this
    feature's design spec for why not ``arnndn``), and compress
    (``acompressor``, evens out quiet/loud parts) the vocal track. Loudness
    balancing against the instrumental happens separately (see
    ``_apply_loudnorm``, added in a later task).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vocal_cleaned.wav"

    try:
        (
            ffmpeg.input(str(input_path))
            .filter("highpass", f=_HIGHPASS_HZ)
            .filter("afftdn")
            .filter("acompressor")
            .output(str(output_path))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to clean vocal {input_path}: {stderr}") from exc

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/mastering.py tests/test_mastering.py
git commit -m "Add vocal cleanup (highpass/denoise/compress) to the mastering module"
```

---

### Task 4: `mastering.py` — two-pass loudness normalization

**Files:**
- Modify: `audio_pipeline/mastering.py`
- Test: `tests/test_mastering.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `_LOUDNORM_TP`/`_LOUDNORM_LRA` constants, `_measure_loudness(input_path: Path, target_i: float) -> dict`, and `_apply_loudnorm(input_path: Path, output_path: Path, target_i: float) -> Path`, raising `RuntimeError` on ffmpeg failure. Task 5 consumes `_apply_loudnorm`'s signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mastering.py`:

```python
def test_apply_loudnorm_moves_measured_loudness_close_to_the_target(tmp_path):
    quiet_path = tmp_path / "quiet.wav"
    _write_tone(quiet_path, amplitude=0.02)

    output_path = tmp_path / "normalized.wav"
    _apply_loudnorm(quiet_path, output_path, target_i=-16)

    after_stats = _measure_loudness(output_path, target_i=-16)
    assert float(after_stats["input_i"]) == pytest.approx(-16, abs=1.0)


def test_apply_loudnorm_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.wav"
    bogus_path.write_bytes(b"this is not real audio")

    with pytest.raises(RuntimeError, match="ffmpeg failed to measure loudness"):
        _apply_loudnorm(bogus_path, tmp_path / "out.wav", target_i=-16)
```

Change the import line at the top of `tests/test_mastering.py` from:

```python
from audio_pipeline.mastering import _clean_vocal, _correct_start_offset
```

to:

```python
from audio_pipeline.mastering import _apply_loudnorm, _clean_vocal, _correct_start_offset, _measure_loudness
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: the two new tests FAIL with `ImportError: cannot import name '_apply_loudnorm'`.

- [ ] **Step 3: Implement**

Change the top of `audio_pipeline/mastering.py` from:

```python
from pathlib import Path

import ffmpeg
import soundfile as sf
```

to:

```python
import json
import os
import re
from pathlib import Path

import ffmpeg
import soundfile as sf
```

Append to `audio_pipeline/mastering.py`:

```python
_LOUDNORM_TP = -1.5
_LOUDNORM_LRA = 11


def _measure_loudness(input_path: Path, target_i: float) -> dict:
    """First pass of ffmpeg's two-pass loudnorm: measures the real loudness
    stats ffmpeg needs to normalize accurately on the second pass (see
    ``_apply_loudnorm``), instead of relying on loudnorm's single-pass mode,
    which is a rougher, real-time-only estimate. Output audio is discarded
    (``-f null``) -- only the stats loudnorm prints to stderr as JSON matter.
    """
    try:
        _, stderr = (
            ffmpeg.input(str(input_path))
            .filter("loudnorm", i=target_i, tp=_LOUDNORM_TP, lra=_LOUDNORM_LRA, print_format="json")
            .output(os.devnull, format="null")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to measure loudness of {input_path}: {stderr_text}") from exc

    stderr_text = stderr.decode(errors="replace")
    match = re.search(r"\{[^{}]*\}", stderr_text, re.DOTALL)
    if not match:
        raise RuntimeError(f"ffmpeg loudnorm did not report measured stats for {input_path}")
    return json.loads(match.group(0))


def _apply_loudnorm(input_path: Path, output_path: Path, target_i: float) -> Path:
    """Second pass: normalizes ``input_path`` to ``target_i`` LUFS using the
    stats ``_measure_loudness`` already measured, writing the result to
    ``output_path``.
    """
    stats = _measure_loudness(input_path, target_i)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg.input(str(input_path))
            .filter(
                "loudnorm",
                i=target_i,
                tp=_LOUDNORM_TP,
                lra=_LOUDNORM_LRA,
                measured_I=stats["input_i"],
                measured_TP=stats["input_tp"],
                measured_LRA=stats["input_lra"],
                measured_thresh=stats["input_thresh"],
                offset=stats["target_offset"],
                linear="true",
                print_format="summary",
            )
            .output(str(output_path))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to normalize loudness of {input_path}: {stderr_text}") from exc

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/mastering.py tests/test_mastering.py
git commit -m "Add two-pass loudness normalization to the mastering module"
```

---

### Task 5: `mastering.py` — mix, limit, and the `master_recording` entry point

**Files:**
- Modify: `audio_pipeline/mastering.py`
- Test: `tests/test_mastering.py`

**Interfaces:**
- Consumes: `transcode_to_wav` (Task 1), `_correct_start_offset` (Task 2), `_clean_vocal` (Task 3), `_apply_loudnorm` (Task 4).
- Produces: `master_recording(vocal_path: str | Path, instrumental_path: str | Path, output_dir: str | Path) -> Path`. Task 6 (`scripts/server.py`) consumes this signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mastering.py`:

```python
def test_master_recording_produces_a_playable_wav(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    _write_tone(vocal_path, freq_hz=440.0, amplitude=0.05)
    _write_tone(instrumental_path, freq_hz=220.0, amplitude=0.2)

    mastered_path = master_recording(vocal_path, instrumental_path, tmp_path / "out")

    assert mastered_path.exists()
    data, samplerate = sf.read(mastered_path)
    assert samplerate > 0
    assert data.shape[0] > 0
    assert not np.isnan(data).any()
    duration_s = data.shape[0] / samplerate
    assert duration_s == pytest.approx(3.0, abs=0.5)


def test_master_recording_does_not_clip(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    _write_tone(vocal_path, freq_hz=440.0, amplitude=0.9)
    _write_tone(instrumental_path, freq_hz=220.0, amplitude=0.9)

    mastered_path = master_recording(vocal_path, instrumental_path, tmp_path / "out")

    data, _ = sf.read(mastered_path)
    assert np.max(np.abs(data)) <= 1.0


def test_master_recording_makes_the_vocal_more_prominent_relative_to_input(tmp_path):
    duration_s = 3.0
    samplerate = 44100
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    _write_tone(vocal_path, duration_s=duration_s, samplerate=samplerate, freq_hz=440.0, amplitude=0.02)
    _write_tone(instrumental_path, duration_s=duration_s, samplerate=samplerate, freq_hz=880.0, amplitude=0.3)

    mastered_path = master_recording(vocal_path, instrumental_path, tmp_path / "out")

    mastered, _ = sf.read(mastered_path)
    spectrum = np.abs(np.fft.rfft(mastered.astype(np.float32)))
    vocal_bin = int(round(440.0 * duration_s))
    instrumental_bin = int(round(880.0 * duration_s))

    before_ratio = 0.02 / 0.3
    after_ratio = spectrum[vocal_bin] / spectrum[instrumental_bin]
    assert after_ratio > before_ratio
```

Change the import line at the top of `tests/test_mastering.py` from:

```python
from audio_pipeline.mastering import _apply_loudnorm, _clean_vocal, _correct_start_offset, _measure_loudness
```

to:

```python
from audio_pipeline.mastering import (
    _apply_loudnorm,
    _clean_vocal,
    _correct_start_offset,
    _measure_loudness,
    master_recording,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: the three new tests FAIL with `ImportError: cannot import name 'master_recording'`.

- [ ] **Step 3: Implement**

Change the top of `audio_pipeline/mastering.py` from:

```python
import json
import os
import re
from pathlib import Path

import ffmpeg
import soundfile as sf
```

to:

```python
import json
import os
import re
from pathlib import Path

import ffmpeg
import soundfile as sf

from audio_pipeline.transcode import transcode_to_wav
```

Append to `audio_pipeline/mastering.py`:

```python
_VOCAL_TARGET_LUFS = -16
_INSTRUMENTAL_TARGET_LUFS = -20
_LIMITER_CEILING = 0.95


def _mix_and_limit(vocal_path: Path, instrumental_path: Path, output_path: Path) -> Path:
    """Mixes the two loudness-normalized tracks and applies a final limiter so
    the vocal's loudness boost can't clip the combined output. ``normalize=0``
    on ``amix`` keeps the loudnorm targets intact -- amix's own default
    (``normalize=1``) would auto-scale both inputs down by input count,
    undoing the deliberate vocal-vs-instrumental balance.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocal_stream = ffmpeg.input(str(vocal_path))
    instrumental_stream = ffmpeg.input(str(instrumental_path))

    try:
        (
            ffmpeg.filter([vocal_stream, instrumental_stream], "amix", inputs=2, duration="longest", normalize=0)
            .filter("alimiter", limit=_LIMITER_CEILING)
            .output(str(output_path))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to mix vocal and instrumental into {output_path}: {stderr_text}") from exc

    return output_path


def master_recording(vocal_path: str | Path, instrumental_path: str | Path, output_dir: str | Path) -> Path:
    """Auto-balance and clean up a player's recorded take -- the single entry
    point scripts/server.py calls. ``vocal_path``/``instrumental_path`` are
    the two separately-recorded tracks from useRecording.ts, in any
    container/codec ffmpeg can read (in practice, webm/opus). Returns the
    path to the mastered wav (start-aligned, vocal cleaned and balanced
    louder than the instrumental, mixed, limited against clipping).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocal_wav = transcode_to_wav(vocal_path, output_dir)
    instrumental_wav = transcode_to_wav(instrumental_path, output_dir)

    aligned_vocal_path, aligned_instrumental_path = _correct_start_offset(
        vocal_wav, instrumental_wav, output_dir
    )
    cleaned_vocal_path = _clean_vocal(aligned_vocal_path, output_dir)
    normalized_vocal_path = _apply_loudnorm(
        cleaned_vocal_path, output_dir / "vocal_normalized.wav", _VOCAL_TARGET_LUFS
    )
    normalized_instrumental_path = _apply_loudnorm(
        aligned_instrumental_path, output_dir / "instrumental_normalized.wav", _INSTRUMENTAL_TARGET_LUFS
    )

    mastered_path = output_dir / "mastered.wav"
    _mix_and_limit(normalized_vocal_path, normalized_instrumental_path, mastered_path)
    return mastered_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/mastering.py tests/test_mastering.py
git commit -m "Add mixing/limiting and the master_recording entry point"
```

---

### Task 6: Wire mastering into the recording endpoint

**Files:**
- Modify: `scripts/server.py`
- Modify: `requirements.txt`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `master_recording(vocal_path, instrumental_path, output_dir) -> Path` from Task 5.
- Produces: `POST /api/recordings/mp3` now accepts a multipart form with `vocal` and `instrumental` file fields plus a `song_id` text field (previously: a single raw request body plus a `song_id` query param). Task 7 (frontend) consumes this new contract exactly.

- [ ] **Step 1: Install the new dependency**

Run: `venv/Scripts/python.exe -m pip install python-multipart`
Expected: installs successfully (FastAPI raises a clear runtime error, "Form data requires 'python-multipart' to be installed", if this is skipped and Step 5's tests are run anyway).

Add to `requirements.txt`, after the existing `uvicorn[standard]>=0.29.0` line:

```
# Required by FastAPI's File()/Form() params (scripts/server.py's recording upload
# endpoint) to parse multipart/form-data request bodies.
python-multipart>=0.0.9
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_server.py`, after the existing `test_delete_recording_404s_for_unknown_filename` test (at the end of the file):

```python
def test_render_recording_mp3_masters_both_uploaded_tracks_and_saves_the_take(tmp_path, monkeypatch):
    recordings_dir = tmp_path / "recordings"
    monkeypatch.setattr(server, "RECORDINGS_DIR", recordings_dir)

    calls = []

    def fake_master_recording(vocal_path, instrumental_path, output_dir):
        calls.append((Path(vocal_path).read_bytes(), Path(instrumental_path).read_bytes()))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mastered_path = output_dir / "mastered.wav"
        t = np.linspace(0, 0.5, 22050, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(mastered_path, tone, 44100)
        return mastered_path

    monkeypatch.setattr(server, "master_recording", fake_master_recording)

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"fake vocal bytes", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert calls == [(b"fake vocal bytes", b"fake instrumental bytes")]

    saved = list(recordings_dir.glob("my-song__*.mp3"))
    assert len(saved) == 1


def test_render_recording_mp3_rejects_an_empty_track(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 400


def test_render_recording_mp3_returns_500_when_mastering_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    def fake_master_recording(vocal_path, instrumental_path, output_dir):
        raise RuntimeError("ffmpeg failed to clean vocal: boom")

    monkeypatch.setattr(server, "master_recording", fake_master_recording)

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"fake vocal bytes", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 500
```

Change the imports at the top of `tests/test_server.py` from:

```python
import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
```

to:

```python
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_server.py::test_render_recording_mp3_masters_both_uploaded_tracks_and_saves_the_take tests/test_server.py::test_render_recording_mp3_rejects_an_empty_track tests/test_server.py::test_render_recording_mp3_returns_500_when_mastering_fails -v`
Expected: all three FAIL. The first two FAIL with `AttributeError: <module 'server'> does not have the attribute 'master_recording'` (raised by `monkeypatch.setattr`, since `server.py` doesn't import that name yet). The old endpoint still only accepts a raw body (no `File`/`Form` params), so the empty-track test's multipart-encoded request body is actually non-empty (it's the full multipart envelope, boundaries and all, even though the `vocal` file's own content is empty) — the old code reads it as one non-empty blob, tries to transcode it as audio, and fails with a 500 from ffmpeg, not the 400 the test expects: `AssertionError: assert 500 == 400`. The third test FAILs the same way as the first (`AttributeError`) since `master_recording` isn't monkeypatchable yet either.

- [ ] **Step 4: Implement**

Change the fastapi import at the top of `scripts/server.py` from:

```python
from fastapi import FastAPI, HTTPException, Request
```

to:

```python
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
```

Add a new import line right after the existing `from audio_pipeline.download import download_audio, probe_title  # noqa: E402` line:

```python
from audio_pipeline.mastering import master_recording  # noqa: E402
```

Replace the entire `render_recording_mp3` function (currently):

```python
@app.post("/api/recordings/mp3")
async def render_recording_mp3(request: Request, song_id: str = "recording") -> FileResponse:
    """Transcode a recorded webm blob (the browser's MediaRecorder output --
    already mixed voice + instrumental client-side, see
    frontend/src/hooks/useRecording.ts) into an mp3, save it under
    RECORDINGS_DIR so it shows up in the "My recordings" list later, and
    return it directly so the browser can also download it immediately.
    """
    webm_bytes = await request.body()
    if not webm_bytes:
        raise HTTPException(status_code=400, detail="Empty recording")

    slug = slugify(song_id)
    recorded_at = datetime.now(timezone.utc)
    epoch_seconds = int(recorded_at.timestamp())
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = RECORDINGS_DIR / f"{slug}__{epoch_seconds}.mp3"

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.webm"
        input_path.write_bytes(webm_bytes)
        try:
            mp3_path = transcode_to_mp3(input_path, Path(tmp_dir))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        shutil.copy(mp3_path, saved_path)

    return FileResponse(
        saved_path,
        media_type="audio/mpeg",
        filename=_recording_download_name(slug, epoch_seconds),
    )
```

with:

```python
@app.post("/api/recordings/mp3")
async def render_recording_mp3(
    vocal: UploadFile = File(...),
    instrumental: UploadFile = File(...),
    song_id: str = Form("recording"),
) -> FileResponse:
    """Master (denoise/balance/align -- see audio_pipeline/mastering.py) and
    transcode a recorded take -- the browser's separately-recorded vocal and
    instrumental MediaRecorder tracks, see frontend/src/hooks/useRecording.ts
    -- into an mp3, save it under RECORDINGS_DIR so it shows up in the "My
    recordings" list later, and return it directly so the browser can also
    download it immediately.
    """
    vocal_bytes = await vocal.read()
    instrumental_bytes = await instrumental.read()
    if not vocal_bytes or not instrumental_bytes:
        raise HTTPException(status_code=400, detail="Empty recording")

    slug = slugify(song_id)
    recorded_at = datetime.now(timezone.utc)
    epoch_seconds = int(recorded_at.timestamp())
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = RECORDINGS_DIR / f"{slug}__{epoch_seconds}.mp3"

    with tempfile.TemporaryDirectory() as tmp_dir:
        vocal_path = Path(tmp_dir) / "vocal.webm"
        instrumental_path = Path(tmp_dir) / "instrumental.webm"
        vocal_path.write_bytes(vocal_bytes)
        instrumental_path.write_bytes(instrumental_bytes)
        try:
            mastered_path = master_recording(vocal_path, instrumental_path, Path(tmp_dir) / "mastered")
            mp3_path = transcode_to_mp3(mastered_path, Path(tmp_dir))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        shutil.copy(mp3_path, saved_path)

    return FileResponse(
        saved_path,
        media_type="audio/mpeg",
        filename=_recording_download_name(slug, epoch_seconds),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_server.py -v`
Expected: all tests PASS, including the three new ones.

- [ ] **Step 6: Commit**

```bash
git add scripts/server.py requirements.txt tests/test_server.py
git commit -m "Wire mastering into the recording upload endpoint (multipart, two tracks)"
```

---

### Task 7: `useRecording.ts` — record two tracks instead of one

**Files:**
- Modify: `frontend/src/hooks/useRecording.ts`

**Interfaces:**
- Consumes: `POST /api/recordings/mp3` accepting multipart fields `vocal`, `instrumental`, `song_id` from Task 6.
- Produces: `useRecording`'s existing exported shape (`UseRecordingResult`) is unchanged — same `status`/`downloadUrl`/`downloadFilename`/`errorMessage`/`start`/`stop`/`reset`. No other file imports from this hook's internals, so no other frontend file needs to change.

No automated test for this step — this project has no browser-API mocking infrastructure for hooks that use `AudioContext`/`MediaRecorder`/`getUserMedia` (see Global Constraints). Verify manually in Task 9.

- [ ] **Step 1: Replace the file**

Replace the entire contents of `frontend/src/hooks/useRecording.ts` with:

```typescript
import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { parseContentDispositionFilename } from '../game/recordings'

export type RecordingStatus = 'idle' | 'recording' | 'processing' | 'done' | 'error'

interface UseRecordingOptions {
  audioRef: RefObject<HTMLAudioElement | null>
  micStreamRef: RefObject<MediaStream | null>
  songId: string
}

export interface UseRecordingResult {
  status: RecordingStatus
  downloadUrl: string | null
  downloadFilename: string | null
  errorMessage: string | null
  start: () => Promise<void>
  stop: () => void
  reset: () => void
}

const FALLBACK_DOWNLOAD_FILENAME = 'recording.mp3'

// Browsers can't record straight to mp3 -- MediaRecorder only speaks the codecs the browser
// itself ships (webm/opus here). The actual mp3 is rendered server-side by the local job server
// via ffmpeg (already a hard dependency of this project, see NOTES.md), so no browser-side mp3
// encoder library is needed at all.
const PREFERRED_MIME_TYPE = 'audio/webm;codecs=opus'

/** Records the instrumental and live mic as two separate takes and uploads both to the local
 * job server, which auto-balances/cleans up the vocal and mixes them down into a single
 * downloadable mp3 (see audio_pipeline/mastering.py) -- so a player's voice comes through
 * clearly against the music instead of a raw, unprocessed mix. Each source is routed through
 * its own `MediaStreamAudioDestinationNode` via the Web Audio API and recorded with its own
 * `MediaRecorder`, started back-to-back so both tracks begin at effectively the same time (see
 * docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md's "Recording start offset"
 * section for why the two tracks can still end up slightly misaligned regardless, and how that's
 * corrected server-side). Recording is manual (start/stop buttons), not tied to playback,
 * matching how the mic itself is manually enabled. */
export function useRecording({ audioRef, micStreamRef, songId }: UseRecordingOptions): UseRecordingResult {
  const [status, setStatus] = useState<RecordingStatus>('idle')
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [downloadFilename, setDownloadFilename] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  // A media element can only ever be handed to `createMediaElementSource` once in its lifetime
  // (a second call throws) -- cached here so re-recording, or recording again after switching
  // songs, doesn't attempt that twice against the same underlying <audio> DOM node.
  const elementSourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const vocalRecorderRef = useRef<MediaRecorder | null>(null)
  const instrumentalRecorderRef = useRef<MediaRecorder | null>(null)
  const vocalChunksRef = useRef<BlobPart[]>([])
  const instrumentalChunksRef = useRef<BlobPart[]>([])
  // Each MediaRecorder's `onstop` fires independently and asynchronously -- this counts how
  // many of the two have actually finished flushing their last chunk, so the upload only
  // starts once both blobs are complete.
  const stoppedCountRef = useRef(0)
  const downloadUrlRef = useRef<string | null>(null)

  const revokeDownloadUrl = useCallback(() => {
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current)
      downloadUrlRef.current = null
    }
  }, [])

  const renderMp3 = useCallback(async () => {
    setStatus('processing')
    try {
      const vocalBlob = new Blob(vocalChunksRef.current, { type: 'audio/webm' })
      const instrumentalBlob = new Blob(instrumentalChunksRef.current, { type: 'audio/webm' })
      const formData = new FormData()
      formData.append('vocal', vocalBlob, 'vocal.webm')
      formData.append('instrumental', instrumentalBlob, 'instrumental.webm')
      formData.append('song_id', songId)

      const response = await fetch('/api/recordings/mp3', { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`Server responded ${response.status}`)

      const filename =
        parseContentDispositionFilename(response.headers.get('Content-Disposition')) ??
        FALLBACK_DOWNLOAD_FILENAME
      const mp3Blob = await response.blob()
      revokeDownloadUrl()
      const url = URL.createObjectURL(mp3Blob)
      downloadUrlRef.current = url
      setDownloadUrl(url)
      setDownloadFilename(filename)
      setStatus('done')
    } catch {
      setErrorMessage("Couldn't render the MP3 — make sure scripts/server.py is running.")
      setStatus('error')
    }
  }, [songId, revokeDownloadUrl])

  const start = useCallback(async () => {
    const audio = audioRef.current
    const micStream = micStreamRef.current
    if (!audio || !micStream) {
      setErrorMessage('Enable the mic before recording.')
      setStatus('error')
      return
    }

    setErrorMessage(null)
    revokeDownloadUrl()
    setDownloadUrl(null)
    setDownloadFilename(null)

    const context = audioContextRef.current ?? new AudioContext()
    audioContextRef.current = context
    await context.resume()

    let elementSource = elementSourceRef.current
    if (!elementSource) {
      // Routing the <audio> element through Web Audio "hijacks" its output -- it must be
      // reconnected to the context's own destination or the song goes silent for the listener.
      elementSource = context.createMediaElementSource(audio)
      elementSource.connect(context.destination)
      elementSourceRef.current = elementSource
    }

    const vocalDestination = context.createMediaStreamDestination()
    context.createMediaStreamSource(micStream).connect(vocalDestination)

    const instrumentalDestination = context.createMediaStreamDestination()
    elementSource.connect(instrumentalDestination)

    const mimeType = MediaRecorder.isTypeSupported(PREFERRED_MIME_TYPE) ? PREFERRED_MIME_TYPE : undefined
    const vocalRecorder = new MediaRecorder(vocalDestination.stream, mimeType ? { mimeType } : undefined)
    const instrumentalRecorder = new MediaRecorder(instrumentalDestination.stream, mimeType ? { mimeType } : undefined)

    vocalChunksRef.current = []
    instrumentalChunksRef.current = []
    stoppedCountRef.current = 0

    vocalRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) vocalChunksRef.current.push(event.data)
    }
    instrumentalRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) instrumentalChunksRef.current.push(event.data)
    }

    const onEitherStop = () => {
      stoppedCountRef.current += 1
      if (stoppedCountRef.current === 2) void renderMp3()
    }
    vocalRecorder.onstop = onEitherStop
    instrumentalRecorder.onstop = onEitherStop

    vocalRecorderRef.current = vocalRecorder
    instrumentalRecorderRef.current = instrumentalRecorder
    // Started back-to-back (no `await` between the two calls) so both tracks begin capturing
    // at effectively the same time.
    vocalRecorder.start()
    instrumentalRecorder.start()
    setStatus('recording')
  }, [audioRef, micStreamRef, revokeDownloadUrl, renderMp3])

  const stop = useCallback(() => {
    vocalRecorderRef.current?.stop()
    instrumentalRecorderRef.current?.stop()
  }, [])

  const reset = useCallback(() => {
    revokeDownloadUrl()
    setDownloadUrl(null)
    setDownloadFilename(null)
    setErrorMessage(null)
    setStatus('idle')
  }, [revokeDownloadUrl])

  useEffect(() => revokeDownloadUrl, [revokeDownloadUrl])

  return { status, downloadUrl, downloadFilename, errorMessage, start, stop, reset }
}
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useRecording.ts
git commit -m "Record the instrumental and mic as two separate tracks for mastering"
```

---

### Task 8: Offset-measurement helper and calibration script

**Files:**
- Modify: `audio_pipeline/mastering.py`
- Create: `scripts/measure_recording_offset.py`
- Test: `tests/test_mastering.py`

**Interfaces:**
- Consumes: nothing from other tasks (independent of `master_recording`'s own pipeline; a standalone calibration tool).
- Produces: `measure_start_offset(vocal_path: str | Path, instrumental_path: str | Path) -> float`. Task 9 uses this (via the CLI script) to determine the real value for `_RECORDING_OFFSET_SECONDS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mastering.py`:

```python
def test_measure_start_offset_detects_a_synthetic_lag(tmp_path):
    samplerate = 44100
    duration_s = 1.0
    lag_seconds = 0.3

    click = np.zeros(int(duration_s * samplerate), dtype=np.float32)
    click_start = int(0.1 * samplerate)
    click[click_start:click_start + 200] = 0.9  # sharp transient

    lag_samples = int(lag_seconds * samplerate)
    lagged_click = np.concatenate([np.zeros(lag_samples, dtype=np.float32), click])

    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, lagged_click, samplerate)
    sf.write(instrumental_path, click, samplerate)

    offset = measure_start_offset(vocal_path, instrumental_path)
    assert offset == pytest.approx(lag_seconds, abs=0.05)
```

Change the import line at the top of `tests/test_mastering.py` from:

```python
from audio_pipeline.mastering import (
    _apply_loudnorm,
    _clean_vocal,
    _correct_start_offset,
    _measure_loudness,
    master_recording,
)
```

to:

```python
from audio_pipeline.mastering import (
    _apply_loudnorm,
    _clean_vocal,
    _correct_start_offset,
    _measure_loudness,
    master_recording,
    measure_start_offset,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py::test_measure_start_offset_detects_a_synthetic_lag -v`
Expected: FAIL with `ImportError: cannot import name 'measure_start_offset'`.

- [ ] **Step 3: Implement**

Append to `audio_pipeline/mastering.py`:

```python
def measure_start_offset(vocal_path: str | Path, instrumental_path: str | Path) -> float:
    """Measures the real start offset between a raw two-track recording's
    vocal and instrumental tracks, for tuning ``_RECORDING_OFFSET_SECONDS``
    (see that constant's docstring and scripts/measure_recording_offset.py,
    which wraps this for one-off manual calibration against a real
    recording).

    Expects a take where the singer claps once, sharply, right on the song's
    first strong instrumental hit -- the clap is the vocal track's first
    onset; that instrumental hit is the instrumental track's first onset.
    Returns ``vocal_onset - instrumental_onset`` (positive = the vocal lags,
    matching ``_RECORDING_OFFSET_SECONDS``'s sign convention).
    """
    import librosa  # local import: only this one-off calibration path needs it

    def first_onset_seconds(path: str | Path) -> float:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
        if len(onset_frames) == 0:
            raise ValueError(f"No onset detected in {path}")
        return float(librosa.frames_to_time(onset_frames[0], sr=sr))

    return first_onset_seconds(vocal_path) - first_onset_seconds(instrumental_path)
```

Create `scripts/measure_recording_offset.py`:

```python
"""One-off calibration: measures the constant start-offset between a raw
two-track recording's vocal and instrumental tracks, so
audio_pipeline/mastering.py's _RECORDING_OFFSET_SECONDS can be set from a
real measurement instead of a guess (see that constant's docstring, and the
"Recording start offset" section of
docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md).

Procedure:
1. Temporarily record a take where you clap once, sharply, right on the
   song's first strong instrumental hit (e.g. the first beat after any
   count-in).
2. Get the two raw (pre-mastering) tracks for that take as wav files -- e.g.
   temporarily add two `shutil.copy(...)` lines in
   scripts/server.py's render_recording_mp3, right after it writes
   vocal_path/instrumental_path, copying them out to a scratch directory
   before mastering runs; remove those lines again once done.
3. Run:
       venv/Scripts/python.exe scripts/measure_recording_offset.py <vocal.wav> <instrumental.wav>
4. Update _RECORDING_OFFSET_SECONDS in audio_pipeline/mastering.py with the
   printed value.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.mastering import measure_start_offset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vocal_wav", type=Path)
    parser.add_argument("instrumental_wav", type=Path)
    args = parser.parse_args()

    offset = measure_start_offset(args.vocal_wav, args.instrumental_wav)
    print(f"Measured offset: {offset:+.3f}s")
    print("(positive = vocal lags instrumental -- set _RECORDING_OFFSET_SECONDS in")
    print(" audio_pipeline/mastering.py to this value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_pipeline/mastering.py scripts/measure_recording_offset.py tests/test_mastering.py
git commit -m "Add a start-offset measurement helper and calibration script"
```

---

### Task 9: Measure the real offset, run the full suite, manual verification, NOTES.md

**Files:**
- Modify: `audio_pipeline/mastering.py`
- Modify: `NOTES.md`

**Interfaces:**
- Consumes: the fully working feature from Tasks 1-8.
- Produces: `_RECORDING_OFFSET_SECONDS` set to a real measured value (or left at `0.0` with the measurement documented as inconclusive/deferred, if a human isn't available to do the mic test in this session — but the code path must never ship with a *guessed* non-zero value), and a NOTES.md entry documenting all manual verification.

- [ ] **Step 1: Run the full automated test suite**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all tests pass.

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no TypeScript errors.

Run: `cd frontend && npm test`
Expected: all existing tests still pass (this feature added no new frontend tests, per Task 7's note — confirm nothing existing regressed).

- [ ] **Step 2: Measure the real recording start-offset**

Start the job server (`venv/Scripts/python.exe scripts/server.py`) and the frontend dev server (`cd frontend && npm run dev`). Temporarily add two debug lines in `scripts/server.py`'s `render_recording_mp3`, right after `instrumental_path.write_bytes(instrumental_bytes)`:

```python
        shutil.copy(vocal_path, "debug_vocal.webm")
        shutil.copy(instrumental_path, "debug_instrumental.webm")
```

In a real (non-automated) browser, load any song, enable the mic, start recording, clap once sharply right on the first strong instrumental beat, then stop recording after a few seconds.

Convert the two debug webm files to wav (reusing the transcode helper from Task 1) and run the calibration script:

```
venv/Scripts/python.exe -c "from audio_pipeline.transcode import transcode_to_wav; transcode_to_wav('debug_vocal.webm', '.'); transcode_to_wav('debug_instrumental.webm', '.')"
venv/Scripts/python.exe scripts/measure_recording_offset.py debug_vocal.wav debug_instrumental.wav
```

Note the printed offset. Remove the two debug lines from `scripts/server.py` and delete the four `debug_*` files (they're test artifacts, not meant to be committed).

- [ ] **Step 3: Apply the measured offset**

In `audio_pipeline/mastering.py`, change:

```python
_RECORDING_OFFSET_SECONDS = 0.0
```

to the measured value from Step 2 (e.g. `_RECORDING_OFFSET_SECONDS = 0.087`), keeping the existing comment above it (still accurate — just update the constant, not the comment, unless the sign came out negative, in which case adjust the comment's "positive = lags" framing to state which direction this specific measurement went).

Run: `venv/Scripts/python.exe -m pytest tests/test_mastering.py -v`
Expected: all tests still PASS (the offset-correction tests all pass an explicit `offset_seconds` argument, so they're unaffected by the default changing).

- [ ] **Step 4: Manual listen-through**

In the same real browser session, record a fresh take (singing along for real this time, not just a calibration clap) and download the resulting mp3. Confirm by ear:
- The vocal is clearly audible and sounds cleaner/less noisy than an unprocessed mic recording would.
- The vocal doesn't drown out the instrumental, and isn't drowned out by it.
- The recording starts in sync — no perceptible beat-lag between when you started singing and where your voice sits against the music.
- No clipping/distortion artifacts.

- [ ] **Step 5: Record the outcome in NOTES.md**

Append a dated entry ("## Auto-balance and clean up recordings (recorded <date>)") documenting: what was built (two-track recording, server-side mastering — align/denoise/compress/loudnorm/mix/limit), the measured `_RECORDING_OFFSET_SECONDS` value from Step 2, and the Step 4 listen-through results (or, if no human listen was possible in this session, state that explicitly and flag it as the outstanding follow-up, per this project's own established convention for real-audio checks it couldn't complete — see NOTES.md's many existing "needs a human" entries for the exact tone/pattern to follow).

- [ ] **Step 6: Commit**

```bash
git add audio_pipeline/mastering.py NOTES.md
git commit -m "Measure and apply the real recording start-offset; verify end-to-end"
```
