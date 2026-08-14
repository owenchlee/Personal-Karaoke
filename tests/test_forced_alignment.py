"""Smoke test for CTC forced alignment (see audio_pipeline/forced_alignment.py).

Uses the same class of synthetic non-speech clip test_melody_extraction.py and
test_separation.py use for their own smoke tests -- this proves the alignment
mechanics run end-to-end and produce one timespan per input token, not that
alignment is accurate against real singing (same caveat already established
throughout this project for every model-backed stage, see NOTES.md). Real
accuracy was checked manually against cached real vocal stems (English and
Cantonese) before this module was wired into the pipeline.
"""
import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.forced_alignment import align_tokens


def _write_synthetic_wav(path, duration_s=3.0, samplerate=16000, freq_hz=220.0):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, tone.astype(np.float32), samplerate)


def test_align_tokens_produces_one_increasing_timespan_per_input_token(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_synthetic_wav(vocal_path)

    result = align_tokens(["one", "two", "three"], vocal_path, "en")

    assert len(result) == 3
    for entry in result:
        assert entry.keys() == {"start", "end"}
        assert entry["start"] <= entry["end"]
    starts = [entry["start"] for entry in result]
    assert starts == sorted(starts)


def test_align_tokens_returns_empty_list_for_no_tokens(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_synthetic_wav(vocal_path)

    assert align_tokens([], vocal_path, "en") == []


def test_align_tokens_rejects_an_unsupported_language(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_synthetic_wav(vocal_path)

    with pytest.raises(ValueError, match="Unsupported language"):
        align_tokens(["hi"], vocal_path, "fr")


def test_align_tokens_reports_progress_ending_at_1(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_synthetic_wav(vocal_path)
    fractions = []

    align_tokens(["one", "two", "three"], vocal_path, "en", on_progress=fractions.append)

    assert fractions  # at least one intermediate call, not silent until the very end
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)  # monotonically increasing, never jumps backward


def test_align_tokens_does_not_report_progress_for_no_tokens(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_synthetic_wav(vocal_path)
    fractions = []

    assert align_tokens([], vocal_path, "en", on_progress=fractions.append) == []
    assert fractions == []


