"""Tests for the shared two-pass ffmpeg loudnorm helpers.

Uses a synthetic tone rather than real audio -- same convention as
tests/test_mastering.py and tests/test_separation.py: proves the ffmpeg
pipeline runs and moves loudness in the right direction, not that it sounds
good on real music.
"""
import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.loudness import measure_loudness, normalize_loudness


def _write_tone(path, duration_s=3.0, samplerate=44100, freq_hz=220.0, amplitude=0.2):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, tone.astype(np.float32), samplerate)


def test_normalize_loudness_moves_measured_loudness_close_to_the_target(tmp_path):
    quiet_path = tmp_path / "quiet.wav"
    _write_tone(quiet_path, amplitude=0.02)

    output_path = tmp_path / "normalized.wav"
    normalize_loudness(quiet_path, output_path, target_i=-16)

    after_stats = measure_loudness(output_path, target_i=-16)
    assert float(after_stats["input_i"]) == pytest.approx(-16, abs=1.0)


def test_normalize_loudness_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.wav"
    bogus_path.write_bytes(b"this is not real audio")

    with pytest.raises(RuntimeError, match="ffmpeg failed to measure loudness"):
        normalize_loudness(bogus_path, tmp_path / "out.wav", target_i=-16)
