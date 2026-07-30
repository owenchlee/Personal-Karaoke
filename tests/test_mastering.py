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

from audio_pipeline.mastering import _clean_vocal, _correct_start_offset


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


def test_clean_vocal_removes_low_frequency_rumble(tmp_path):
    duration_s = 2.0
    samplerate = 44100
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    voice = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    rumble = 0.2 * np.sin(2 * np.pi * 20.0 * t)  # well below the 90Hz highpass cutoff
    mixed = (voice + rumble).astype(np.float32)

    input_path = tmp_path / "noisy_vocal.wav"
    sf.write(input_path, mixed, samplerate)

    cleaned_path = _clean_vocal(input_path, tmp_path)
    cleaned, cleaned_sr = sf.read(cleaned_path)
    assert cleaned_sr == samplerate

    rumble_bin = int(round(20.0 * duration_s))
    original_rumble_energy = np.abs(np.fft.rfft(mixed))[rumble_bin]
    cleaned_rumble_energy = np.abs(np.fft.rfft(cleaned.astype(np.float32)))[rumble_bin]
    assert cleaned_rumble_energy < original_rumble_energy * 0.1


def test_clean_vocal_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.wav"
    bogus_path.write_bytes(b"this is not real audio")

    with pytest.raises(RuntimeError, match="ffmpeg failed to clean vocal"):
        _clean_vocal(bogus_path, tmp_path / "out")
