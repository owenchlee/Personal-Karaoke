"""Smoke test for the Demucs separation wrapper.

Uses a synthetic sine-wave clip rather than a real vocal recording -- this
only proves the pipeline runs end-to-end and produces valid audio files, not
that vocal isolation is musically accurate. Accuracy against real songs is a
manual check (see NOTES.md) once a real sample file is available.

First run downloads the ~80MB htdemucs model checkpoint and requires
internet access; subsequent runs use the local cache.
"""
import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.separation import separate_stems


def _write_synthetic_clip(path, duration_s=2.0, samplerate=44100, freq_hz=220.0):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * freq_hz * t)
    stereo = np.stack([tone, tone], axis=1).astype(np.float32)
    sf.write(path, stereo, samplerate)


def test_separate_stems_produces_readable_wavs(tmp_path):
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    vocals_path, instrumental_path = separate_stems(input_path, output_dir)

    for path in (vocals_path, instrumental_path):
        assert path.exists()
        assert path.suffix == ".wav"

        data, samplerate = sf.read(path)
        assert samplerate > 0
        assert data.shape[0] > 0
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        output_duration_s = data.shape[0] / samplerate
        assert output_duration_s == pytest.approx(input_duration_s, abs=0.5)


def test_instrumental_and_vocals_reconstruct_original_mix(tmp_path):
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    original, original_samplerate = sf.read(input_path)
    vocals_path, instrumental_path = separate_stems(input_path, output_dir)

    vocals, _ = sf.read(vocals_path)
    instrumental, samplerate = sf.read(instrumental_path)
    assert samplerate == original_samplerate

    # atol accounts for 16-bit PCM quantization on the saved wav files, not
    # just floating point rounding -- the reconstruction is exact in the
    # float32 tensors before demucs_api.save_audio quantizes them to disk.
    n = min(len(original), len(vocals), len(instrumental))
    reconstructed = vocals[:n] + instrumental[:n]
    np.testing.assert_allclose(reconstructed, original[:n], atol=1e-3)
