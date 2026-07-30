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

from audio_pipeline.separation import _progress_callback, separate_stems


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


def test_separate_stems_reports_progress_up_to_1(tmp_path):
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=2.0)

    progress_seen = []
    separate_stems(input_path, output_dir, on_progress=progress_seen.append)

    assert progress_seen, "on_progress was never called"
    assert progress_seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= fraction <= 1.0 for fraction in progress_seen)


def test_separate_stems_accepts_bs_roformer(tmp_path):
    # First run downloads the BS-RoFormer checkpoint (see NOTES.md's BS-RoFormer validation
    # entry for its size/timing) and requires internet access.
    # audio-separator's MDXC/BS-RoFormer path crashes on clips under 10s (a real bug in that
    # package, not this one -- "tensor a (0) must match tensor b" from its own overlap_add;
    # real songs are always well over 10s, so this only affects the synthetic test clip).
    input_duration_s = 12.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    vocals_path, instrumental_path = separate_stems(input_path, output_dir, model="bs_roformer")

    for path in (vocals_path, instrumental_path):
        assert path.exists()
        data, samplerate = sf.read(path)
        assert samplerate > 0
        assert data.shape[0] > 0


def test_progress_callback_only_reports_on_segment_end():
    seen = []
    callback = _progress_callback(seen.append)

    callback(
        {"state": "start", "segment_offset": 0, "audio_length": 1000, "model_idx_in_bag": 0, "models": 1}
    )
    assert seen == []

    callback(
        {"state": "end", "segment_offset": 500, "audio_length": 1000, "model_idx_in_bag": 0, "models": 1}
    )
    callback(
        {"state": "end", "segment_offset": 1000, "audio_length": 1000, "model_idx_in_bag": 0, "models": 1}
    )
    assert seen == [0.5, 1.0]


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


def test_progress_callback_accounts_for_multiple_submodels_in_a_bag():
    seen = []
    callback = _progress_callback(seen.append)

    callback(
        {"state": "end", "segment_offset": 1000, "audio_length": 1000, "model_idx_in_bag": 0, "models": 2}
    )
    callback(
        {"state": "end", "segment_offset": 1000, "audio_length": 1000, "model_idx_in_bag": 1, "models": 2}
    )
    assert seen == [0.5, 1.0]
