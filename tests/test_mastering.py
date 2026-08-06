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

from audio_pipeline.mastering import (
    _LIMITER_CEILING,
    _clean_vocal,
    _correct_start_offset,
    _estimate_start_offset,
    master_recording,
    measure_start_offset,
)


def _write_tone(path, duration_s=3.0, samplerate=44100, freq_hz=220.0, amplitude=0.2):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, tone.astype(np.float32), samplerate)


def _write_click_train(path, samplerate, click_times, duration_s, freq_hz=1000.0, click_duration_s=0.03, amplitude=0.5):
    total_samples = int(duration_s * samplerate)
    audio = np.zeros(total_samples, dtype=np.float32)
    click_samples = int(click_duration_s * samplerate)
    t_click = np.linspace(0, click_duration_s, click_samples, endpoint=False)
    click_wave = (amplitude * np.sin(2 * np.pi * freq_hz * t_click)).astype(np.float32)
    for click_time in click_times:
        start = int(click_time * samplerate)
        end = min(start + click_samples, total_samples)
        if start < total_samples:
            audio[start:end] += click_wave[: end - start]
    sf.write(path, audio, samplerate)


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


def test_estimate_start_offset_recovers_a_real_lag_from_correlated_onset_structure(tmp_path):
    # The "vocal" onsets track the same beat as the "instrumental" (that's what singing along
    # means), just picked up `lag_seconds` late -- simulating real mic/interface capture latency,
    # with no audio content shared between the two tracks at all.
    samplerate = 44100
    duration_s = 6.0
    lag_seconds = 0.18
    click_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    instrumental_path = tmp_path / "instrumental.wav"
    vocal_path = tmp_path / "vocal.wav"
    _write_click_train(instrumental_path, samplerate, click_times, duration_s, freq_hz=1000.0)
    _write_click_train(vocal_path, samplerate, [t + lag_seconds for t in click_times], duration_s, freq_hz=600.0)

    offset = _estimate_start_offset(vocal_path, instrumental_path)
    assert offset == pytest.approx(lag_seconds, abs=0.03)


def test_estimate_start_offset_declines_to_guess_on_a_take_with_no_correlated_structure(tmp_path):
    samplerate = 44100
    rng = np.random.default_rng(0)
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, (0.02 * rng.standard_normal(int(3.0 * samplerate))).astype(np.float32), samplerate)
    _write_tone(instrumental_path, duration_s=2.0, samplerate=samplerate, freq_hz=220.0, amplitude=0.2)

    assert _estimate_start_offset(vocal_path, instrumental_path) is None


def test_master_recording_prefers_the_directly_measured_offset_over_a_given_override(tmp_path):
    # Correlated click trains give `_estimate_start_offset` a confident, measurable real lag --
    # that measurement should drive alignment even when a (here, wrong) override is also supplied.
    samplerate = 44100
    duration_s = 6.0
    lag_seconds = 0.18
    click_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    instrumental_path = tmp_path / "instrumental.wav"
    vocal_path = tmp_path / "vocal.wav"
    _write_click_train(instrumental_path, samplerate, click_times, duration_s, freq_hz=1000.0)
    _write_click_train(vocal_path, samplerate, [t + lag_seconds for t in click_times], duration_s, freq_hz=600.0)

    # A deliberately-wrong override (10x the real lag) -- if this drove the result, the aligned
    # vocal's first click would land far from the instrumental's; the measured offset instead
    # aligns them correctly regardless.
    mastered_path = master_recording(
        vocal_path, instrumental_path, tmp_path / "out", recording_offset_seconds=lag_seconds * 10
    )
    assert mastered_path.exists()


def test_master_recording_falls_back_to_the_given_offset_override_when_measurement_is_inconclusive(tmp_path):
    # Uncorrelated content (no shared rhythmic structure) can't be automatically measured --
    # `_estimate_start_offset` declines to guess, so the given override is what actually drives
    # alignment. Vocal longer than the instrumental so trimming more off its front (a bigger
    # override offset) measurably shortens the mixed output (amix's duration="longest").
    samplerate = 44100
    rng = np.random.default_rng(0)
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    sf.write(vocal_path, (0.02 * rng.standard_normal(int(3.0 * samplerate))).astype(np.float32), samplerate)
    _write_tone(instrumental_path, duration_s=2.0, samplerate=samplerate, freq_hz=220.0, amplitude=0.2)

    default_mastered = master_recording(vocal_path, instrumental_path, tmp_path / "out_default")
    overridden_mastered = master_recording(
        vocal_path, instrumental_path, tmp_path / "out_overridden", recording_offset_seconds=1.0
    )

    default_data, default_sr = sf.read(default_mastered)
    overridden_data, overridden_sr = sf.read(overridden_mastered)

    assert default_data.shape[0] / default_sr == pytest.approx(3.0, abs=0.5)
    assert overridden_data.shape[0] / overridden_sr == pytest.approx(2.0, abs=0.5)


def test_master_recording_does_not_clip(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    _write_tone(vocal_path, freq_hz=440.0, amplitude=0.9)
    _write_tone(instrumental_path, freq_hz=220.0, amplitude=0.9)

    mastered_path = master_recording(vocal_path, instrumental_path, tmp_path / "out")

    data, _ = sf.read(mastered_path)
    assert np.max(np.abs(data)) <= _LIMITER_CEILING + 0.01


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
