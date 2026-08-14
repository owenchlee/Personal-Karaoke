"""Smoke tests for the key-change (pitch-shift) pipeline step.

Uses a synthetic tone rather than a real sung take -- same convention as tests/test_mastering.py:
proves the pyrubberband/soundfile pipeline runs end-to-end and moves pitch (not tempo) by the
requested amount, not that it sounds good on a real voice/instrumental mix.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.transpose import _midi_to_hz, transpose_song


def _write_tone(path, duration_s=2.0, samplerate=44100, freq_hz=220.0, amplitude=0.3):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, np.stack([tone, tone], axis=1).astype(np.float32), samplerate)


def _dominant_frequency(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1 / sr)
    return freqs[np.argmax(spectrum)]


def _write_notes(path, notes):
    path.write_text(json.dumps(notes))


def test_transpose_song_shifts_instrumental_pitch_by_requested_semitones(tmp_path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    _write_tone(song_dir / "instrumental.wav", freq_hz=220.0)
    _write_notes(song_dir / "notes.json", [])

    result = transpose_song(song_dir, semitones=3, output_dir=tmp_path / "shifted")

    assert result.instrumental_path.exists()
    expected_hz = 220.0 * 2 ** (3 / 12)
    shifted_hz = _dominant_frequency(result.instrumental_path)
    assert shifted_hz == pytest.approx(expected_hz, rel=0.03)


def test_transpose_song_preserves_duration(tmp_path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    _write_tone(song_dir / "instrumental.wav", duration_s=2.5, freq_hz=220.0)
    _write_notes(song_dir / "notes.json", [])

    result = transpose_song(song_dir, semitones=-4, output_dir=tmp_path / "shifted")

    original, sr_orig = sf.read(song_dir / "instrumental.wav")
    shifted, sr_shifted = sf.read(result.instrumental_path)
    assert len(shifted) / sr_shifted == pytest.approx(len(original) / sr_orig, abs=0.05)


def test_transpose_song_shifts_note_pitches_and_recomputes_hz_without_touching_timing(tmp_path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    _write_tone(song_dir / "instrumental.wav")
    notes = [
        {"pitch_midi": 60, "pitch_hz": _midi_to_hz(60), "onset": 0.0, "offset": 1.0, "duration": 1.0, "velocity": 0.8},
        {"pitch_midi": 67, "pitch_hz": _midi_to_hz(67), "onset": 1.0, "offset": 2.0, "duration": 1.0, "velocity": 0.6},
    ]
    _write_notes(song_dir / "notes.json", notes)

    result = transpose_song(song_dir, semitones=5, output_dir=tmp_path / "shifted")

    shifted_notes = json.loads(result.notes_path.read_text())
    assert [n["pitch_midi"] for n in shifted_notes] == [65, 72]
    for original, shifted in zip(notes, shifted_notes):
        assert shifted["pitch_hz"] == pytest.approx(_midi_to_hz(shifted["pitch_midi"]), abs=0.01)
        assert shifted["onset"] == original["onset"]
        assert shifted["offset"] == original["offset"]
        assert shifted["duration"] == original["duration"]


def test_transpose_song_is_a_no_op_shift_at_zero_semitones(tmp_path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    _write_tone(song_dir / "instrumental.wav", freq_hz=220.0)
    _write_notes(
        song_dir / "notes.json",
        [{"pitch_midi": 60, "pitch_hz": _midi_to_hz(60), "onset": 0.0, "offset": 1.0, "duration": 1.0, "velocity": 0.8}],
    )

    result = transpose_song(song_dir, semitones=0, output_dir=tmp_path / "shifted")

    shifted_notes = json.loads(result.notes_path.read_text())
    assert shifted_notes[0]["pitch_midi"] == 60

    original_hz = _dominant_frequency(song_dir / "instrumental.wav")
    shifted_hz = _dominant_frequency(result.instrumental_path)
    assert shifted_hz == pytest.approx(original_hz, rel=0.02)


def test_transpose_song_raises_a_clear_error_when_rubberband_binary_is_missing(tmp_path, monkeypatch):
    import audio_pipeline.transpose as transpose_module

    monkeypatch.setattr(transpose_module.shutil, "which", lambda name: None)

    song_dir = tmp_path / "song"
    song_dir.mkdir()
    _write_tone(song_dir / "instrumental.wav")
    _write_notes(song_dir / "notes.json", [])

    with pytest.raises(RuntimeError, match="rubberband"):
        transpose_song(song_dir, semitones=2, output_dir=tmp_path / "shifted")
