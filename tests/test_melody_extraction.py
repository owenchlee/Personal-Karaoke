"""Smoke test for the basic-pitch melody extraction wrapper.

Uses a synthetic sine-wave clip rather than a real vocal recording -- this only
proves the pipeline runs end-to-end and produces valid MIDI/JSON output, not
that the extracted melody is musically accurate. Accuracy against a real vocal
stem is a manual check (see NOTES.md) once the Phase 0 sample stem is available.
"""
import json

import numpy as np
import pretty_midi
import soundfile as sf

from audio_pipeline.melody_extraction import _enforce_monophony, extract_melody

# basic-pitch's default minimum/maximum frequency bounds correspond
# (loosely) to this MIDI range; a plausible sung vocal note should fall
# well within it.
_PLAUSIBLE_VOCAL_MIDI_RANGE = (21, 108)


def _write_synthetic_clip(path, duration_s=2.0, samplerate=22050, freq_hz=220.0):
    # A pure sine has no harmonics, which is out-of-distribution for
    # basic-pitch's model (trained on real instrument/vocal timbres) and
    # produces low-confidence, fragmented detections regardless of
    # threshold tuning. Additive harmonics make this a much closer proxy
    # for a real sung note.
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    envelope = np.minimum(1.0, t * 20) * np.minimum(1.0, (duration_s - t) * 20)
    tone = np.zeros_like(t)
    for harmonic, amplitude in enumerate([1.0, 0.5, 0.3, 0.15, 0.08], start=1):
        tone += amplitude * np.sin(2 * np.pi * freq_hz * harmonic * t)
    tone = 0.3 * envelope * tone / np.max(np.abs(tone))
    sf.write(path, tone.astype(np.float32), samplerate)


def test_extract_melody_produces_readable_midi_and_json(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path)

    result = extract_melody(input_path, output_dir)

    assert result.midi_path.exists()
    assert result.midi_path.suffix == ".mid"
    midi_data = pretty_midi.PrettyMIDI(str(result.midi_path))
    all_notes = [note for instrument in midi_data.instruments for note in instrument.notes]
    assert len(all_notes) > 0

    assert result.notes_path.exists()
    assert result.notes_path.suffix == ".json"
    notes = json.loads(result.notes_path.read_text())
    assert isinstance(notes, list)
    assert len(notes) > 0

    low, high = _PLAUSIBLE_VOCAL_MIDI_RANGE
    for note in notes:
        assert note.keys() == {"pitch_midi", "pitch_hz", "onset", "offset", "duration", "velocity"}
        assert low <= note["pitch_midi"] <= high
        assert note["pitch_hz"] > 0
        assert note["onset"] < note["offset"]
        assert note["duration"] == round(note["offset"] - note["onset"], 4)
        assert 0.0 <= note["velocity"] <= 1.0

    byonset = sorted(notes, key=lambda n: n["onset"])
    for earlier, later in zip(byonset, byonset[1:]):
        assert later["onset"] >= earlier["offset"], "extracted notes must never overlap in time"


def _no_overlaps(notes):
    byonset = sorted(notes, key=lambda n: n["onset"])
    return all(b["onset"] >= a["offset"] for a, b in zip(byonset, byonset[1:]))


def test_enforce_monophony_drops_a_note_fully_contained_in_another():
    # A short "octave artifact" note sitting entirely inside a longer real note.
    notes = [
        {"pitch_midi": 52, "pitch_hz": 164.81, "onset": 0.0, "offset": 2.0, "duration": 2.0, "velocity": 0.5},
        {"pitch_midi": 64, "pitch_hz": 329.63, "onset": 0.5, "offset": 0.8, "duration": 0.3, "velocity": 0.9},
    ]
    result = _enforce_monophony(notes)
    assert _no_overlaps(result)
    assert [n["pitch_midi"] for n in result] == [52]


def test_enforce_monophony_trims_a_partial_overlap_in_favor_of_higher_velocity():
    notes = [
        {"pitch_midi": 52, "pitch_hz": 164.81, "onset": 0.0, "offset": 1.0, "duration": 1.0, "velocity": 0.3},
        {"pitch_midi": 55, "pitch_hz": 196.0, "onset": 0.8, "offset": 1.8, "duration": 1.0, "velocity": 0.9},
    ]
    result = _enforce_monophony(notes)
    assert _no_overlaps(result)
    assert len(result) == 2
    first, second = sorted(result, key=lambda n: n["onset"])
    assert first["offset"] == second["onset"] == 0.8


def test_enforce_monophony_handles_a_three_note_overlap_chain():
    notes = [
        {"pitch_midi": 50, "pitch_hz": 146.83, "onset": 0.0, "offset": 1.0, "duration": 1.0, "velocity": 0.9},
        {"pitch_midi": 52, "pitch_hz": 164.81, "onset": 0.9, "offset": 2.0, "duration": 1.1, "velocity": 0.2},
        {"pitch_midi": 54, "pitch_hz": 185.0, "onset": 0.95, "offset": 3.0, "duration": 2.05, "velocity": 0.95},
    ]
    result = _enforce_monophony(notes)
    assert _no_overlaps(result)
