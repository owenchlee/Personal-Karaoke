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

from audio_pipeline.melody_extraction import extract_melody

# basic-pitch's default minimum/maximum frequency bounds correspond
# (loosely) to this MIDI range; a plausible sung vocal note should fall
# well within it.
_PLAUSIBLE_VOCAL_MIDI_RANGE = (21, 108)


def _write_synthetic_clip(path, duration_s=2.0, samplerate=22050, freq_hz=220.0):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq_hz * t)
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
