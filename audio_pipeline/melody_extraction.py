"""Vocal melody extraction via basic-pitch.

Given an isolated vocal stem (see audio_pipeline.separation), extract the
reference vocal melody. The primary deliverable is a JSON note-event list
(what the browser-side note highway will consume directly); a MIDI file is
also written as a secondary artifact for human inspection in a DAW. No
scoring against a user recording here -- that's a later phase.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from basic_pitch.inference import predict


@dataclass
class MelodyResult:
    midi_path: Path
    notes_path: Path


def _midi_to_hz(pitch_midi: int) -> float:
    return 440.0 * 2 ** ((pitch_midi - 69) / 12)


def extract_melody(vocal_stem_path: str | Path, output_dir: str | Path) -> MelodyResult:
    """Run basic-pitch pitch extraction on ``vocal_stem_path`` and save the
    resulting melody as both a MIDI file and a JSON note-event list inside
    ``output_dir``.

    Returns a ``MelodyResult`` with paths to both saved files.
    """
    vocal_stem_path = Path(vocal_stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _model_output, midi_data, note_events = predict(vocal_stem_path)

    midi_path = output_dir / f"{vocal_stem_path.stem}_melody.mid"
    midi_data.write(str(midi_path))

    notes = [
        {
            "pitch_midi": int(pitch_midi),
            "pitch_hz": round(_midi_to_hz(int(pitch_midi)), 2),
            "onset": round(float(start_s), 4),
            "offset": round(float(end_s), 4),
            "duration": round(float(end_s) - float(start_s), 4),
            "velocity": round(min(max(float(amplitude), 0.0), 1.0), 4),
        }
        for start_s, end_s, pitch_midi, amplitude, _pitch_bends in note_events
    ]

    notes_path = output_dir / f"{vocal_stem_path.stem}_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))

    return MelodyResult(midi_path=midi_path, notes_path=notes_path)
