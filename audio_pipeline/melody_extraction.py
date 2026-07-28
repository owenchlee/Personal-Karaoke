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

import numpy as np
import pretty_midi
import soundfile as sf
from basic_pitch.inference import predict


@dataclass
class MelodyResult:
    midi_path: Path
    notes_path: Path


def _midi_to_hz(pitch_midi: int) -> float:
    return 440.0 * 2 ** ((pitch_midi - 69) / 12)


# Tuned against a real 274s test song (see NOTES.md "Melody extraction quality
# fix"). basic-pitch's defaults (onset=0.5, frame=0.3, melodia_trick=True)
# fragment a single held/vibrato note into many sub-200ms pieces and invent
# extra notes to bridge pitch bends. Disabling melodia_trick removes its
# note-inserting bridge behavior; lowering frame_threshold lets a real note
# bridge brief energy dips instead of dropping out. onset_threshold is kept
# at basic-pitch's own default -- raising it looked like it reduced
# fragmentation, but it was measurably suppressing real singing (checked by
# comparing note coverage against the vocal stem's own energy envelope: at
# onset=0.65 the game had gaps -- "no bar" -- over 15% of clearly-sung audio,
# worse than the 8% baseline). minimum_frequency/maximum_frequency bound the
# extraction to a plausible sung-vocal range (~C2-E6), rejecting low-frequency
# rumble/bleed picked up by the separator, without affecting real coverage.
_ONSET_THRESHOLD = 0.5
_FRAME_THRESHOLD = 0.2
_MINIMUM_NOTE_LENGTH_MS = 150.0
_MINIMUM_FREQUENCY_HZ = 65.0
_MAXIMUM_FREQUENCY_HZ = 1300.0

# basic-pitch's own confidence ("velocity") is not a reliable silence
# detector: on a real song's instrumental-only intro, where the vocal stem
# is near-total silence, it still reported moderate-confidence notes (e.g.
# velocity 0.3-0.5), which showed up in the game as note bars during
# instrumental-only sections. Gating each note against the vocal stem's own
# measured RMS energy fixes this directly. On the test song the split is
# stark and not threshold-sensitive: spurious notes during silence measured
# ~0.00004 RMS, while every real sung note measured >= 0.05 -- any gate
# between roughly 0.001 and 0.04 drops exactly the same 24 spurious notes
# and keeps all real ones, so 0.01 is a comfortable, non-fragile choice.
_SILENCE_RMS_GATE = 0.01


def _note_rms(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> float:
    start_sample = max(0, int(start_s * sample_rate))
    end_sample = min(len(audio), int(end_s * sample_rate))
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment**2)))


# One singer sings one pitch at a time, but basic-pitch is a polyphonic
# transcription model and doesn't know that -- it happily reports two
# simultaneous "notes" when a strong harmonic triggers its own pitch
# detection alongside the fundamental (58% of overlaps on the test song were
# exactly one octave apart, the classic signature of this). Enforced here as
# a strict monophonic constraint: notes are processed in onset order and
# never allowed to overlap the previously kept note.
#   - If the new note's span is fully inside the kept note's span, it's
#     almost always the harmonic artifact (by construction the kept note is
#     then the longer of the two) -- drop it.
#   - Otherwise it's a real melodic transition where the two detections
#     briefly overlap -- keep whichever has higher confidence (velocity) and
#     trim the other's boundary back to remove the overlap, dropping it
#     entirely if trimming would leave a sliver under _MIN_TRIMMED_DURATION_S.
_MIN_TRIMMED_DURATION_S = 0.05


def _enforce_monophony(notes: list[dict]) -> list[dict]:
    ordered = sorted((dict(note) for note in notes), key=lambda n: n["onset"])
    kept: list[dict] = []
    for note in ordered:
        while kept:
            last = kept[-1]
            if note["onset"] >= last["offset"]:
                break
            if note["offset"] <= last["offset"]:
                note = None
                break
            if note["velocity"] > last["velocity"]:
                last["offset"] = note["onset"]
                last["duration"] = round(last["offset"] - last["onset"], 4)
                if last["duration"] < _MIN_TRIMMED_DURATION_S:
                    kept.pop()
                    continue
                break
            else:
                note["onset"] = last["offset"]
                note["duration"] = round(note["offset"] - note["onset"], 4)
                if note["duration"] < _MIN_TRIMMED_DURATION_S:
                    note = None
                break
        if note is not None:
            kept.append(note)
    return kept


def extract_melody(vocal_stem_path: str | Path, output_dir: str | Path) -> MelodyResult:
    """Run basic-pitch pitch extraction on ``vocal_stem_path`` and save the
    resulting melody as both a MIDI file and a JSON note-event list inside
    ``output_dir``.

    Returns a ``MelodyResult`` with paths to both saved files.
    """
    vocal_stem_path = Path(vocal_stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _model_output, _midi_data, note_events = predict(
        vocal_stem_path,
        onset_threshold=_ONSET_THRESHOLD,
        frame_threshold=_FRAME_THRESHOLD,
        minimum_note_length=_MINIMUM_NOTE_LENGTH_MS,
        minimum_frequency=_MINIMUM_FREQUENCY_HZ,
        maximum_frequency=_MAXIMUM_FREQUENCY_HZ,
        melodia_trick=False,
    )

    audio, sample_rate = sf.read(vocal_stem_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    note_events = [
        event
        for event in note_events
        if _note_rms(audio, sample_rate, float(event[0]), float(event[1])) >= _SILENCE_RMS_GATE
    ]

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

    notes = _enforce_monophony(notes)

    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, round(note["velocity"] * 127)),
                pitch=note["pitch_midi"],
                start=note["onset"],
                end=note["offset"],
            )
        )
    midi_data.instruments.append(instrument)

    midi_path = output_dir / f"{vocal_stem_path.stem}_melody.mid"
    midi_data.write(str(midi_path))

    notes_path = output_dir / f"{vocal_stem_path.stem}_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))

    return MelodyResult(midi_path=midi_path, notes_path=notes_path)
