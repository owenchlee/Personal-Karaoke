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

import librosa
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


# basic-pitch's octave confusion (see `_enforce_monophony` above) doesn't only produce
# *simultaneous* overlapping notes -- it also sometimes drops a brief same-pitch-class note an
# octave away in the middle of an otherwise-steady held note, with basic-pitch reporting it as a
# separate, sequential (non-overlapping) note rather than an overlapping one. Measured directly on
# `cache/test-song/notes.json` (450 notes, post-monophony): filtering for a note `B` sitting
# between two touching neighbors `A`/`C` (zero-gap on both sides -- i.e. basic-pitch itself
# detected no silence there, so this reads as one continuous vocalization it chopped into three)
# whose pitches are within a semitone of each other, with `B` at least 7 semitones from both, found
# exactly 8 clean cases, with `B`'s duration ranging 0.10-0.61s and, notably, no consistent
# confidence (velocity) pattern distinguishing `B` from its neighbors (one case had `B`'s velocity
# *higher* than both neighbors) -- confirming, same as the RMS-silence-gate finding elsewhere in
# this file, that basic-pitch's own confidence isn't a reliable signal here either. The touching
# (near-zero gap) condition is the actual discriminator, not duration or velocity.
_SANDWICH_MAX_GAP_S = 0.02
_SANDWICH_MIN_JUMP_SEMITONES = 7
_SANDWICH_MAX_NEIGHBOR_DIFF_SEMITONES = 1
# A generous sanity cap, not a fitted threshold -- every measured real case was well under this;
# it exists only to stop the pass from ever swallowing an implausibly long "blip".
_SANDWICH_MAX_DURATION_S = 1.0


def _collapse_octave_blips(notes: list[dict]) -> list[dict]:
    """Absorb a short, touching, octave-jump-and-back note into its preceding neighbor.

    Notes must already be monophonic (see `_enforce_monophony`) and sorted by onset. When a note
    `B` sits with (near-)zero gap between two neighbors `A`/`C` of matching pitch, and jumps away
    from and back to that pitch by at least `_SANDWICH_MIN_JUMP_SEMITONES`, it's treated as a
    transcription artifact of the single held note around it rather than a real quick leap -- `B`
    is dropped and the preceding note (`A`) is extended to cover its span, so no time coverage is
    lost. If `A` and `C` are then exactly the same pitch (not just within tolerance -- measured on
    the test song, 5 of 8 real cases were an exact match, the rest one semitone apart), they're the
    same held note on either side of the artifact, so `C` is folded into the extended `A` too,
    rather than left as a separate but touching same-pitch note.
    """
    if len(notes) < 3:
        return notes

    result = [dict(notes[0])]
    i = 1
    while i < len(notes) - 1:
        prev = result[-1]
        current = notes[i]
        nxt = notes[i + 1]

        gap_before = current["onset"] - prev["offset"]
        gap_after = nxt["onset"] - current["offset"]
        neighbor_diff = abs(prev["pitch_midi"] - nxt["pitch_midi"])
        jump_before = abs(current["pitch_midi"] - prev["pitch_midi"])
        jump_after = abs(current["pitch_midi"] - nxt["pitch_midi"])

        is_blip = (
            gap_before <= _SANDWICH_MAX_GAP_S
            and gap_after <= _SANDWICH_MAX_GAP_S
            and neighbor_diff <= _SANDWICH_MAX_NEIGHBOR_DIFF_SEMITONES
            and jump_before >= _SANDWICH_MIN_JUMP_SEMITONES
            and jump_after >= _SANDWICH_MIN_JUMP_SEMITONES
            and current["duration"] <= _SANDWICH_MAX_DURATION_S
        )

        if is_blip:
            prev["offset"] = current["offset"]
            prev["duration"] = round(prev["offset"] - prev["onset"], 4)
            if nxt["pitch_midi"] == prev["pitch_midi"]:
                prev["offset"] = nxt["offset"]
                prev["duration"] = round(prev["offset"] - prev["onset"], 4)
                i += 2  # drop both current and nxt; resume after nxt next iteration
            else:
                i += 1  # drop current only; nxt stays a separate note
            continue

        result.append(dict(current))
        i += 1

    if i == len(notes) - 1:
        result.append(dict(notes[-1]))
    return result


# basic-pitch is a general-purpose (polyphonic) transcription model; pYIN is a monophonic pitch
# tracker purpose-built for exactly this kind of single-voice signal, and this codebase already
# trusted it once as independent ground truth (see the mapping-strategy comment on `getPitchRange`
# in the frontend's `coords.ts`, which cites a pYIN cross-check). Measured directly against
# `cache/test-song/notes.json` after the fixes above: 98/438 notes (22%) disagreed with pYIN's
# median pitch for that note's own time span by >=1.5 semitones, and 86 of those were *exact*
# octave disagreements -- basic-pitch's octave confusion isn't limited to the touching-blip
# signature `_collapse_octave_blips` targets; it also shows up as a single wrong-octave note with
# no matching neighbor to detect it against. Separately, of 185 notes with a real gap before the
# next note, 67 (36%) still had the *same pitch class* voiced by pYIN in the window right after the
# note's reported offset -- real singing continuing past where basic-pitch cut the note off.
#
# Fix: keep basic-pitch for note segmentation (onset/offset timing is its actual job), but use
# pYIN as a second opinion for the two things it's individually more reliable at -- the pitch
# within each segment, and whether the singer kept going past the reported offset.
_PYIN_HOP_LENGTH = 512
# Reuse the same plausible-vocal-range bounds already applied to basic-pitch above, so both
# trackers are constrained to the same register and can't disagree merely because one considered
# a pitch outside the other's search range.
_PYIN_FMIN_HZ = _MINIMUM_FREQUENCY_HZ
_PYIN_FMAX_HZ = _MAXIMUM_FREQUENCY_HZ
# A note needs at least this fraction of its frames voiced by pYIN before its pitch correction is
# trusted -- a very short or quiet note can have too few voiced frames for a reliable median, and
# in that case basic-pitch's original pitch (still the best available evidence) is left alone.
_PYIN_MIN_VOICED_FRACTION = 0.3
# How close (in pitch class, octave-agnostic) a pYIN frame must be to a note's corrected pitch to
# count as "still singing this note" when extending its offset.
_OFFSET_EXTEND_MATCH_SEMITONES = 1.5
# A cap, not a fitted threshold -- stops a misread from ever merging a note into the next phrase;
# real extensions measured on the test song were mostly well under this.
_OFFSET_EXTEND_MAX_SECONDS = 0.5
# How long a stretch of non-matching/silent steps is tolerated mid-scan before concluding the note
# really has ended -- see the extension loop below for why this exists (a single noisy pYIN frame
# is common and shouldn't be mistaken for the singer stopping). Matches the same order of
# magnitude as the frontend's own `HOLD_SECONDS` tolerance for the same kind of brief interruption.
_OFFSET_EXTEND_GAP_TOLERANCE_SECONDS = 0.15


def _pyin_track(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent monophonic pitch track: per-frame time and MIDI pitch (NaN where unvoiced)."""
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio, fmin=_PYIN_FMIN_HZ, fmax=_PYIN_FMAX_HZ, sr=sample_rate, hop_length=_PYIN_HOP_LENGTH
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=_PYIN_HOP_LENGTH)
    midi = np.where(voiced_flag, librosa.hz_to_midi(f0), np.nan)
    return times, midi


def _pyin_window(
    times: np.ndarray, midi: np.ndarray, start_s: float, end_s: float
) -> tuple[np.ndarray, int]:
    """Voiced MIDI values and total frame count (voiced + unvoiced) inside ``[start_s, end_s)``."""
    mask = (times >= start_s) & (times < end_s)
    window = midi[mask]
    return window[~np.isnan(window)], len(window)


def _pitch_class_distance(a, b: float) -> np.ndarray:
    diff = np.abs(a - b) % 12
    return np.minimum(diff, 12 - diff)


# `_collapse_octave_blips` only catches the *sandwiched* octave-artifact signature (near-zero gap
# on both sides, matching neighbor pitch). Measured directly against three real cached songs after
# every fix above (`_enforce_monophony`, `_collapse_octave_blips`, pitch correction): a small
# number of notes remain that basic-pitch reports with essentially zero independent confirmation
# from pYIN -- not just a low-confidence disagreement (the pitch-correction pass above already
# retunes those), but pYIN finding *no* voiced content at all in the note's own window, i.e. no
# evidence anything was actually sung there. On `test-song`, exactly 1 of 438 notes had a pYIN
# voiced fraction below 0.3, and it was also a >=7-semitone jump away from both neighbors and only
# 105ms long, at RMS 0.019 (barely above the silence gate) -- a short, near-silent, completely
# unconfirmed pitch spike, every signal pointing the same way.
#
# Voiced fraction alone is not a safe signal by itself, though: a Korean-language test song had 24
# notes with voiced fraction under 0.3, but most had normal RMS and less-than-7-semitone jumps from
# their neighbors -- real sung notes pYIN simply struggled to track (breathy delivery, heavy
# ornamentation), not artifacts. Requiring all three signs together -- near-zero independent
# confirmation, a large jump from a neighbor, and a short duration -- is what keeps the two failure
# modes apart: this combination caught the one real artifact in `test-song` and 15 of the 24
# flagged notes in the Korean song, while leaving the small-jump/normal-duration ones alone.
_SPURIOUS_MAX_VOICED_FRACTION = 0.3
_SPURIOUS_MIN_JUMP_SEMITONES = 7
_SPURIOUS_MAX_DURATION_S = 0.35


def _drop_unconfirmed_pitch_spikes(
    notes: list[dict], times: np.ndarray, midi: np.ndarray
) -> list[dict]:
    """Drop notes with no independent pYIN confirmation that also jump far from a neighbor and are
    short -- see the constants above for the evidence behind requiring all three together. Must run
    after pitch correction (so jumps are measured against corrected pitches, not basic-pitch's raw
    guesses).
    """
    kept = []
    for i, note in enumerate(notes):
        prev_pitch = notes[i - 1]["pitch_midi"] if i > 0 else None
        next_pitch = notes[i + 1]["pitch_midi"] if i + 1 < len(notes) else None
        jump_before = abs(note["pitch_midi"] - prev_pitch) if prev_pitch is not None else 0
        jump_after = abs(note["pitch_midi"] - next_pitch) if next_pitch is not None else 0

        voiced, total = _pyin_window(times, midi, note["onset"], note["offset"])
        voiced_fraction = len(voiced) / total if total > 0 else 0.0

        is_spurious = (
            note["duration"] <= _SPURIOUS_MAX_DURATION_S
            and voiced_fraction < _SPURIOUS_MAX_VOICED_FRACTION
            and (jump_before >= _SPURIOUS_MIN_JUMP_SEMITONES or jump_after >= _SPURIOUS_MIN_JUMP_SEMITONES)
        )
        if not is_spurious:
            kept.append(note)
    return kept


def _refine_with_pyin(notes: list[dict], audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Correct each note's pitch and extend its offset using an independent pYIN track, then drop
    any note left with no independent pitch confirmation at all.

    Notes must already be monophonic and onset-sorted (run after `_enforce_monophony` /
    `_collapse_octave_blips`). Pitch correction and offset extension are done as two separate
    passes over the full list, in that order, so extension always checks against each note's
    final (corrected) pitch rather than basic-pitch's original guess; dropping unconfirmed spikes
    runs last so it also sees each note's final (possibly-extended) duration.
    """
    if not notes:
        return notes

    times, midi = _pyin_track(audio, sample_rate)
    refined = [dict(note) for note in notes]
    audio_duration = len(audio) / sample_rate

    for note in refined:
        voiced, total = _pyin_window(times, midi, note["onset"], note["offset"])
        if total > 0 and len(voiced) / total >= _PYIN_MIN_VOICED_FRACTION:
            corrected_midi = int(round(float(np.median(voiced))))
            note["pitch_midi"] = corrected_midi
            note["pitch_hz"] = round(_midi_to_hz(corrected_midi), 2)

    step_s = _PYIN_HOP_LENGTH / sample_rate
    for i, note in enumerate(refined):
        next_onset = refined[i + 1]["onset"] if i + 1 < len(refined) else audio_duration
        max_offset = min(note["offset"] + _OFFSET_EXTEND_MAX_SECONDS, next_onset, audio_duration)

        # Walk forward in small steps, but don't give up the instant one step doesn't match --
        # measured directly on the real test song: a single noisy pYIN frame (a consonant, a
        # vibrato dip, one low-RMS zero-crossing) is common even in the middle of genuine
        # continued singing, and stopping on the very first bad step meant this pass barely
        # extended anything in practice (an early version of this loop moved the early-cutoff
        # rate by less than a percentage point on the real song, despite pYIN clearly still
        # tracking the right pitch class through many of those windows). Instead, keep scanning
        # through brief gaps -- same "brief interruptions shouldn't reset tracking" logic already
        # used for the live mic indicator's `HOLD_SECONDS` on the frontend -- and only actually
        # commit the offset up to the *last* step that genuinely matched, so a silent/mismatched
        # tail is never counted even if the scan kept looking a little further past it.
        extended_offset = note["offset"]
        last_match_time = note["offset"]
        t = note["offset"]
        while t < max_offset:
            window_end = min(t + step_s, max_offset)
            voiced, total = _pyin_window(times, midi, t, window_end)
            matches = (
                total > 0
                and len(voiced) > 0
                and np.any(_pitch_class_distance(voiced, note["pitch_midi"]) <= _OFFSET_EXTEND_MATCH_SEMITONES)
                and _note_rms(audio, sample_rate, t, window_end) >= _SILENCE_RMS_GATE
            )
            if matches:
                last_match_time = window_end
                extended_offset = window_end
            elif window_end - last_match_time > _OFFSET_EXTEND_GAP_TOLERANCE_SECONDS:
                break
            t = window_end

        note["offset"] = round(extended_offset, 4)
        note["duration"] = round(note["offset"] - note["onset"], 4)

    return _drop_unconfirmed_pitch_spikes(refined, times, midi)


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
    notes = _collapse_octave_blips(notes)
    notes = _refine_with_pyin(notes, audio, sample_rate)

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
