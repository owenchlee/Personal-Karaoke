"""Smoke test for the basic-pitch melody extraction wrapper.

Uses a synthetic sine-wave clip rather than a real vocal recording -- this only
proves the pipeline runs end-to-end and produces valid MIDI/JSON output, not
that the extracted melody is musically accurate. Accuracy against a real vocal
stem is a manual check (see NOTES.md) once the Phase 0 sample stem is available.
"""
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from audio_pipeline.melody_extraction import (
    _collapse_octave_blips,
    _drop_unconfirmed_pitch_spikes,
    _enforce_monophony,
    _pitch_class_distance,
    _pyin_track,
    _pyin_window,
    _refine_with_pyin,
    _repair_melody_gaps,
    _should_correct_note_pitch,
    extract_melody,
)

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


def _make_note(pitch_midi, onset, offset, velocity=0.5):
    return {
        "pitch_midi": pitch_midi,
        "pitch_hz": 440.0 * 2 ** ((pitch_midi - 69) / 12),
        "onset": onset,
        "offset": offset,
        "duration": round(offset - onset, 4),
        "velocity": velocity,
    }


def test_collapse_octave_blips_absorbs_a_touching_octave_jump_and_back():
    # A held note (pitch 60) basic-pitch chopped into three: itself, a brief octave-up blip, then
    # itself again, all touching with zero gaps -- the exact artifact _enforce_monophony's own
    # docstring describes for *overlapping* notes, but here sequential instead.
    notes = [
        _make_note(60, 0.0, 1.0),
        _make_note(72, 1.0, 1.2),
        _make_note(60, 1.2, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [60]
    assert result[0]["onset"] == 0.0
    assert result[0]["offset"] == 2.0
    assert result[0]["duration"] == 2.0


def test_collapse_octave_blips_leaves_a_real_silence_gap_alone():
    # Same pitch pattern, but a real gap before the blip -- basic-pitch detected actual silence
    # there, so this isn't one continuous vocalization getting mis-chopped; leave it alone.
    notes = [
        _make_note(60, 0.0, 1.0),
        _make_note(72, 1.3, 1.5),
        _make_note(60, 1.5, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [60, 72, 60]


def test_collapse_octave_blips_requires_neighbors_to_share_a_pitch_class():
    # The neighbors themselves are far apart in pitch, so the middle note isn't sandwiched between
    # two views of "the same real note" -- a real melodic passage, not an artifact.
    notes = [
        _make_note(60, 0.0, 1.0),
        _make_note(72, 1.0, 1.2),
        _make_note(65, 1.2, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [60, 72, 65]


def test_collapse_octave_blips_requires_a_large_enough_jump():
    # Only a small (real, plausible) melodic movement, not the kind of big away-and-back leap
    # that's the actual octave-confusion signature.
    notes = [
        _make_note(60, 0.0, 1.0),
        _make_note(63, 1.0, 1.2),
        _make_note(60, 1.2, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [60, 63, 60]


def test_collapse_octave_blips_leaves_short_note_lists_alone():
    notes = [_make_note(60, 0.0, 1.0), _make_note(72, 1.0, 1.2)]
    assert _collapse_octave_blips(notes) == notes


def test_collapse_octave_blips_keeps_neighbors_separate_when_not_exactly_equal_pitch():
    # Neighbors are close enough (within tolerance) to trigger the blip check, but not an exact
    # pitch match -- absorb the blip into the preceding note, but don't force the two neighbors
    # into one note; they may be a real, deliberate one-semitone re-attack.
    notes = [
        _make_note(60, 0.0, 1.0),
        _make_note(72, 1.0, 1.2),
        _make_note(61, 1.2, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [60, 61]
    assert result[0]["offset"] == 1.2
    assert result[1]["onset"] == 1.2


def _harmonic_tone(freq_hz, duration_s, samplerate=22050):
    # Same synthesis approach as `_write_synthetic_clip` above, but returns the raw array
    # directly (no file I/O) for feeding straight into `_refine_with_pyin`, which -- like pYIN
    # itself -- takes an in-memory audio array rather than a path.
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = np.zeros_like(t)
    for harmonic, amplitude in enumerate([1.0, 0.5, 0.3, 0.15, 0.08], start=1):
        tone += amplitude * np.sin(2 * np.pi * freq_hz * harmonic * t)
    return (0.3 * tone / np.max(np.abs(tone))).astype(np.float32)


def test_pitch_class_distance_is_octave_agnostic():
    assert _pitch_class_distance(60, 60) == 0
    assert _pitch_class_distance(60, 72) == 0  # exact octave
    assert _pitch_class_distance(60, 61) == 1
    assert _pitch_class_distance(60, 65) == 5
    assert _pitch_class_distance(60, 71) == 1  # wraps the other way around the octave


def test_pyin_window_separates_voiced_from_unvoiced_and_counts_total_frames():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    midi = np.array([60.0, np.nan, 60.5, np.nan, 61.0])
    voiced, total = _pyin_window(times, midi, 0.0, 0.4)
    assert total == 4  # frames at t=0.0, 0.1, 0.2, 0.3 fall in [0.0, 0.4)
    assert list(voiced) == [60.0, 60.5]


def test_refine_with_pyin_corrects_a_wrong_octave_and_extends_a_cut_short_offset(tmp_path):
    samplerate = 22050
    # A real, continuously-sung note at MIDI 60 (C4, ~261.6Hz) starting at the very top of the clip
    # -- but the note event basic-pitch supposedly reported is an octave too high (72) and cut off
    # early, the exact two failure modes measured on the real test song (see `_refine_with_pyin`'s
    # docstring). The clip's duration matches exactly as far as a full extension can reach (offset
    # + `_OFFSET_EXTEND_MAX_SECONDS`), and the note starts at 0.0 matching the audio, so there's no
    # leftover real-singing gap outside the note for `_repair_melody_gaps` to independently catch --
    # that behavior has its own dedicated tests above.
    audio = _harmonic_tone(261.63, duration_s=1.5, samplerate=samplerate)
    notes = [
        {
            "pitch_midi": 72,
            "pitch_hz": 523.25,
            "onset": 0.0,
            "offset": 1.0,
            "duration": 1.0,
            "velocity": 0.8,
        }
    ]

    refined = _refine_with_pyin(notes, audio, samplerate)

    assert len(refined) == 1
    assert refined[0]["pitch_midi"] == 60
    assert refined[0]["pitch_hz"] == pytest.approx(261.63, abs=1.0)
    # Extended well past the original (wrong) 1.0s offset, since the tone keeps sounding.
    assert refined[0]["offset"] > 1.3
    assert refined[0]["duration"] == pytest.approx(refined[0]["offset"] - 0.0)


def test_refine_with_pyin_keeps_a_high_note_above_the_old_frequency_ceiling(tmp_path):
    # `_MAXIMUM_FREQUENCY_HZ` (and therefore `_PYIN_FMAX_HZ`) used to be 1300Hz (~E6) -- below this
    # note's ~1568Hz (G6, MIDI 91), a real and not-unusual belted pop high note. With the old
    # ceiling, pyin's search couldn't represent the true fundamental at all and would lock onto an
    # in-range subharmonic (typically exactly an octave down, MIDI 79), which `_refine_with_pyin`
    # would then trust as a "confirmed" correction -- silently dropping a correctly-detected high
    # note an octave. With the ceiling raised past this pitch, pyin should confirm the note at its
    # real pitch instead of dragging it down.
    samplerate = 22050
    audio = _harmonic_tone(1567.98, duration_s=1.5, samplerate=samplerate)
    notes = [
        {
            "pitch_midi": 91,
            "pitch_hz": 1567.98,
            "onset": 0.0,
            "offset": 1.0,
            "duration": 1.0,
            "velocity": 0.8,
        }
    ]

    refined = _refine_with_pyin(notes, audio, samplerate)

    assert len(refined) == 1
    assert refined[0]["pitch_midi"] == 91


def test_refine_with_pyin_tolerates_a_brief_gap_mid_extension(tmp_path):
    # A real held note with one short, quiet dropout in the middle (a consonant, a breath) --
    # exactly the pattern measured on the real test song where a single noisy pYIN frame right
    # after the reported offset used to kill the whole extension immediately. The singing clearly
    # resumes moments later, so the extension should bridge the brief gap rather than stopping dead.
    samplerate = 22050
    segment = _harmonic_tone(261.63, duration_s=0.6, samplerate=samplerate)
    gap = np.zeros(int(0.05 * samplerate), dtype=np.float32)
    audio = np.concatenate([segment, gap, segment])
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.1, "offset": 0.5, "duration": 0.4, "velocity": 0.8},
    ]

    refined = _refine_with_pyin(notes, audio, samplerate)

    # The note continues (across the brief gap) well past its original 0.5s offset.
    assert refined[0]["offset"] > 0.5 + 0.05 + 0.1


def test_refine_with_pyin_does_not_extend_past_the_next_notes_onset(tmp_path):
    samplerate = 22050
    audio = _harmonic_tone(261.63, duration_s=2.0, samplerate=samplerate)
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.2, "offset": 1.0, "duration": 0.8, "velocity": 0.8},
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 1.1, "offset": 1.5, "duration": 0.4, "velocity": 0.8},
    ]

    refined = _refine_with_pyin(notes, audio, samplerate)

    assert refined[0]["offset"] <= 1.1
    assert refined[0]["offset"] <= refined[1]["onset"]


def test_drop_unconfirmed_pitch_spikes_drops_a_short_unvoiced_big_jump_note():
    # Mirrors the one real case measured on `test-song`: a short note, a big jump from its
    # neighbor, and pYIN finding nothing voiced in its window at all.
    notes = [
        _make_note(56, 0.0, 1.0),
        _make_note(71, 1.0, 1.15),
        _make_note(55, 1.15, 2.0),
    ]
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8])
    midi = np.array([56.0, 56.0, 56.0, 56.0, 56.0, np.nan, np.nan, np.nan, 55.0, 55.0, 55.0, 55.0])

    result = _drop_unconfirmed_pitch_spikes(notes, times, midi)

    assert [n["pitch_midi"] for n in result] == [56, 55]


def test_drop_unconfirmed_pitch_spikes_keeps_a_small_jump_even_if_unvoiced():
    # Same lack of pYIN confirmation, but the jump from its neighbors is small -- real singing
    # pYIN just failed to track (breathy delivery, ornamentation), not the octave-confusion
    # signature; measured on a real Korean-language test song, most low-voiced-fraction notes
    # looked like this, not like the artifact case above.
    notes = [
        _make_note(56, 0.0, 1.0),
        _make_note(59, 1.0, 1.15),
        _make_note(56, 1.15, 2.0),
    ]
    times = np.array([0.0, 0.5, 1.0, 1.05, 1.1, 1.5])
    midi = np.array([56.0, 56.0, np.nan, np.nan, np.nan, 56.0])

    result = _drop_unconfirmed_pitch_spikes(notes, times, midi)

    assert [n["pitch_midi"] for n in result] == [56, 59, 56]


def test_drop_unconfirmed_pitch_spikes_keeps_a_long_note_even_if_unvoiced_and_a_big_jump():
    # Same jump and lack of confirmation, but the note itself is long -- outside the measured
    # artifact signature (every real case was well under the duration cap), so leave it alone
    # rather than risk dropping a real long note pYIN happened to lose track of.
    notes = [
        _make_note(56, 0.0, 1.0),
        _make_note(71, 1.0, 2.0),
        _make_note(55, 2.0, 3.0),
    ]
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    midi = np.array([56.0, 56.0, np.nan, np.nan, 55.0, 55.0])

    result = _drop_unconfirmed_pitch_spikes(notes, times, midi)

    assert [n["pitch_midi"] for n in result] == [56, 71, 55]


def test_drop_unconfirmed_pitch_spikes_keeps_a_well_confirmed_short_big_jump_note():
    # Same short duration and big jump, but pYIN independently confirms the same pitch throughout
    # -- a real fast melodic movement (melisma/grace note), not an artifact. Measured on real songs:
    # the vast majority of short, big-jump notes look like this, which is exactly why voiced
    # fraction has to gate the jump/duration check rather than the reverse.
    notes = [
        _make_note(56, 0.0, 1.0),
        _make_note(71, 1.0, 1.15),
        _make_note(55, 1.15, 2.0),
    ]
    times = np.array([0.0, 0.5, 1.0, 1.05, 1.1, 1.5])
    midi = np.array([56.0, 56.0, 71.0, 71.0, 71.0, 55.0])

    result = _drop_unconfirmed_pitch_spikes(notes, times, midi)

    assert [n["pitch_midi"] for n in result] == [56, 71, 55]


def test_should_correct_note_pitch_trusts_a_well_confirmed_correction():
    assert _should_correct_note_pitch(voiced_fraction=0.5, disagreement_semitones=1) is True


def test_should_correct_note_pitch_trusts_a_sparse_but_unanimous_big_disagreement():
    # basic-pitch's inference isn't perfectly deterministic run-to-run (see the constant comment
    # above `_PYIN_SPARSE_MIN_VOICED_FRACTION`): a note can end up with too few voiced pYIN frames
    # to clear the main 0.3 bar, yet the handful that are voiced unanimously disagree by a full
    # register -- strong evidence of a wrong note even with sparse data. Measured concretely on
    # `bruno-mars-count-on-me-lyrics`: reprocessing the identical vocal stem twice with no code
    # change reported MIDI 81 in one run and 53 in the other for the same moment.
    assert _should_correct_note_pitch(voiced_fraction=0.15, disagreement_semitones=7) is True


def test_should_correct_note_pitch_ignores_a_sparse_small_disagreement():
    # Same sparse voicing, but the disagreement is small -- a couple of noisy near-miss frames, not
    # unanimous evidence of a wrong note, so basic-pitch's original pitch is left alone.
    assert _should_correct_note_pitch(voiced_fraction=0.15, disagreement_semitones=2) is False


def test_should_correct_note_pitch_ignores_a_big_disagreement_with_too_little_voicing():
    # Even a huge disagreement isn't trusted below the sparse floor -- one stray voiced frame isn't
    # enough evidence either way.
    assert _should_correct_note_pitch(voiced_fraction=0.05, disagreement_semitones=7) is False


def test_repair_melody_gaps_fills_a_real_singing_gap_basic_pitch_missed():
    # Mirrors the real case measured on `bruno-mars-count-on-me-lyrics`: a real, sustained,
    # differently-pitched stretch of singing sitting in the gap between two detected notes, with no
    # basic-pitch note there at all.
    samplerate = 22050
    tone_a = _harmonic_tone(261.63, duration_s=0.5, samplerate=samplerate)  # C4 (MIDI 60)
    tone_gap = _harmonic_tone(329.63, duration_s=0.5, samplerate=samplerate)  # E4 (MIDI 64)
    tone_b = _harmonic_tone(392.0, duration_s=0.5, samplerate=samplerate)  # G4 (MIDI 67)
    audio = np.concatenate([tone_a, tone_gap, tone_b])
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.0, "offset": 0.5, "duration": 0.5, "velocity": 0.8},
        {"pitch_midi": 67, "pitch_hz": 392.0, "onset": 1.0, "offset": 1.5, "duration": 0.5, "velocity": 0.8},
    ]

    times, midi = _pyin_track(audio, samplerate)
    repaired = _repair_melody_gaps(notes, times, midi, audio, samplerate)

    assert len(repaired) == 3
    assert repaired[0]["pitch_midi"] == 60
    assert repaired[1]["pitch_midi"] == 64
    assert repaired[1]["onset"] == pytest.approx(0.5, abs=0.05)
    assert repaired[1]["offset"] == pytest.approx(1.0, abs=0.05)
    assert repaired[2]["pitch_midi"] == 67


def test_repair_melody_gaps_leaves_a_real_silent_gap_alone():
    samplerate = 22050
    tone_a = _harmonic_tone(261.63, duration_s=0.5, samplerate=samplerate)
    silence = np.zeros(int(0.5 * samplerate), dtype=np.float32)
    tone_b = _harmonic_tone(392.0, duration_s=0.5, samplerate=samplerate)
    audio = np.concatenate([tone_a, silence, tone_b])
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.0, "offset": 0.5, "duration": 0.5, "velocity": 0.8},
        {"pitch_midi": 67, "pitch_hz": 392.0, "onset": 1.0, "offset": 1.5, "duration": 0.5, "velocity": 0.8},
    ]

    times, midi = _pyin_track(audio, samplerate)
    repaired = _repair_melody_gaps(notes, times, midi, audio, samplerate)

    assert [n["pitch_midi"] for n in repaired] == [60, 67]


def test_repair_melody_gaps_ignores_a_too_short_blip():
    # A blip well under `_GAP_REPAIR_MIN_RUN_SECONDS` (150ms) shouldn't be promoted to a real note --
    # same "too short to trust" reasoning as `_SPURIOUS_MAX_DURATION_S` elsewhere in this module.
    samplerate = 22050
    tone_a = _harmonic_tone(261.63, duration_s=0.5, samplerate=samplerate)
    blip = _harmonic_tone(329.63, duration_s=0.05, samplerate=samplerate)
    silence = np.zeros(int(0.45 * samplerate), dtype=np.float32)
    tone_b = _harmonic_tone(392.0, duration_s=0.5, samplerate=samplerate)
    audio = np.concatenate([tone_a, blip, silence, tone_b])
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.0, "offset": 0.5, "duration": 0.5, "velocity": 0.8},
        {"pitch_midi": 67, "pitch_hz": 392.0, "onset": 1.0, "offset": 1.5, "duration": 0.5, "velocity": 0.8},
    ]

    times, midi = _pyin_track(audio, samplerate)
    repaired = _repair_melody_gaps(notes, times, midi, audio, samplerate)

    assert [n["pitch_midi"] for n in repaired] == [60, 67]


def test_repair_melody_gaps_fills_a_gap_before_the_first_note():
    samplerate = 22050
    lead_in = _harmonic_tone(220.0, duration_s=0.5, samplerate=samplerate)  # A3 (MIDI 57)
    silence = np.zeros(int(0.3 * samplerate), dtype=np.float32)
    tone_a = _harmonic_tone(261.63, duration_s=0.5, samplerate=samplerate)
    audio = np.concatenate([lead_in, silence, tone_a])
    # tone_a's real audio starts at 0.8s (0.5 lead-in + 0.3 silence) -- the note list must agree
    # with where the audio actually is, or the "gap" this test means to exercise (the silence at
    # 0.5-0.8s) gets contaminated with real tone_a audio the note list wrongly excludes.
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.8, "offset": 1.3, "duration": 0.5, "velocity": 0.8},
    ]

    times, midi = _pyin_track(audio, samplerate)
    repaired = _repair_melody_gaps(notes, times, midi, audio, samplerate)

    assert len(repaired) == 2
    assert repaired[0]["pitch_midi"] == 57
    assert repaired[1]["pitch_midi"] == 60


def test_refine_with_pyin_drops_a_note_with_no_independent_confirmation(tmp_path):
    samplerate = 22050
    # Two real, continuously-sung notes with a short near-silent gap between them -- but basic-pitch
    # (wrongly) reported a third, short, far-off-pitch note sitting in that gap, the same failure
    # mode measured on the real test song. The gap is wide enough (0.35s) that the bogus note's own
    # window (0.7-0.8s) sits well clear of pYIN's frame-smearing at each real tone's edge (confirmed
    # directly: pYIN still reports voiced content up to ~0.05s past a tone's actual boundary).
    first = _harmonic_tone(261.63, duration_s=0.6, samplerate=samplerate)  # C4 (MIDI 60)
    silence = np.zeros(int(0.35 * samplerate), dtype=np.float32)
    second = _harmonic_tone(246.94, duration_s=0.6, samplerate=samplerate)  # B3 (MIDI 59)
    audio = np.concatenate([first, silence, second])
    notes = [
        {"pitch_midi": 60, "pitch_hz": 261.63, "onset": 0.1, "offset": 0.55, "duration": 0.45, "velocity": 0.8},
        {"pitch_midi": 80, "pitch_hz": 987.77, "onset": 0.7, "offset": 0.8, "duration": 0.1, "velocity": 0.4},
        {"pitch_midi": 59, "pitch_hz": 246.94, "onset": 0.95, "offset": 1.35, "duration": 0.4, "velocity": 0.8},
    ]

    refined = _refine_with_pyin(notes, audio, samplerate)

    assert [n["pitch_midi"] for n in refined] == [60, 59]


def test_collapse_octave_blips_handles_a_blip_ending_the_note_list():
    # The blip's trailing neighbor is also the very last note in the song -- exercises the
    # end-of-list bookkeeping so the final note isn't dropped or double-counted (a prior unrelated
    # note is included so the blip isn't at the very start of the list either).
    notes = [
        _make_note(58, -1.0, 0.0),
        _make_note(60, 0.0, 1.0),
        _make_note(72, 1.0, 1.2),
        _make_note(60, 1.2, 2.0),
    ]
    result = _collapse_octave_blips(notes)
    assert [n["pitch_midi"] for n in result] == [58, 60]
    assert result[-1]["offset"] == 2.0
