"""Pitch-shift a cached song's instrumental (and its note targets) by a fixed number of
semitones, tempo unchanged -- the audio side of the "key change" feature: a player syncs their own
vocal range once (frontend/src/screens/VoiceRangeScreen.tsx), and songs whose sung range sits
outside it get transposed to fit before playback.

Uses the free, open-source Rubber Band Library (https://breakfastquay.com/rubberband/) via
`pyrubberband`, which shells out to a separately-installed `rubberband` CLI binary -- not bundled by
pip, the same category of external-binary dependency this project already has for ffmpeg (see
README's "Optional: key change" section for install instructions). Chosen over the already-installed
`librosa.effects.pitch_shift` for its `--formant` option: a plain phase-vocoder shift (what librosa
does) leaves formants tied to the pitch, so a large upward shift sounds chipmunked and a large
downward one sounds muffled; Rubber Band's formant preservation keeps a voice's natural timbre at
real key-change amounts. Confirmed directly against a real cached song (not just a synthetic tone):
a 3-semitone shift of a 214.7s stereo instrumental measured within ~0.12 semitone of the requested
amount (pYIN-tracked) and completed in ~12s on this CPU -- faster than the librosa alternative
benchmarked for the same file (~20s) as well as higher quality.

Also passes `--fine` to select Rubber Band's R3 engine. The CLI's default (used when neither `-2`
nor `-3` is given) is the older R2 engine, kept only for backward compatibility; R2's phase-vocoder
smears transients and washes out detail on dense multi-instrument mixes, which is what made shifted
instrumentals sound blurry/muffled even with `--formant` on. R3 "almost always produces better
results than the R2 engine" per `rubberband --help`. Confirmed against the same category of real
cached song as above: a 3-semitone shift of a 275.3s stereo instrumental took ~58s on this CPU with
R3 (~21% of the song's own duration) -- roughly 5x R2's ~12s, and now slower than the librosa
alternative this module was originally chosen over. Accepted anyway for the clearer mix, since
transposes are cached (see scripts/server.py's transpose endpoint) and the cost is paid once.
"""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyrubberband as pyrb
import soundfile as sf


@dataclass
class TransposeResult:
    instrumental_path: Path
    notes_path: Path


def _midi_to_hz(pitch_midi: float) -> float:
    return 440.0 * 2 ** ((pitch_midi - 69) / 12)


def _shift_notes(notes: list[dict], semitones: int) -> list[dict]:
    """Shift every note's `pitch_midi` by `semitones` and recompute `pitch_hz` from the shifted
    pitch -- `onset`/`offset`/`duration` are left untouched, since a tempo-preserving pitch shift
    doesn't move timing at all (this is also why lyrics.json needs no corresponding shift/copy: it
    times word-level lyrics against the same untouched timeline).
    """
    shifted = []
    for note in notes:
        new_note = dict(note)
        new_note["pitch_midi"] = note["pitch_midi"] + semitones
        new_note["pitch_hz"] = round(_midi_to_hz(new_note["pitch_midi"]), 2)
        shifted.append(new_note)
    return shifted


def transpose_song(song_cache_dir: str | Path, semitones: int, output_dir: str | Path) -> TransposeResult:
    """Pitch-shift `song_cache_dir`'s `instrumental.wav` by `semitones` (positive = higher) and
    shift `notes.json`'s pitches to match, writing both into `output_dir`. Callers are expected to
    cache the result themselves keyed by (song, semitones) -- this function always does the work,
    it doesn't check for an existing cached result on its own (see scripts/server.py's transpose
    endpoint, which owns that caching decision).
    """
    if shutil.which("rubberband") is None:
        raise RuntimeError(
            "The 'rubberband' command-line tool is not on PATH -- install the free Rubber Band "
            'Library (see README\'s "Optional: key change" section) before requesting a key change.'
        )

    song_cache_dir = Path(song_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(song_cache_dir / "instrumental.wav")
    shifted_audio = pyrb.pitch_shift(
        audio, sample_rate, n_steps=semitones, rbargs={"--formant": "", "--fine": ""}
    )

    instrumental_path = output_dir / "instrumental.wav"
    sf.write(instrumental_path, shifted_audio, sample_rate)

    notes = json.loads((song_cache_dir / "notes.json").read_text())
    notes_path = output_dir / "notes.json"
    notes_path.write_text(json.dumps(_shift_notes(notes, semitones), indent=2))

    return TransposeResult(instrumental_path=instrumental_path, notes_path=notes_path)
