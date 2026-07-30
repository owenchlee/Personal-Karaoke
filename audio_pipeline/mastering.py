"""Auto-balance and clean-up mastering pass for a player's recorded take.

useRecording.ts (frontend) records the instrumental and the live mic as two
separate tracks instead of one pre-mixed stream (see
docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md). This
module takes those two tracks and produces a single mastered wav: aligned to
the same start time, the vocal denoised/compressed/highpassed, both tracks
loudness-balanced against each other (vocal louder), then mixed and limited
so the result can't clip.

master_recording() (added in a later task) is the single entry point
scripts/server.py calls; the rest of this module is its implementation,
ordered in the same sequence the audio actually flows through: align ->
clean the vocal -> loudness-normalize each track -> mix -> limit.
"""
from pathlib import Path

import soundfile as sf

# Positive = the vocal (mic) track lags the instrumental in a raw two-track recording.
# The mic capture path (getUserMedia -> hardware -> driver buffering) has real warm-up
# latency the instrumental's already-decoded <audio> element source doesn't, so the
# singer's actual voice lands later in a raw recording than when they actually sang it --
# see the "Recording start offset" section of
# docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md.
#
# 0.0 (this untuned default) is a no-op. Do NOT replace this with a guessed "typical mic
# latency" number -- measure the real value with scripts/measure_recording_offset.py
# against a real test recording (a sharp clap right on the song's first strong
# instrumental beat) before relying on this correction. See that script's own docstring
# for the exact procedure.
_RECORDING_OFFSET_SECONDS = 0.0


def _correct_start_offset(
    vocal_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    offset_seconds: float = _RECORDING_OFFSET_SECONDS,
) -> tuple[Path, Path]:
    """Trim ``offset_seconds`` off whichever track leads, so both start at the
    same real moment. Positive ``offset_seconds`` means the vocal lags (the
    common case, see the module docstring) -- trims that many seconds off the
    *front* of the vocal track. Negative means the vocal leads -- trims off
    the instrumental's front instead. Zero is a no-op: both tracks are copied
    through unchanged.
    """
    vocal_data, vocal_sr = sf.read(vocal_path)
    instrumental_data, instrumental_sr = sf.read(instrumental_path)

    if offset_seconds > 0:
        trim_samples = int(round(offset_seconds * vocal_sr))
        vocal_data = vocal_data[trim_samples:]
    elif offset_seconds < 0:
        trim_samples = int(round(-offset_seconds * instrumental_sr))
        instrumental_data = instrumental_data[trim_samples:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_vocal_path = output_dir / "vocal_aligned.wav"
    aligned_instrumental_path = output_dir / "instrumental_aligned.wav"
    sf.write(aligned_vocal_path, vocal_data, vocal_sr)
    sf.write(aligned_instrumental_path, instrumental_data, instrumental_sr)
    return aligned_vocal_path, aligned_instrumental_path
