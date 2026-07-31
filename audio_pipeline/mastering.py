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
import json
import os
import re
from pathlib import Path

import ffmpeg
import soundfile as sf

from audio_pipeline.transcode import transcode_to_wav

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


_HIGHPASS_HZ = 90


def _clean_vocal(input_path: Path, output_dir: Path) -> Path:
    """Highpass (removes rumble/handling noise below ``_HIGHPASS_HZ``), denoise
    (ffmpeg's built-in ``afftdn`` -- no extra model download, see this
    feature's design spec for why not ``arnndn``), and compress
    (``acompressor``, evens out quiet/loud parts) the vocal track. Loudness
    balancing against the instrumental happens separately (see
    ``_apply_loudnorm``, added in a later task).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vocal_cleaned.wav"

    try:
        (
            ffmpeg.input(str(input_path))
            .filter("highpass", f=_HIGHPASS_HZ)
            .filter("afftdn")
            .filter("acompressor")
            .output(str(output_path))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to clean vocal {input_path}: {stderr}") from exc

    return output_path


_LOUDNORM_TP = -1.5
_LOUDNORM_LRA = 11


def _measure_loudness(input_path: Path, target_i: float) -> dict:
    """First pass of ffmpeg's two-pass loudnorm: measures the real loudness
    stats ffmpeg needs to normalize accurately on the second pass (see
    ``_apply_loudnorm``), instead of relying on loudnorm's single-pass mode,
    which is a rougher, real-time-only estimate. Output audio is discarded
    (``-f null``) -- only the stats loudnorm prints to stderr as JSON matter.
    """
    try:
        _, stderr = (
            ffmpeg.input(str(input_path))
            .filter("loudnorm", i=target_i, tp=_LOUDNORM_TP, lra=_LOUDNORM_LRA, print_format="json")
            .output(os.devnull, format="null")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to measure loudness of {input_path}: {stderr_text}") from exc

    stderr_text = stderr.decode(errors="replace")
    match = re.search(r"\{[^{}]*\}", stderr_text, re.DOTALL)
    if not match:
        raise RuntimeError(f"ffmpeg loudnorm did not report measured stats for {input_path}")
    return json.loads(match.group(0))


def _apply_loudnorm(input_path: Path, output_path: Path, target_i: float) -> Path:
    """Second pass: normalizes ``input_path`` to ``target_i`` LUFS using the
    stats ``_measure_loudness`` already measured, writing the result to
    ``output_path``.
    """
    stats = _measure_loudness(input_path, target_i)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg.input(str(input_path))
            .filter(
                "loudnorm",
                i=target_i,
                tp=_LOUDNORM_TP,
                lra=_LOUDNORM_LRA,
                measured_I=stats["input_i"],
                measured_TP=stats["input_tp"],
                measured_LRA=stats["input_lra"],
                measured_thresh=stats["input_thresh"],
                offset=stats["target_offset"],
                linear="true",
                print_format="summary",
            )
            .output(str(output_path), ar=44100)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to normalize loudness of {input_path}: {stderr_text}") from exc

    return output_path


_VOCAL_TARGET_LUFS = -16
_INSTRUMENTAL_TARGET_LUFS = -20
_LIMITER_CEILING = 0.95


def _mix_and_limit(vocal_path: Path, instrumental_path: Path, output_path: Path) -> Path:
    """Mixes the two loudness-normalized tracks and applies a final limiter so
    the vocal's loudness boost can't clip the combined output. ``normalize=0``
    on ``amix`` keeps the loudnorm targets intact -- amix's own default
    (``normalize=1``) would auto-scale both inputs down by input count,
    undoing the deliberate vocal-vs-instrumental balance.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocal_stream = ffmpeg.input(str(vocal_path))
    instrumental_stream = ffmpeg.input(str(instrumental_path))

    try:
        (
            ffmpeg.filter([vocal_stream, instrumental_stream], "amix", inputs=2, duration="longest", normalize=0)
            .filter("alimiter", limit=_LIMITER_CEILING)
            .output(str(output_path))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to mix vocal and instrumental into {output_path}: {stderr_text}") from exc

    return output_path


def master_recording(vocal_path: str | Path, instrumental_path: str | Path, output_dir: str | Path) -> Path:
    """Auto-balance and clean up a player's recorded take -- the single entry
    point scripts/server.py calls. ``vocal_path``/``instrumental_path`` are
    the two separately-recorded tracks from useRecording.ts, in any
    container/codec ffmpeg can read (in practice, webm/opus). Returns the
    path to the mastered wav (start-aligned, vocal cleaned and balanced
    louder than the instrumental, mixed, limited against clipping).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocal_wav = transcode_to_wav(vocal_path, output_dir)
    instrumental_wav = transcode_to_wav(instrumental_path, output_dir)

    aligned_vocal_path, aligned_instrumental_path = _correct_start_offset(
        vocal_wav, instrumental_wav, output_dir
    )
    cleaned_vocal_path = _clean_vocal(aligned_vocal_path, output_dir)
    normalized_vocal_path = _apply_loudnorm(
        cleaned_vocal_path, output_dir / "vocal_normalized.wav", _VOCAL_TARGET_LUFS
    )
    normalized_instrumental_path = _apply_loudnorm(
        aligned_instrumental_path, output_dir / "instrumental_normalized.wav", _INSTRUMENTAL_TARGET_LUFS
    )

    mastered_path = output_dir / "mastered.wav"
    _mix_and_limit(normalized_vocal_path, normalized_instrumental_path, mastered_path)
    return mastered_path


def measure_start_offset(vocal_path: str | Path, instrumental_path: str | Path) -> float:
    """Measures the real start offset between a raw two-track recording's
    vocal and instrumental tracks, for tuning ``_RECORDING_OFFSET_SECONDS``
    (see that constant's docstring and scripts/measure_recording_offset.py,
    which wraps this for one-off manual calibration against a real
    recording).

    Expects a take where the singer claps once, sharply, right on the song's
    first strong instrumental hit -- the clap is the vocal track's first
    onset; that instrumental hit is the instrumental track's first onset.
    Returns ``vocal_onset - instrumental_onset`` (positive = the vocal lags,
    matching ``_RECORDING_OFFSET_SECONDS``'s sign convention).
    """
    import librosa  # local import: only this one-off calibration path needs it

    def first_onset_seconds(path: str | Path) -> float:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
        if len(onset_frames) == 0:
            raise ValueError(f"No onset detected in {path}")
        return float(librosa.frames_to_time(onset_frames[0], sr=sr))

    return first_onset_seconds(vocal_path) - first_onset_seconds(instrumental_path)
