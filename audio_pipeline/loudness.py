"""Two-pass loudness normalization via ffmpeg's `loudnorm` filter.

Shared by mastering.py (balances a player's recorded take before mixdown)
and pipeline.py (brings a freshly separated instrumental up to a normal
playback level). A Demucs/BS-RoFormer stem has no consistent loudness
target -- it's just whatever amplitude the separation model happened to
output, which can be far quieter than a commercially mastered track (see
NOTES.md's "Instrumental playback is too quiet" entry).
"""
import json
import os
import re
from pathlib import Path

import ffmpeg

_LOUDNORM_TP = -1.5
_LOUDNORM_LRA = 11


def measure_loudness(input_path: str | Path, target_i: float) -> dict:
    """First pass of ffmpeg's two-pass loudnorm: measures the real loudness
    stats ffmpeg needs to normalize accurately on the second pass (see
    ``normalize_loudness``), instead of relying on loudnorm's single-pass
    mode, which is a rougher, real-time-only estimate. Output audio is
    discarded (``-f null``) -- only the stats loudnorm prints to stderr as
    JSON matter.
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


def normalize_loudness(input_path: str | Path, output_path: str | Path, target_i: float) -> Path:
    """Second pass: normalizes ``input_path`` to ``target_i`` LUFS using the
    stats ``measure_loudness`` already measured, writing the result to
    ``output_path``.
    """
    stats = measure_loudness(input_path, target_i)
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
