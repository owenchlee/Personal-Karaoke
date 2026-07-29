"""Vocal/instrumental separation via Demucs.

Given any audio file Demucs can read, produce both an isolated vocal stem
(input to melody extraction) and a reconstructed instrumental track (played
back during the game). No pitch extraction here -- that's melody_extraction.py.

Runs on GPU when one is available (see audio_pipeline.device), CPU otherwise.
"""
import os
from pathlib import Path
from typing import Callable

from demucs import api as demucs_api

from audio_pipeline.device import get_device

_MODEL_NAME = "htdemucs"
_VOCALS_STEM = "vocals"

# Demucs splits the track into segments and, by default, processes them one
# at a time even though the CPU has cores to spare (jobs=0). Measured on a
# real 274s song on a 12-core machine: jobs=0 took 113.6s, jobs=4 took 78.8s
# (~30% faster) -- capped at 4 rather than using every core since each
# parallel job holds its own copy of intermediate tensors in memory. This is a
# CPU-specific tuning knob: on GPU, Demucs processes segments on-device and
# this setting has no meaningful effect.
_DEFAULT_JOBS = min(4, os.cpu_count() or 1)


def _progress_callback(on_progress: Callable[[float], None]) -> Callable[[dict], None]:
    def callback(info: dict) -> None:
        if info.get("state") != "end":
            return
        models = info.get("models") or 1
        model_idx = info.get("model_idx_in_bag") or 0
        audio_length = info.get("audio_length") or 1
        within_model = min(info.get("segment_offset", 0) / audio_length, 1.0)
        on_progress(min(max((model_idx + within_model) / models, 0.0), 1.0))

    return callback


def separate_stems(
    input_path: str | Path,
    output_dir: str | Path,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, Path]:
    """Run Demucs on ``input_path`` and save both the isolated vocal stem and
    a reconstructed instrumental track (the original mix minus vocals) as wav
    files inside ``output_dir``.

    ``on_progress``, if given, is called repeatedly with a 0-1 fraction as
    Demucs finishes each internal segment -- lets a caller (e.g. the job
    server) show real separation progress instead of a stage that just sits
    there for the ~80s+ this normally takes.

    Returns ``(vocals_path, instrumental_path)``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = demucs_api.Separator(
        model=_MODEL_NAME,
        device=get_device(),
        jobs=_DEFAULT_JOBS,
        callback=_progress_callback(on_progress) if on_progress else None,
    )
    original_wav, stems = separator.separate_audio_file(input_path)
    # Demucs's own callback reports each segment's *start* offset, not how
    # much is done -- for a short clip processed as a single segment that
    # never gets above 0.0 mid-flight. separate_audio_file() having returned
    # is the actual ground truth that this stage is complete, regardless of
    # what the last mid-flight estimate said.
    if on_progress:
        on_progress(1.0)

    if _VOCALS_STEM not in stems:
        raise RuntimeError(
            f"Model '{_MODEL_NAME}' did not produce a '{_VOCALS_STEM}' stem; "
            f"got stems: {sorted(stems)}"
        )

    vocals = stems[_VOCALS_STEM]
    instrumental = original_wav - vocals

    vocals_path = output_dir / f"{input_path.stem}_vocals.wav"
    instrumental_path = output_dir / f"{input_path.stem}_instrumental.wav"
    demucs_api.save_audio(vocals, vocals_path, samplerate=separator.samplerate)
    demucs_api.save_audio(instrumental, instrumental_path, samplerate=separator.samplerate)
    return vocals_path, instrumental_path
