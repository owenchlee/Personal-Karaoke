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

_SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft", "bs_roformer")
_VOCALS_STEM = "vocals"

# Confirmed current/downloadable directly from the installed audio-separator package's own
# load_model() default (not just the README) -- see NOTES.md's "Separation model choice:
# BS-RoFormer validation" entry for the full validation writeup this filename and the output
# naming pattern below come from.
_BS_ROFORMER_MODEL_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

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


def _separate_bs_roformer(
    input_path: Path,
    output_dir: Path,
    on_progress: Callable[[float], None] | None,
) -> tuple[Path, Path]:
    """BS-RoFormer via the `audio-separator` package -- higher separation
    quality than either Demucs model, at a real cost: ~3.5x the input audio's
    own duration on this CPU (vs. Demucs's near-realtime), confirmed in
    NOTES.md's validation entry. No progress hook exists in the confirmed
    API, so ``on_progress`` (if given) only fires once, at completion.
    """
    from audio_separator.separator import Separator

    separator = Separator(output_dir=str(output_dir))
    separator.load_model(model_filename=_BS_ROFORMER_MODEL_FILENAME)
    output_files = separator.separate(str(input_path))

    if on_progress:
        on_progress(1.0)

    vocals_path = next((f for f in output_files if "(Vocals)" in Path(f).name), None)
    instrumental_path = next(
        (f for f in output_files if "(Instrumental)" in Path(f).name), None
    )
    if vocals_path is None or instrumental_path is None:
        raise RuntimeError(
            f"Model 'bs_roformer' did not produce both a Vocals and Instrumental stem; "
            f"got output files: {output_files}"
        )

    # separate() returns bare filenames written inside output_dir, not absolute paths.
    vocals_path = Path(vocals_path)
    instrumental_path = Path(instrumental_path)
    if not vocals_path.is_absolute():
        vocals_path = output_dir / vocals_path
    if not instrumental_path.is_absolute():
        instrumental_path = output_dir / instrumental_path

    return vocals_path, instrumental_path


def separate_stems(
    input_path: str | Path,
    output_dir: str | Path,
    model: str = "htdemucs",
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, Path]:
    """Separate ``input_path`` into an isolated vocal stem and a reconstructed
    instrumental track (played back during the game), saved as wav files
    inside ``output_dir``.

    ``model`` selects the separation backend: "htdemucs" (default, fast,
    ~80s for a 274s song on this CPU -- see NOTES.md), "htdemucs_ft" (a
    fine-tuned bag of 4 Demucs models, noticeably less bleed, ~4x slower), or
    "bs_roformer" (higher quality still, via the `audio-separator` package,
    ~3.5x the input's own duration -- clearly slower than realtime, see
    NOTES.md's BS-RoFormer validation entry for measured timing and an RMS
    sanity check). Raises ``ValueError`` for anything else.

    ``on_progress``, if given, is called repeatedly with a 0-1 fraction as
    separation progresses -- lets a caller (e.g. the job server) show real
    separation progress instead of a stage that just sits there. For
    "bs_roformer" this only fires once, at completion (no progress hook
    exists in its confirmed API).

    Returns ``(vocals_path, instrumental_path)``.
    """
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported separation model {model!r}; expected one of {_SUPPORTED_MODELS}"
        )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model == "bs_roformer":
        return _separate_bs_roformer(input_path, output_dir, on_progress)

    separator = demucs_api.Separator(
        model=model,
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
            f"Model '{model}' did not produce a '{_VOCALS_STEM}' stem; "
            f"got stems: {sorted(stems)}"
        )

    vocals = stems[_VOCALS_STEM]
    instrumental = original_wav - vocals

    vocals_path = output_dir / f"{input_path.stem}_vocals.wav"
    instrumental_path = output_dir / f"{input_path.stem}_instrumental.wav"
    demucs_api.save_audio(vocals, vocals_path, samplerate=separator.samplerate)
    demucs_api.save_audio(instrumental, instrumental_path, samplerate=separator.samplerate)
    return vocals_path, instrumental_path
