"""Vocal/instrumental separation via Demucs, CPU-only.

Given any audio file Demucs can read, produce both an isolated vocal stem
(input to melody extraction) and a reconstructed instrumental track (played
back during the game). No pitch extraction here -- that's melody_extraction.py.
"""
from pathlib import Path

from demucs import api as demucs_api

_MODEL_NAME = "htdemucs"
_VOCALS_STEM = "vocals"


def separate_stems(input_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Run Demucs on ``input_path`` and save both the isolated vocal stem and
    a reconstructed instrumental track (the original mix minus vocals) as wav
    files inside ``output_dir``.

    Returns ``(vocals_path, instrumental_path)``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = demucs_api.Separator(model=_MODEL_NAME, device="cpu")
    original_wav, stems = separator.separate_audio_file(input_path)

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
