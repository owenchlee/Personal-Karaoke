"""Video -> audio extraction via ffmpeg.

First real deliverable in the video-to-notes pipeline: pull the audio track out
of a video file (or pass through an audio file unchanged) so the rest of the
pipeline (separation, melody extraction) always operates on a wav.
"""
from pathlib import Path

import ffmpeg


def extract_audio(video_path: str | Path, output_dir: str | Path) -> Path:
    """Extract the audio track from ``video_path`` and save it as a wav file
    inside ``output_dir``.

    Returns the path to the saved wav file.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / f"{video_path.stem}.wav"

    try:
        (
            ffmpeg.input(str(video_path))
            .output(str(wav_path), ac=2, ar=44100, vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}: {stderr}") from exc

    return wav_path
