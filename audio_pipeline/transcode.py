"""Generic audio transcoding via ffmpeg.

Currently used for one thing: turning the webm/opus blob the browser's
MediaRecorder produces (for the "record your take" feature) into a real mp3
for download. Kept separate from video_extraction.py, whose job is
specifically video-container -> wav for the pipeline's own input step, not
arbitrary format conversion.
"""
from pathlib import Path

import ffmpeg


def transcode_to_mp3(input_path: str | Path, output_dir: str | Path) -> Path:
    """Transcode ``input_path`` to an mp3 file inside ``output_dir``.

    Returns the path to the saved mp3 file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_path = output_dir / f"{input_path.stem}.mp3"

    try:
        (
            ffmpeg.input(str(input_path))
            .output(str(mp3_path), acodec="libmp3lame", audio_bitrate="192k")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to transcode {input_path} to mp3: {stderr}") from exc

    return mp3_path
