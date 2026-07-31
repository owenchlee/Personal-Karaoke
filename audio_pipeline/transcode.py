"""Generic audio transcoding via ffmpeg.

Used for two things: turning the webm/opus blob the browser's MediaRecorder
produces (for the "record your take" feature) into a real mp3 for download
(``transcode_to_mp3``), and transcoding a recorded take's tracks to a fixed-
sample-rate wav (``transcode_to_wav``) for audio_pipeline/mastering.py's
sample-accurate processing. Kept separate from video_extraction.py, whose job
is specifically video-container -> wav for the pipeline's own input step, not
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


def transcode_to_wav(input_path: str | Path, output_dir: str | Path) -> Path:
    """Transcode ``input_path`` (any container/codec ffmpeg can read, e.g. the
    webm/opus a browser's MediaRecorder produces) to a 44.1kHz wav file inside
    ``output_dir`` -- for downstream processing (audio_pipeline/mastering.py)
    that needs sample-accurate access via soundfile, which can't read
    webm/opus directly. Forces a fixed sample rate so two independently-
    transcoded tracks (e.g. a recording's separate vocal/instrumental takes)
    always line up sample-for-sample.

    Returns the path to the saved wav file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / f"{input_path.stem}.wav"

    try:
        (
            ffmpeg.input(str(input_path))
            .output(str(wav_path), ar=44100)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to transcode {input_path} to wav: {stderr}") from exc

    return wav_path
