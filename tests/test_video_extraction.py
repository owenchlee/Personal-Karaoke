"""Smoke test for the ffmpeg video->audio extraction wrapper.

Generates a synthetic test video on the fly (ffmpeg's testsrc + sine filters)
rather than committing a real video file -- this only proves the extraction
mechanics work (a video container in, a readable wav out), not anything about
real-world video/audio codecs or vocal content.
"""
import ffmpeg
import numpy as np
import soundfile as sf

from audio_pipeline.video_extraction import extract_audio


def _write_synthetic_video(path, duration_s=2.0):
    video = ffmpeg.input(f"testsrc=duration={duration_s}:size=320x240:rate=15", f="lavfi")
    audio = ffmpeg.input(f"sine=frequency=220:duration={duration_s}", f="lavfi")
    (
        ffmpeg.output(video, audio, str(path), vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def test_extract_audio_produces_readable_wav(tmp_path):
    input_duration_s = 2.0
    video_path = tmp_path / "synthetic_input.mp4"
    output_dir = tmp_path / "out"
    _write_synthetic_video(video_path, duration_s=input_duration_s)

    wav_path = extract_audio(video_path, output_dir)

    assert wav_path.exists()
    assert wav_path.suffix == ".wav"

    data, samplerate = sf.read(wav_path)
    assert samplerate > 0
    assert data.shape[0] > 0
    assert not np.isnan(data).any()
    assert not np.isinf(data).any()
