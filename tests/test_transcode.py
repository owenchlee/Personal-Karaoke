"""Smoke test for the mp3 transcoding helper used by the recording feature.

Uses a synthetic sine-wave clip -- this only proves ffmpeg + libmp3lame
produce a valid, correctly-tagged mp3 file end-to-end, not that any specific
recording sounds correct (same caveat as the other pipeline smoke tests).
"""
import ffmpeg
import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.transcode import transcode_to_mp3


def _write_synthetic_clip(path, duration_s=2.0, samplerate=44100, freq_hz=220.0):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * freq_hz * t)
    stereo = np.stack([tone, tone], axis=1).astype(np.float32)
    sf.write(path, stereo, samplerate)


def test_transcode_to_mp3_produces_a_valid_mp3(tmp_path):
    input_duration_s = 2.0
    input_path = tmp_path / "synthetic_input.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_clip(input_path, duration_s=input_duration_s)

    mp3_path = transcode_to_mp3(input_path, output_dir)

    assert mp3_path.exists()
    assert mp3_path.suffix == ".mp3"
    assert mp3_path.stat().st_size > 0

    probe = ffmpeg.probe(str(mp3_path))
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert audio_streams, "no audio stream found in transcoded mp3"
    assert audio_streams[0]["codec_name"] == "mp3"
    assert float(probe["format"]["duration"]) == pytest.approx(input_duration_s, abs=0.5)


def test_transcode_to_mp3_raises_a_clear_error_on_a_bogus_input(tmp_path):
    bogus_path = tmp_path / "not_audio.webm"
    bogus_path.write_bytes(b"this is not a real media file")

    with pytest.raises(RuntimeError, match="ffmpeg failed to transcode"):
        transcode_to_mp3(bogus_path, tmp_path / "out")
