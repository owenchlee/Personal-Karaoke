"""Tests for the end-to-end video -> cached assets orchestration.

Uses a synthetic clip (not a real song) for the full-pipeline smoke test --
this only proves the stages wire together correctly, not that separation or
melody extraction are musically accurate (covered by their own module tests
and by manual real-song validation, see NOTES.md).
"""
import json
from unittest.mock import patch

import ffmpeg

from audio_pipeline.pipeline import process_song


def _write_synthetic_video(path, duration_s=2.0):
    video = ffmpeg.input(f"testsrc=duration={duration_s}:size=320x240:rate=15", f="lavfi")
    audio = ffmpeg.input(f"sine=frequency=220:duration={duration_s}", f="lavfi")
    (
        ffmpeg.output(video, audio, str(path), vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def test_process_song_runs_end_to_end_and_produces_assets(tmp_path):
    video_path = tmp_path / "synthetic_song.mp4"
    cache_dir = tmp_path / "cache"
    _write_synthetic_video(video_path)

    assets = process_song(video_path, cache_dir=cache_dir)

    assert assets.instrumental_path.exists()
    assert assets.vocals_path.exists()
    assert assets.midi_path.exists()
    assert assets.notes_path.exists()

    notes = json.loads(assets.notes_path.read_text())
    assert isinstance(notes, list)

    meta_path = cache_dir / "synthetic-song" / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["song_id"] == "synthetic-song"


def test_process_song_skips_reprocessing_when_cached(tmp_path):
    cache_dir = tmp_path / "cache"
    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"fake wav data")
    (song_cache_dir / "notes.json").write_text("[]")

    fake_video_path = tmp_path / "video.mp4"
    fake_video_path.write_bytes(b"fake video data")

    with patch("audio_pipeline.pipeline.extract_audio") as mock_extract_audio:
        assets = process_song(fake_video_path, cache_dir=cache_dir, song_id="my-song")

    mock_extract_audio.assert_not_called()
    assert assets.instrumental_path == song_cache_dir / "instrumental.wav"
    assert assets.notes_path == song_cache_dir / "notes.json"
