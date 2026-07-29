"""Tests for the end-to-end video -> cached assets orchestration.

Uses a synthetic clip (not a real song) for the full-pipeline smoke test --
this only proves the stages wire together correctly, not that separation or
melody extraction are musically accurate (covered by their own module tests
and by manual real-song validation, see NOTES.md).
"""
import json
from unittest.mock import patch

import ffmpeg

from audio_pipeline.pipeline import _remove_background_vocal_notes, is_cached, process_song


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
    assert assets.lyrics_path.exists()

    notes = json.loads(assets.notes_path.read_text())
    assert isinstance(notes, list)

    words = json.loads(assets.lyrics_path.read_text())
    assert isinstance(words, list)

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
    (song_cache_dir / "lyrics.json").write_text("[]")

    fake_video_path = tmp_path / "video.mp4"
    fake_video_path.write_bytes(b"fake video data")

    with patch("audio_pipeline.pipeline.extract_audio") as mock_extract_audio:
        assets = process_song(fake_video_path, cache_dir=cache_dir, song_id="my-song")

    mock_extract_audio.assert_not_called()
    assert assets.instrumental_path == song_cache_dir / "instrumental.wav"
    assert assets.notes_path == song_cache_dir / "notes.json"


def test_is_cached_true_once_all_assets_exist(tmp_path):
    cache_dir = tmp_path / "cache"
    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"fake wav data")
    (song_cache_dir / "notes.json").write_text("[]")
    (song_cache_dir / "lyrics.json").write_text("[]")

    assert is_cached(cache_dir, "my-song") is True


def test_is_cached_false_when_never_processed_or_incomplete(tmp_path):
    cache_dir = tmp_path / "cache"

    assert is_cached(cache_dir, "never-processed") is False

    partial_dir = cache_dir / "partial-song"
    partial_dir.mkdir(parents=True)
    (partial_dir / "instrumental.wav").write_bytes(b"fake wav data")

    assert is_cached(cache_dir, "partial-song") is False


def test_remove_background_vocal_notes_drops_overlapping_notes(tmp_path):
    notes_path = tmp_path / "notes.json"
    notes_path.write_text(json.dumps([
        {"pitch_midi": 60, "onset": 0.0, "offset": 1.0},  # before the range -- kept
        {"pitch_midi": 62, "onset": 5.0, "offset": 6.0},  # inside the range -- dropped
        {"pitch_midi": 64, "onset": 9.5, "offset": 10.5},  # straddles the range end -- dropped
        {"pitch_midi": 65, "onset": 11.0, "offset": 12.0},  # after the range -- kept
    ]))

    _remove_background_vocal_notes(notes_path, [(5.0, 10.0)])

    result = json.loads(notes_path.read_text())
    assert [note["pitch_midi"] for note in result] == [60, 65]


def test_process_song_reports_stage_progress_via_on_progress_callback(tmp_path):
    cache_dir = tmp_path / "cache"
    fake_video_path = tmp_path / "video.mp4"
    fake_video_path.write_bytes(b"fake video data")

    extracted_wav = tmp_path / "extracted.wav"
    extracted_wav.write_bytes(b"fake wav data")
    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"fake vocals")
    instrumental_path = tmp_path / "instrumental.wav"
    instrumental_path.write_bytes(b"fake instrumental")
    midi_path = tmp_path / "melody.mid"
    midi_path.write_bytes(b"fake midi")
    notes_path = tmp_path / "notes.json"
    notes_path.write_text("[]")
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text("[]")

    melody_result = type("Melody", (), {"midi_path": midi_path, "notes_path": notes_path})()
    lyrics_result = type(
        "Lyrics", (), {"lyrics_path": lyrics_path, "background_vocal_ranges": []}
    )()

    progress_seen = []

    with (
        patch("audio_pipeline.pipeline.extract_audio", return_value=extracted_wav),
        patch(
            "audio_pipeline.pipeline.separate_stems",
            return_value=(vocals_path, instrumental_path),
        ),
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result),
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result),
    ):
        process_song(
            fake_video_path, cache_dir=cache_dir, song_id="my-song",
            on_progress=lambda stage, fraction=0.0: progress_seen.append(stage),
        )

    assert progress_seen == ["separating", "extracting_melody", "transcribing_lyrics"]
