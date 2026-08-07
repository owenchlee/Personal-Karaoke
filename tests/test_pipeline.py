"""Tests for the end-to-end video -> cached assets orchestration.

Uses a synthetic clip (not a real song) for the full-pipeline smoke test --
this only proves the stages wire together correctly, not that separation or
melody extraction are musically accurate (covered by their own module tests
and by manual real-song validation, see NOTES.md).
"""
import json
from pathlib import Path
from unittest.mock import patch

import ffmpeg

import pytest

from audio_pipeline.pipeline import (
    _remove_background_vocal_notes,
    _remove_notes_without_lyrics,
    is_cached,
    process_song,
    reprocess_melody_and_lyrics,
)


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


def test_remove_notes_without_lyrics_drops_notes_outside_lyric_coverage(tmp_path):
    # Single lyric line spanning 10.0-11.0s.
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps([
        {"word": "hello", "start": 10.0, "end": 10.4, "line": 0},
        {"word": "world", "start": 10.6, "end": 11.0, "line": 0},
    ]))

    notes_path = tmp_path / "notes.json"
    notes_path.write_text(json.dumps([
        {"pitch_midi": 60, "onset": 1.0, "offset": 2.0},  # far before the line -- dropped
        {"pitch_midi": 62, "onset": 10.2, "offset": 10.8},  # inside the line -- kept
        {"pitch_midi": 64, "onset": 11.3, "offset": 11.4},  # just within the 0.5s pad -- kept
        {"pitch_midi": 65, "onset": 12.0, "offset": 12.5},  # well past the pad -- dropped
    ]))

    _remove_notes_without_lyrics(notes_path, lyrics_path)

    result = json.loads(notes_path.read_text())
    assert [note["pitch_midi"] for note in result] == [62, 64]


def test_remove_notes_without_lyrics_keeps_notes_in_an_interior_transcription_gap(tmp_path):
    # Two lyric lines with a gap between them (e.g. the lyrics pipeline failed to transcribe a
    # melisma/ad-lib there) -- a note sitting in that interior gap, with no line coverage of its
    # own, is still real singing between two lines the player is already shown, and should be kept.
    # Notes before the first line / after the last line are still genuine no-context humming and
    # should still be dropped.
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps([
        {"word": "hello", "start": 10.0, "end": 10.4, "line": 0},
        {"word": "world", "start": 10.6, "end": 11.0, "line": 0},
        {"word": "again", "start": 20.0, "end": 20.5, "line": 1},
    ]))

    notes_path = tmp_path / "notes.json"
    notes_path.write_text(json.dumps([
        {"pitch_midi": 60, "onset": 1.0, "offset": 2.0},  # far before the first line -- dropped
        {"pitch_midi": 62, "onset": 10.2, "offset": 10.8},  # inside the first line -- kept
        {"pitch_midi": 63, "onset": 15.0, "offset": 15.5},  # interior gap, no line coverage -- kept
        {"pitch_midi": 64, "onset": 20.1, "offset": 20.4},  # inside the second line -- kept
        {"pitch_midi": 65, "onset": 30.0, "offset": 30.5},  # well past the last line -- dropped
    ]))

    _remove_notes_without_lyrics(notes_path, lyrics_path)

    result = json.loads(notes_path.read_text())
    assert [note["pitch_midi"] for note in result] == [62, 63, 64]


def test_remove_notes_without_lyrics_drops_all_notes_when_lyrics_empty(tmp_path):
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps([]))

    notes_path = tmp_path / "notes.json"
    notes_path.write_text(json.dumps([
        {"pitch_midi": 60, "onset": 1.0, "offset": 2.0},
    ]))

    _remove_notes_without_lyrics(notes_path, lyrics_path)

    result = json.loads(notes_path.read_text())
    assert result == []


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

    def fake_normalize(input_path, output_path, target_i):
        Path(output_path).write_bytes(Path(input_path).read_bytes())
        return Path(output_path)

    with (
        patch("audio_pipeline.pipeline.extract_audio", return_value=extracted_wav),
        patch(
            "audio_pipeline.pipeline.separate_stems",
            return_value=(vocals_path, instrumental_path),
        ),
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result),
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result),
        patch("audio_pipeline.pipeline.normalize_loudness", side_effect=fake_normalize),
    ):
        process_song(
            fake_video_path, cache_dir=cache_dir, song_id="my-song",
            on_progress=lambda stage, fraction=0.0: progress_seen.append(stage),
        )

    assert progress_seen == ["separating", "extracting_melody", "transcribing_lyrics"]


def test_process_song_passes_the_requested_separation_model_through_and_records_it(tmp_path):
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

    def fake_normalize(input_path, output_path, target_i):
        Path(output_path).write_bytes(Path(input_path).read_bytes())
        return Path(output_path)

    with (
        patch("audio_pipeline.pipeline.extract_audio", return_value=extracted_wav),
        patch("audio_pipeline.pipeline.separate_stems") as mock_separate_stems,
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result),
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result),
        patch("audio_pipeline.pipeline.normalize_loudness", side_effect=fake_normalize),
    ):
        mock_separate_stems.return_value = (vocals_path, instrumental_path)
        process_song(
            fake_video_path, cache_dir=cache_dir, song_id="my-song",
            separation_model="htdemucs_ft",
        )

    assert mock_separate_stems.call_args.kwargs["model"] == "htdemucs_ft"

    meta = json.loads((cache_dir / "my-song" / "meta.json").read_text())
    assert meta["separation_model"] == "htdemucs_ft"


def test_process_song_normalizes_the_instrumental_before_publishing(tmp_path):
    # Raw separator output has no consistent loudness (see audio_pipeline/pipeline.py's
    # _INSTRUMENTAL_PLAYBACK_TARGET_LUFS docstring) -- this checks the published
    # instrumental.wav actually goes through normalize_loudness rather than being a bare
    # copy/rename of separate_stems's raw output.
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

    def fake_normalize(input_path, output_path, target_i):
        Path(output_path).write_bytes(b"normalized instrumental")
        return Path(output_path)

    with (
        patch("audio_pipeline.pipeline.extract_audio", return_value=extracted_wav),
        patch(
            "audio_pipeline.pipeline.separate_stems",
            return_value=(vocals_path, instrumental_path),
        ),
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result),
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result),
        patch(
            "audio_pipeline.pipeline.normalize_loudness", side_effect=fake_normalize
        ) as mock_normalize,
    ):
        assets = process_song(fake_video_path, cache_dir=cache_dir, song_id="my-song")

    final_instrumental_path = cache_dir / "my-song" / "instrumental.wav"
    mock_normalize.assert_called_once_with(instrumental_path, final_instrumental_path, -14)
    assert assets.instrumental_path == final_instrumental_path
    assert assets.instrumental_path.read_bytes() == b"normalized instrumental"
    # The raw separator output is cleaned up, not left behind alongside the published copy.
    assert not instrumental_path.exists()


def test_reprocess_melody_and_lyrics_reruns_extraction_from_the_cached_vocal_stem(tmp_path):
    cache_dir = tmp_path / "cache"
    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    vocals_path = song_cache_dir / "vocals.wav"
    vocals_path.write_bytes(b"original vocals")
    instrumental_path = song_cache_dir / "instrumental.wav"
    instrumental_path.write_bytes(b"original instrumental")
    (song_cache_dir / "notes.json").write_text(json.dumps([{"pitch_midi": 60, "onset": 0.0, "offset": 1.0}]))
    (song_cache_dir / "lyrics.json").write_text(json.dumps([{"word": "old", "start": 0.0, "end": 0.5, "line": 0}]))
    (song_cache_dir / "meta.json").write_text(json.dumps({"song_id": "my-song", "title": "My Song"}))

    fresh_midi_path = tmp_path / "fresh_melody.mid"
    fresh_midi_path.write_bytes(b"fresh midi")
    fresh_notes_path = tmp_path / "fresh_notes.json"
    fresh_notes_path.write_text(json.dumps([{"pitch_midi": 62, "onset": 5.0, "offset": 6.0}]))
    fresh_lyrics_path = tmp_path / "fresh_lyrics.json"
    fresh_lyrics_path.write_text(json.dumps([{"word": "new", "start": 5.0, "end": 5.5, "line": 0}]))

    melody_result = type("Melody", (), {"midi_path": fresh_midi_path, "notes_path": fresh_notes_path})()
    lyrics_result = type(
        "Lyrics", (), {"lyrics_path": fresh_lyrics_path, "background_vocal_ranges": []}
    )()

    with (
        patch("audio_pipeline.pipeline.extract_melody", return_value=melody_result) as mock_melody,
        patch("audio_pipeline.pipeline.extract_lyrics", return_value=lyrics_result) as mock_lyrics,
    ):
        assets = reprocess_melody_and_lyrics(cache_dir, "my-song", lyrics_query="My Song")

    mock_melody.assert_called_once_with(vocals_path, song_cache_dir)
    assert mock_lyrics.call_args.args[0] == vocals_path
    assert mock_lyrics.call_args.kwargs["lyrics_query"] == "My Song"

    # instrumental/vocals are left completely untouched -- no re-download, no re-separation
    assert vocals_path.read_bytes() == b"original vocals"
    assert instrumental_path.read_bytes() == b"original instrumental"
    assert assets.vocals_path == vocals_path
    assert assets.instrumental_path == instrumental_path

    # the cached notes/lyrics/midi are replaced with the freshly-extracted (and re-filtered) ones
    assert json.loads(assets.notes_path.read_text()) == [{"pitch_midi": 62, "onset": 5.0, "offset": 6.0}]
    assert json.loads(assets.lyrics_path.read_text()) == [{"word": "new", "start": 5.0, "end": 5.5, "line": 0}]
    assert assets.midi_path.read_bytes() == b"fresh midi"

    # meta.json's other fields (title) survive; only processed_at is refreshed
    meta = json.loads((song_cache_dir / "meta.json").read_text())
    assert meta["title"] == "My Song"
    assert "processed_at" in meta


def test_reprocess_melody_and_lyrics_raises_when_no_cached_stems_exist(tmp_path):
    cache_dir = tmp_path / "cache"

    with pytest.raises(FileNotFoundError):
        reprocess_melody_and_lyrics(cache_dir, "never-processed")
