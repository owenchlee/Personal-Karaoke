"""Tests for the lrclib.net synced-lyrics lookup.

The LRC-parsing/tokenizing/timing logic is tested directly (pure functions,
no network). ``fetch_synced_lyrics`` itself is tested against a mocked
``requests.get`` so the suite never depends on the real lrclib.net service.
"""
from unittest.mock import Mock, patch

from audio_pipeline.lyrics_lookup import (
    _distribute_words,
    _parse_lrc,
    _pick_best_synced_lyrics,
    _tokenize_line,
    _words_from_lrc,
    fetch_synced_lyrics,
)

_CANTONESE_HELLO = chr(0x4F60) + chr(0x597D)  # "你好" ("hello")


def test_parse_lrc_extracts_timestamp_and_text():
    lrc = "[00:12.34]First line\n[00:16.00]Second line"

    assert _parse_lrc(lrc) == [(12.34, "First line"), (16.0, "Second line")]


def test_parse_lrc_skips_metadata_tags_and_blank_lines():
    lrc = "[ar:Some Artist]\n[ti:Some Title]\n[00:01.00]Real line\n[00:05.00]   \n"

    assert _parse_lrc(lrc) == [(1.0, "Real line")]


def test_tokenize_line_splits_latin_text_on_whitespace():
    assert _tokenize_line("the only thing that I know") == [
        "the", "only", "thing", "that", "I", "know",
    ]


def test_tokenize_line_splits_cjk_text_by_character():
    assert _tokenize_line(_CANTONESE_HELLO) == [_CANTONESE_HELLO[0], _CANTONESE_HELLO[1]]


def test_distribute_words_weights_by_character_length_within_the_line_window():
    result = _distribute_words(["a", "bb"], line_start=10.0, line_end=13.0)

    assert result[0] == ("a", 10.0, 11.0)
    assert result[1] == ("bb", 11.0, 13.0)


def test_words_from_lrc_produces_flatten_words_shaped_output():
    # Equal-length words on both lines keep the expected timing simple: an
    # even split of each line's window (weighting-by-length is covered by
    # test_distribute_words_weights_by_character_length_within_the_line_window).
    lrc = "[00:00.00]Hello world\n[00:02.00]Foo Bar"

    result = _words_from_lrc(lrc)

    assert result == [
        {"word": "Hello", "start": 0.0, "end": 1.0, "line": 0},
        {"word": "world", "start": 1.0, "end": 2.0, "line": 0},
        {"word": "Foo", "start": 2.0, "end": 4.0, "line": 1},
        {"word": "Bar", "start": 4.0, "end": 6.0, "line": 1},
    ]


def test_pick_best_synced_lyrics_prefers_closest_duration_match():
    results = [
        {"syncedLyrics": "[00:00.00]Cover version", "duration": 300},
        {"syncedLyrics": "[00:00.00]Original version", "duration": 210},
    ]

    assert _pick_best_synced_lyrics(results, duration_seconds=212) == "[00:00.00]Original version"


def test_pick_best_synced_lyrics_ignores_results_without_synced_lyrics():
    results = [{"plainLyrics": "no timing here"}, {"syncedLyrics": "[00:00.00]Timed"}]

    assert _pick_best_synced_lyrics(results, duration_seconds=None) == "[00:00.00]Timed"


def test_pick_best_synced_lyrics_returns_none_when_no_candidates():
    assert _pick_best_synced_lyrics([{"plainLyrics": "x"}], duration_seconds=None) is None


def test_fetch_synced_lyrics_returns_words_on_a_good_match():
    response = Mock()
    response.json.return_value = [
        {"syncedLyrics": "[00:00.00]Hello world", "duration": 100},
    ]
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response) as mock_get:
        result = fetch_synced_lyrics("some song", duration_seconds=100)

    mock_get.assert_called_once()
    # Single line, no following line to bound it, so the fallback
    # last-line duration (4s) applies, split evenly between the two
    # equal-length words.
    assert result == [
        {"word": "Hello", "start": 0.0, "end": 2.0, "line": 0},
        {"word": "world", "start": 2.0, "end": 4.0, "line": 0},
    ]


def test_fetch_synced_lyrics_returns_none_when_no_results():
    response = Mock()
    response.json.return_value = []
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response):
        assert fetch_synced_lyrics("nonexistent song") is None


def test_fetch_synced_lyrics_returns_none_when_only_unsynced_lyrics_exist():
    response = Mock()
    response.json.return_value = [{"plainLyrics": "no timing here"}]
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response):
        assert fetch_synced_lyrics("some song") is None


def test_fetch_synced_lyrics_returns_none_on_network_error():
    import requests

    with patch(
        "audio_pipeline.lyrics_lookup.requests.get",
        side_effect=requests.ConnectionError("no network"),
    ):
        assert fetch_synced_lyrics("some song") is None
