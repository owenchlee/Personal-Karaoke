"""Tests for the yt-dlp download wrapper.

Mocks ``yt_dlp.YoutubeDL`` so these run offline and fast -- no real network
access, mirroring how the rest of this pipeline's smoke tests avoid needing
external services where a unit boundary allows it.
"""
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from audio_pipeline.download import download_audio, probe_title

_URL = "https://example.com/watch?v=abc123"


def _mock_ydl(info: dict, prepared_filename: str = "song.webm") -> MagicMock:
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.__exit__.return_value = False
    ydl.extract_info.return_value = info
    ydl.prepare_filename.return_value = prepared_filename
    return ydl


def test_probe_title_returns_the_reported_title_without_downloading():
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"})

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl) as YoutubeDL:
        title = probe_title(_URL)

    assert title == "My Song"
    ydl.extract_info.assert_called_once_with(_URL, download=False)
    opts = YoutubeDL.call_args[0][0]
    assert opts["skip_download"] is True


def test_probe_title_falls_back_to_id_then_to_song():
    ydl = _mock_ydl({"title": None, "id": "abc123"})
    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        assert probe_title(_URL) == "abc123"

    ydl = _mock_ydl({"title": None, "id": None})
    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        assert probe_title(_URL) == "song"


def test_probe_title_wraps_download_errors():
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.__exit__.return_value = False
    ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("boom")

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        with pytest.raises(RuntimeError, match="Couldn't fetch info"):
            probe_title(_URL)


def test_download_audio_downloads_and_enables_concurrent_fragments(tmp_path):
    prepared = str(tmp_path / "abc123.webm")
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"}, prepared_filename=prepared)

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl) as YoutubeDL:
        downloaded_path, title = download_audio(_URL, tmp_path)

    assert title == "My Song"
    assert downloaded_path == tmp_path / "abc123.webm"
    ydl.extract_info.assert_called_once_with(_URL, download=True)
    opts = YoutubeDL.call_args[0][0]
    assert opts["format"] == "bestaudio/best"
    assert opts["concurrent_fragment_downloads"] == 8


def test_download_audio_wraps_download_errors(tmp_path):
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.__exit__.return_value = False
    ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("boom")

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        with pytest.raises(RuntimeError, match="Couldn't download audio"):
            download_audio(_URL, tmp_path)


def test_download_audio_retries_transient_403_then_succeeds(tmp_path):
    prepared = str(tmp_path / "abc123.webm")
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"}, prepared_filename=prepared)
    ydl.extract_info.side_effect = [
        yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden"),
        {"title": "My Song", "id": "abc123"},
    ]

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl), \
            patch("audio_pipeline.download.time.sleep") as sleep:
        downloaded_path, title = download_audio(_URL, tmp_path)

    assert title == "My Song"
    assert downloaded_path == tmp_path / "abc123.webm"
    assert ydl.extract_info.call_count == 2
    sleep.assert_called_once()


def test_download_audio_does_not_retry_non_transient_errors(tmp_path):
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.__exit__.return_value = False
    ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Video unavailable")

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl), \
            patch("audio_pipeline.download.time.sleep") as sleep:
        with pytest.raises(RuntimeError, match="Couldn't download audio"):
            download_audio(_URL, tmp_path)

    assert ydl.extract_info.call_count == 1
    sleep.assert_not_called()


def test_download_audio_resolves_youtube_search_results_url_to_top_hit(tmp_path):
    prepared = str(tmp_path / "abc123.webm")
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"}, prepared_filename=prepared)
    search_url = "https://www.youtube.com/results?search_query=here%27s+your+perfect+lyrics"

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        downloaded_path, title = download_audio(search_url, tmp_path)

    assert title == "My Song"
    assert downloaded_path == tmp_path / "abc123.webm"
    ydl.extract_info.assert_called_once_with(
        "ytsearch1:here's your perfect lyrics", download=True
    )


def test_probe_title_resolves_youtube_search_results_url_to_top_hit():
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"})
    search_url = "https://www.youtube.com/results?search_query=some+song"

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        title = probe_title(search_url)

    assert title == "My Song"
    ydl.extract_info.assert_called_once_with("ytsearch1:some song", download=False)


def test_probe_title_leaves_regular_urls_unchanged():
    ydl = _mock_ydl({"title": "My Song", "id": "abc123"})

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        probe_title(_URL)

    ydl.extract_info.assert_called_once_with(_URL, download=False)


def test_download_audio_unwraps_ytsearch_playlist_result(tmp_path):
    """``ytsearch1:`` comes back as a one-item *playlist* -- its top-level
    "title" is the search query, not the video's, and prepare_filename would
    build a bogus name from it. Real video info lives in entries[0].
    """
    prepared = str(tmp_path / "abc123.webm")
    ydl = _mock_ydl(
        {
            "_type": "playlist",
            "title": "here's your perfect lyrics",
            "entries": [{"title": "Real Video Title", "id": "abc123"}],
        },
        prepared_filename=prepared,
    )
    search_url = "https://www.youtube.com/results?search_query=here%27s+your+perfect+lyrics"

    with patch("audio_pipeline.download.yt_dlp.YoutubeDL", return_value=ydl):
        downloaded_path, title = download_audio(search_url, tmp_path)

    assert title == "Real Video Title"
    assert downloaded_path == tmp_path / "abc123.webm"
    ydl.prepare_filename.assert_called_once_with({"title": "Real Video Title", "id": "abc123"})
