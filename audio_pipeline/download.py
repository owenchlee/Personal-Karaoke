"""URL -> local audio file download via yt-dlp.

Lets a song be sourced from a link (e.g. a YouTube URL) instead of a local
file, so the browser-facing "paste a link" flow can feed straight into the
existing ``process_song`` pipeline (which already accepts any media file --
see ``video_extraction.py``).
"""
import time
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import yt_dlp

_BASE_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    # YouTube's default "web" client needs a PO token to hand out format URLs
    # and 403s without one; these clients don't, so try them first and fall
    # back across all of them rather than getting stuck retrying whichever
    # one happened to be blocked for this IP at the moment.
    "extractor_args": {"youtube": {"player_client": ["android_vr", "tv", "ios", "web"]}},
}

# Transient YouTube-side failures (rate limiting, a momentarily-blocked
# client/format combo) that are usually gone if you just try again -- as
# opposed to e.g. "video unavailable", where retrying only wastes time.
_RETRY_ERROR_HINTS = ("403", "Forbidden", "HTTP Error 5")
_MAX_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 5


def _resolve_url(url: str) -> str:
    """Turn a YouTube *search results* page (what you get copying the URL bar
    after searching, e.g. ``youtube.com/results?search_query=...``) into a
    yt-dlp search query for its top video hit.

    Feeding a results URL straight to yt-dlp makes it fall into the
    ``youtube:tab`` extractor, which treats the page as a listing to crawl --
    for some queries that surfaces an unrelated unlisted/radio playlist and
    fails with a cryptic "this playlist type is unviewable", even though the
    user never asked for a playlist at all. Resolving to the single top
    result up front sidesteps that extractor entirely.
    """
    parsed = urlparse(url)
    if parsed.hostname not in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        return url
    if parsed.path != "/results":
        return url
    query = parse_qs(parsed.query).get("search_query", [None])[0]
    if not query:
        return url
    return f"ytsearch1:{query}"


def _title_from_info(info: dict) -> str:
    return info.get("title") or info.get("id") or "song"


def _unwrap_search_result(info: dict) -> dict:
    """``ytsearch1:...`` (see ``_resolve_url``) comes back as a one-item
    *playlist* wrapping the actual video info -- its top-level "title" is the
    search query itself, not the video's, and it has none of the per-video
    fields ``prepare_filename`` needs. Unwrap to the single entry so callers
    see the same shape as a direct video URL would give them.
    """
    if info.get("_type") == "playlist":
        entries = list(info.get("entries") or [])
        if entries:
            return entries[0]
    return info


def _extract_info_with_retries(ydl: "yt_dlp.YoutubeDL", url: str, *, download: bool) -> dict:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            info = ydl.extract_info(url, download=download)
            return _unwrap_search_result(info)
        except yt_dlp.utils.DownloadError as exc:
            is_last_attempt = attempt == _MAX_ATTEMPTS
            if is_last_attempt or not any(hint in str(exc) for hint in _RETRY_ERROR_HINTS):
                raise
            time.sleep(_RETRY_DELAY_SECONDS * attempt)  # backoff: 5s, 10s, 15s, 20s
    raise AssertionError("unreachable")  # loop always returns or raises


def probe_title(url: str) -> str:
    """Fetch just ``url``'s reported title, without downloading any media.

    A single fast metadata round-trip -- lets a caller (e.g. the job server)
    compute the same cache slug ``download_audio`` would end up using, so it
    can check whether the song is already cached before paying for a full
    download.
    """
    url = _resolve_url(url)
    ydl_opts = {**_BASE_YDL_OPTS, "skip_download": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = _extract_info_with_retries(ydl, url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"Couldn't fetch info for {url}: {exc}") from exc

    return _title_from_info(info)


def _progress_hook(on_progress: Callable[[float], None]) -> Callable[[dict], None]:
    def hook(d: dict) -> None:
        if d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes")
        if not total or downloaded is None:
            return
        on_progress(min(downloaded / total, 1.0))

    return hook


def download_audio(
    url: str,
    output_dir: str | Path,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, str]:
    """Download the best available audio track for ``url`` into ``output_dir``.

    Returns ``(downloaded_path, title)``. ``title`` is the source's reported
    title (falls back to its id, then "song"), suitable as a human-readable
    song name / cache slug seed.

    ``on_progress``, if given, is called repeatedly with a 0-1 fraction of
    bytes downloaded so far -- lets a caller (e.g. the job server) show real
    download progress instead of a stage that just sits there until it's done.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = _resolve_url(url)

    ydl_opts = {
        **_BASE_YDL_OPTS,
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        # Fetch fragments of a single stream (DASH/HLS sources) in parallel
        # instead of one at a time.
        "concurrent_fragment_downloads": 8,
        "progress_hooks": [_progress_hook(on_progress)] if on_progress else [],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = _extract_info_with_retries(ydl, url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"Couldn't download audio from {url}: {exc}") from exc

    return downloaded_path, _title_from_info(info)
