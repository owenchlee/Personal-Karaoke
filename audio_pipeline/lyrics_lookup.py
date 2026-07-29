"""Online synced-lyrics lookup via lrclib.net.

lrclib.net is a free, open community database of time-synced lyrics built
specifically for third-party media/karaoke apps to query -- unlike lyrics
sites such as Genius or AZLyrics, it has no auth wall and its terms allow
exactly this kind of automated per-song lookup.

Used as a first attempt at lyrics for a song: when a match with synced
lyrics exists, its text is already known-correct (unlike a transcription)
and already time-matched to the song at line granularity. lrclib only
gives per-line timestamps, not per-word, so each line's words are given an
even split of that line's time window -- an approximation, not a measured
alignment. Callers should fall back to local transcription
(``lyrics_extraction.extract_lyrics``) when this returns ``None``.
"""
import re

import requests

_SEARCH_URL = "https://lrclib.net/api/search"
_REQUEST_TIMEOUT_SECONDS = 10
_FALLBACK_LAST_LINE_DURATION_SECONDS = 4.0

_LRC_LINE_RE = re.compile(r"^\[(\d{1,3}):(\d{2}(?:\.\d{1,3})?)\](.*)$")
_CJK_RE = re.compile("[一-鿿]")


def fetch_synced_lyrics(query: str, duration_seconds: float | None = None) -> list[dict] | None:
    """Look up ``query`` (e.g. a song title) on lrclib.net and return a word
    list in the same shape ``lyrics_extraction._flatten_words`` produces:
    ``[{"word", "start", "end", "line"}, ...]``.

    ``duration_seconds``, if given, is used to prefer the search result
    closest in length to the actual audio, since title search alone can
    return same-titled covers/remixes with different timing.

    Returns ``None`` if no synced match was found or the lookup failed for
    any reason (network error, no results, only unsynced lyrics available)
    -- callers should treat that as "try local transcription instead", not
    as an error.
    """
    try:
        response = requests.get(
            _SEARCH_URL, params={"q": query}, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        results = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(results, list) or not results:
        return None

    synced = _pick_best_synced_lyrics(results, duration_seconds)
    if not synced:
        return None

    words = _words_from_lrc(synced)
    return words or None


def _pick_best_synced_lyrics(results: list[dict], duration_seconds: float | None) -> str | None:
    candidates = [r for r in results if r.get("syncedLyrics")]
    if not candidates:
        return None
    if duration_seconds is None:
        return candidates[0]["syncedLyrics"]
    best = min(
        candidates,
        key=lambda r: abs((r.get("duration") or 0) - duration_seconds),
    )
    return best["syncedLyrics"]


def _parse_lrc(lrc_text: str) -> list[tuple[float, str]]:
    """Parse standard ``[mm:ss.xx]text`` synced-lyrics lines, skipping
    metadata tags (``[ar:...]``, ``[offset:...]``, etc.) and blank lines.
    """
    lines = []
    for raw_line in lrc_text.splitlines():
        match = _LRC_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        text = match.group(3).strip()
        if not text:
            continue
        minutes, seconds = match.groups()[0], match.groups()[1]
        timestamp = int(minutes) * 60 + float(seconds)
        lines.append((timestamp, text))
    return lines


def _tokenize_line(text: str) -> list[str]:
    """Split a lyric line into displayable "word" units. CJK text (e.g.
    Cantonese) has no spaces between words, so it's split character by
    character -- matching how faster-whisper itself tokenizes Chinese
    script word-by-word -- while Latin-script text splits on whitespace.
    """
    if _CJK_RE.search(text):
        return [char for char in text if not char.isspace()]
    return text.split()


def _distribute_words(
    tokens: list[str], line_start: float, line_end: float
) -> list[tuple[str, float, float]]:
    """Split [line_start, line_end) across tokens, weighted by character
    length, as an approximation of individual word timing within a line
    that's only known to be synced at line granularity.
    """
    duration = max(line_end - line_start, 0.01)
    total_chars = sum(len(token) for token in tokens) or len(tokens)
    out = []
    cursor = line_start
    for token in tokens:
        share = len(token) / total_chars if total_chars else 1 / len(tokens)
        word_end = min(cursor + duration * share, line_end)
        if word_end <= cursor:
            word_end = cursor + 0.01
        out.append((token, round(cursor, 4), round(word_end, 4)))
        cursor = word_end
    return out


def _words_from_lrc(lrc_text: str) -> list[dict]:
    parsed_lines = _parse_lrc(lrc_text)
    words = []
    line = 0
    for i, (start, text) in enumerate(parsed_lines):
        end = (
            parsed_lines[i + 1][0]
            if i + 1 < len(parsed_lines)
            else start + _FALLBACK_LAST_LINE_DURATION_SECONDS
        )
        tokens = _tokenize_line(text)
        if not tokens:
            continue
        for word, word_start, word_end in _distribute_words(tokens, start, end):
            words.append({"word": word, "start": word_start, "end": word_end, "line": line})
        line += 1
    return words
