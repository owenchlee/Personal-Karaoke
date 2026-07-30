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

from audio_pipeline.text_script import is_mostly_supported_script

_SEARCH_URL = "https://lrclib.net/api/search"
_REQUEST_TIMEOUT_SECONDS = 10
_FALLBACK_LAST_LINE_DURATION_SECONDS = 4.0

# lrclib only tells us when a line *starts*; naively treating "the next
# line's start" as "this line's end" is fine back-to-back, but any pause
# before that next line -- a held note trailing into silence, an
# instrumental break -- then gets counted as this line's own singing time
# and distributed across its words, so a short line's fill animation
# crawls on for the length of the following gap instead of finishing when
# it's actually done. Cap a line's assumed duration to something plausible
# for its own text length instead of trusting an arbitrarily long gap.
_MAX_SECONDS_PER_CHAR = 0.5

_LRC_TAG_RE = re.compile(r"^\[(\d{1,3}):(\d{2}(?:\.\d{1,3})?)\]")
_CJK_RE = re.compile("[一-鿿]")

_WORD_RE = re.compile(r"[a-z0-9]+")

# lrclib's plain-text lines routinely include the backing/ad-lib vocal alongside the lead line
# (e.g. "that I fell for you (mm-hmm)"), or as an entire line of its own (e.g. "(I hope it's worth
# it)") -- these are not what the player is meant to sing, so they're stripped out of the displayed
# lyrics entirely. Matches both parenthesized and square-bracketed spans (either convention shows up
# depending on the source release), non-nested.
_BACKGROUND_VOCAL_RE = re.compile(r"[(\[][^)\]]*[)\]]")


def _strip_background_vocals(text: str) -> str:
    stripped = _BACKGROUND_VOCAL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def fetch_synced_lyrics(
    query: str, duration_seconds: float | None = None
) -> tuple[list[dict], list[tuple[float, float]]] | None:
    """Look up ``query`` (e.g. a song title) on lrclib.net and return a word
    list in the same shape ``lyrics_extraction._flatten_words`` produces:
    ``[{"word", "start", "end", "line"}, ...]``, alongside a list of
    ``(start, end)`` time ranges that were entirely backing/ad-lib vocals
    (see ``_strip_background_vocals``) -- callers should exclude these same
    ranges from anything else derived from this song's timing, e.g. the note
    highway, since they aren't part of the line the player is meant to sing.

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

    synced = _pick_best_synced_lyrics(results, duration_seconds, query)
    if not synced:
        return None

    words, background_ranges = _words_from_lrc(synced)
    if not words:
        return None
    return words, background_ranges


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


# lrclib's search endpoint does a loose full-text match across track/artist/album text, not a
# "this is the song you meant" match -- querying it for just "Demons" surfaces at least half a
# dozen genuinely different songs actually titled "Demons" (by Fiddler On Green, DJ Paul Elstak,
# Fatboy Slim, Kurt Hugo Schneider, and others), any of which can coincidentally land closer in
# duration to a given video than the real Imagine Dragons track -- confirmed directly as the cause
# of a real user report of getting entirely wrong lyrics for that song. Duration-closeness alone
# has no defense against this: two unrelated songs sharing a title easily share a duration too.
# Require the candidate's own reported artist to actually appear in the query text (as it would
# for a realistic video title like "Imagine Dragons - Demons (Official Video)") before trusting a
# duration match -- this doesn't need an exact title match, since real queries carry extra noise
# ("(Official Video)", "[Lyrics]", etc.) that a strict equality check would break on.
def _artist_matches_query(candidate: dict, query_words: set[str]) -> bool:
    artist_words = _words(candidate.get("artistName") or "")
    if not artist_words or not query_words:
        return True  # nothing to check the query against -- don't reject on missing metadata
    return not artist_words.isdisjoint(query_words)


def _pick_best_synced_lyrics(
    results: list[dict], duration_seconds: float | None, query: str | None = None
) -> str | None:
    """Pick the closest-duration candidate whose lyrics are actually in one
    of the two scripts this app supports, and whose reported artist is
    plausibly the one the query was actually about.

    lrclib matches by title text only, so a same/similar-titled song in a
    completely different language (e.g. Thai) or by a completely different
    artist (e.g. a different band's own song called "Demons") can be the
    closest-duration result -- walk candidates in closest-duration order and
    skip any whose lyrics are predominantly a wrong script or whose artist
    doesn't show up anywhere in the query, rather than trusting the single
    closest match and returning known-wrong lyrics verbatim.
    """
    candidates = [r for r in results if r.get("syncedLyrics")]
    if duration_seconds is not None:
        candidates = sorted(
            candidates, key=lambda r: abs((r.get("duration") or 0) - duration_seconds)
        )
    query_words = _words(query) if query else set()
    for candidate in candidates:
        if not _artist_matches_query(candidate, query_words):
            continue
        lyrics = candidate["syncedLyrics"]
        if is_mostly_supported_script(lyrics):
            return lyrics
    return None


def _parse_lrc(lrc_text: str) -> list[tuple[float, str]]:
    """Parse standard ``[mm:ss.xx]text`` synced-lyrics lines, skipping
    metadata tags (``[ar:...]``, ``[offset:...]``, etc.) and blank lines.

    A single line can carry more than one timestamp tag (e.g.
    ``[01:02.34][03:45.67]Some lyrics``) -- lrclib's convention for a line
    that recurs verbatim, such as a repeated chorus, instead of duplicating
    the text. Each leading tag produces its own ``(timestamp, text)`` entry
    sharing that same text, since otherwise every occurrence but the first
    would be silently dropped -- leaving that whole repeat of the song with
    no lyric data, and the line's stray second tag rendered as literal text.
    Entries are re-sorted by timestamp afterward, since a repeated line's
    later occurrence is written inline with its first rather than at its own
    position in the file.
    """
    lines = []
    for raw_line in lrc_text.splitlines():
        remainder = raw_line.strip()
        timestamps = []
        while True:
            match = _LRC_TAG_RE.match(remainder)
            if not match:
                break
            minutes, seconds = match.groups()
            timestamps.append(int(minutes) * 60 + float(seconds))
            remainder = remainder[match.end():]
        if not timestamps:
            continue
        text = remainder.strip()
        if not text:
            continue
        for timestamp in timestamps:
            lines.append((timestamp, text))
    lines.sort(key=lambda entry: entry[0])
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


def _words_from_lrc(lrc_text: str) -> tuple[list[dict], list[tuple[float, float]]]:
    parsed_lines = _parse_lrc(lrc_text)
    words = []
    background_ranges: list[tuple[float, float]] = []
    line = 0
    for i, (start, raw_text) in enumerate(parsed_lines):
        next_start = parsed_lines[i + 1][0] if i + 1 < len(parsed_lines) else None
        end = next_start if next_start is not None else start + _FALLBACK_LAST_LINE_DURATION_SECONDS

        text = _strip_background_vocals(raw_text)
        tokens = _tokenize_line(text)
        if not tokens:
            # A line that had real text but nothing left after stripping backing/ad-lib
            # vocals (e.g. "(I hope it's worth it)") was entirely background -- exclude its
            # whole span from anything else derived from this song's timing too, not just
            # the displayed lyrics.
            if _tokenize_line(raw_text):
                background_ranges.append((start, end))
            continue
        if next_start is not None:
            total_chars = sum(len(token) for token in tokens) or len(tokens)
            max_plausible_end = start + total_chars * _MAX_SECONDS_PER_CHAR
            end = min(next_start, max_plausible_end)
        for word, word_start, word_end in _distribute_words(tokens, start, end):
            words.append({"word": word, "start": word_start, "end": word_end, "line": line})
        line += 1
    return words, background_ranges
