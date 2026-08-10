"""Tests for the lrclib.net synced-lyrics lookup.

The LRC-parsing/tokenizing/timing logic is tested directly (pure functions,
no network). ``fetch_synced_lyrics`` itself is tested against a mocked
``requests.get`` so the suite never depends on the real lrclib.net service.
"""
from unittest.mock import Mock, patch

from audio_pipeline.lyrics_lookup import (
    _distribute_words,
    _parse_lrc,
    _parse_title,
    _pick_best_synced_lyrics,
    _tokenize_line,
    _words_from_lrc,
    clean_display_title,
    fetch_synced_lyrics,
)

_CANTONESE_HELLO = chr(0x4F60) + chr(0x597D)  # "你好" ("hello")


def test_parse_lrc_extracts_timestamp_and_text():
    lrc = "[00:12.34]First line\n[00:16.00]Second line"

    assert _parse_lrc(lrc) == [(12.34, "First line"), (16.0, "Second line")]


def test_parse_lrc_skips_metadata_tags_and_blank_lines():
    lrc = "[ar:Some Artist]\n[ti:Some Title]\n[00:01.00]Real line\n[00:05.00]   \n"

    assert _parse_lrc(lrc) == [(1.0, "Real line")]


def test_parse_lrc_expands_a_line_with_multiple_timestamp_tags():
    # lrclib tags a repeated line (e.g. a chorus) with multiple timestamps
    # instead of duplicating the text -- each tag must become its own timed
    # entry, sharing that text, or the later repeat silently loses its
    # lyrics and the second tag leaks into the displayed word as literal text.
    lrc = "[00:10.00][00:30.00]Chorus line\n[00:20.00]Verse line"

    assert _parse_lrc(lrc) == [
        (10.0, "Chorus line"),
        (20.0, "Verse line"),
        (30.0, "Chorus line"),
    ]


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

    words, background_ranges = _words_from_lrc(lrc)

    assert words == [
        {"word": "Hello", "start": 0.0, "end": 1.0, "line": 0},
        {"word": "world", "start": 1.0, "end": 2.0, "line": 0},
        {"word": "Foo", "start": 2.0, "end": 4.0, "line": 1},
        {"word": "Bar", "start": 4.0, "end": 6.0, "line": 1},
    ]
    assert background_ranges == []


def test_words_from_lrc_caps_a_short_lines_duration_instead_of_filling_a_long_gap():
    # "Hi" starting at 0s followed by a line that doesn't start until 20s
    # later (e.g. an instrumental break) must not stretch "Hi"'s fill
    # animation across that whole 20s gap -- it's capped to a plausible
    # singing duration for its own two characters instead.
    lrc = "[00:00.00]Hi\n[00:20.00]Next line"

    words, _background_ranges = _words_from_lrc(lrc)

    assert words[0] == {"word": "Hi", "start": 0.0, "end": 1.0, "line": 0}


def test_words_from_lrc_strips_a_trailing_backing_vocal_from_a_line():
    # Real lrclib data (e.g. "that I fell for you (mm-hmm)") tucks a backing/ad-lib vocal in
    # parentheses at the end of an otherwise-real line -- only the real lyric should be displayed.
    lrc = "[00:00.00]that I fell for you (mm-hmm)\n[00:10.00]Next line"

    words, background_ranges = _words_from_lrc(lrc)

    assert [w["word"] for w in words if w["line"] == 0] == ["that", "I", "fell", "for", "you"]
    assert background_ranges == []


def test_words_from_lrc_drops_a_line_that_is_entirely_a_backing_vocal():
    # e.g. "(I hope it's worth it, I hope it's worth it)" -- nothing real left after stripping the
    # parenthetical, so the whole line is dropped from lyrics *and* its span is reported so callers
    # can also exclude it from the note highway.
    lrc = "[00:00.00]Real line\n[00:05.00](I hope it's worth it)\n[00:09.00]Another line"

    words, background_ranges = _words_from_lrc(lrc)

    assert [w["line"] for w in words] == [0, 0, 1, 1]
    assert background_ranges == [(5.0, 9.0)]


def test_words_from_lrc_handles_square_bracket_backing_vocals_too():
    lrc = "[00:00.00]Real line [background]"

    words, background_ranges = _words_from_lrc(lrc)

    assert [w["word"] for w in words] == ["Real", "line"]
    assert background_ranges == []


def test_parse_title_splits_artist_and_track_on_a_spaced_dash():
    assert _parse_title("Adele - Hello") == ("Hello", "Adele")


def test_parse_title_strips_promotional_bracket_tags():
    assert _parse_title("Adele - Hello (Official Music Video) [4K]") == ("Hello", "Adele")
    assert _parse_title("Photograph (Official Audio)") == ("Photograph", None)


def test_parse_title_keeps_a_real_title_parenthetical():
    # Only bracket groups that read as promotional noise are stripped -- a genuine subtitle stays.
    assert _parse_title("Sign of the Times (Interlude)") == ("Sign of the Times (Interlude)", None)


def test_parse_title_strips_a_featuring_credit():
    assert _parse_title("DJ Khaled - Wild Thoughts (feat. Rihanna)") == ("Wild Thoughts", "DJ Khaled")
    assert _parse_title("Song Name ft. Someone Else") == ("Song Name", None)


def test_parse_title_leaves_an_intra_word_hyphen_alone():
    # "Jay-Z" must not be split into artist "Jay" / track "Z" -- only a spaced dash separates fields.
    assert _parse_title("Jay-Z - 99 Problems") == ("99 Problems", "Jay-Z")


def test_parse_title_without_a_dash_has_no_artist():
    assert _parse_title("Bohemian Rhapsody") == ("Bohemian Rhapsody", None)


def test_parse_title_falls_back_to_the_raw_query_when_cleaning_empties_it():
    # A title that is nothing but a bracketed noise tag shouldn't become an empty search string.
    assert _parse_title("(Official Video)") == ("(Official Video)", None)


def test_parse_title_strips_bare_noise_tags_glued_onto_cjk_words():
    # Real Cantonese upload title: promotional tags ("lossless FLAC", "lyrics", "pure enjoy") glued
    # directly onto CJK words with no bracket and no space -- confirmed live against lrclib that the
    # raw title returns zero search results, silently falling through to local transcription, while
    # the cleaned title matches the correct song immediately (see the module comment above
    # `_strip_noise_tokens`).
    assert _parse_title("陳慧嫻 夜機 無損音樂FLAC 歌詞LYRICS 純享") == ("陳慧嫻 夜機", None)


def test_parse_title_bare_noise_stripping_does_not_eat_a_real_short_word():
    # A real bare title word must not be dropped just because a noise keyword is a substring of it
    # somewhere in the middle -- only an exact/prefix/suffix match counts as noise.
    assert _parse_title("Redheaded Stranger") == ("Redheaded Stranger", None)


def test_clean_display_title_strips_bracket_noise_and_emoji_but_keeps_the_dash():
    # Unlike _parse_title, display cleaning must never split/reorder on the dash.
    assert clean_display_title("Love In The Dark - Adele (Lyrics) \U0001F3B5") == "Love In The Dark - Adele"


def test_clean_display_title_strips_bare_noise_tags_glued_onto_cjk_words():
    assert clean_display_title("陳慧嫻 夜機 無損音樂FLAC 歌詞LYRICS 純享") == "陳慧嫻 夜機"


def test_clean_display_title_keeps_a_real_title_parenthetical():
    assert clean_display_title("Sign of the Times (Interlude)") == "Sign of the Times (Interlude)"


def test_clean_display_title_falls_back_to_the_raw_title_when_cleaning_empties_it():
    assert clean_display_title("(Official Video)") == "(Official Video)"


def test_fetch_synced_lyrics_uses_structured_search_when_an_artist_is_present():
    response = Mock()
    response.json.return_value = [{"syncedLyrics": "[00:00.00]Hello world", "duration": 100}]
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response) as mock_get:
        fetch_synced_lyrics("Adele - Hello (Official Music Video)", duration_seconds=100)

    _args, kwargs = mock_get.call_args
    assert kwargs["params"] == {"track_name": "Hello", "artist_name": "Adele"}


def test_fetch_synced_lyrics_falls_back_to_free_text_when_structured_search_misses():
    empty = Mock()
    empty.json.return_value = []
    hit = Mock()
    hit.json.return_value = [{"syncedLyrics": "[00:00.00]Hello world", "duration": 100}]
    with patch(
        "audio_pipeline.lyrics_lookup.requests.get", side_effect=[empty, hit]
    ) as mock_get:
        result = fetch_synced_lyrics("Adele - Hello", duration_seconds=100)

    assert result is not None
    assert mock_get.call_count == 2
    # Second (fallback) attempt is a free-text query combining the cleaned artist and track.
    _args, kwargs = mock_get.call_args_list[1]
    assert kwargs["params"] == {"q": "Adele Hello"}


def test_pick_best_synced_lyrics_prefers_closest_duration_match():
    results = [
        {"syncedLyrics": "[00:00.00]Cover version", "duration": 300},
        {"syncedLyrics": "[00:00.00]Original version", "duration": 210},
    ]

    assert _pick_best_synced_lyrics(results, duration_seconds=212) == "[00:00.00]Original version"


def test_pick_best_synced_lyrics_skips_a_closer_duration_match_by_the_wrong_artist():
    # Real-world case: lrclib's title-only search for a common song title (e.g. "Demons") returns
    # several completely unrelated songs that happen to share that title. A wrong-artist candidate
    # must lose to a right-artist one even when it's the closer duration match.
    results = [
        {
            "syncedLyrics": "[00:00.00]Wrong song entirely",
            "duration": 210,
            "artistName": "Some Other Band",
        },
        {
            "syncedLyrics": "[00:00.00]Real song",
            "duration": 236,
            "artistName": "Imagine Dragons",
        },
    ]

    assert (
        _pick_best_synced_lyrics(
            results, duration_seconds=212, query="Imagine Dragons - Demons (Official Music Video)"
        )
        == "[00:00.00]Real song"
    )


def test_pick_best_synced_lyrics_accepts_closest_duration_when_artist_metadata_is_missing():
    # A candidate with no artistName at all can't be checked against the query -- don't reject it
    # on missing metadata, just fall through to the existing duration/script behavior.
    results = [{"syncedLyrics": "[00:00.00]Original version", "duration": 210}]

    assert (
        _pick_best_synced_lyrics(results, duration_seconds=212, query="Some Song Title")
        == "[00:00.00]Original version"
    )


def test_pick_best_synced_lyrics_accepts_closest_duration_when_query_is_not_given():
    results = [{"syncedLyrics": "[00:00.00]Original version", "duration": 210, "artistName": "Anyone"}]

    assert _pick_best_synced_lyrics(results, duration_seconds=212) == "[00:00.00]Original version"


def test_pick_best_synced_lyrics_ignores_results_without_synced_lyrics():
    results = [{"plainLyrics": "no timing here"}, {"syncedLyrics": "[00:00.00]Timed"}]

    assert _pick_best_synced_lyrics(results, duration_seconds=None) == "[00:00.00]Timed"


def test_pick_best_synced_lyrics_returns_none_when_no_candidates():
    assert _pick_best_synced_lyrics([{"plainLyrics": "x"}], duration_seconds=None) is None


def test_pick_best_synced_lyrics_skips_a_closer_duration_match_in_the_wrong_script():
    # lrclib matches by title text only -- a same-titled Thai song can be
    # the closest-duration result even when a real English/Cantonese match
    # also exists further away in duration.
    thai_line = "[00:00.00]" + "สวัสดี" * 10
    english_line = "[00:00.00]" + "the only thing that I know how to do"
    results = [
        {"syncedLyrics": thai_line, "duration": 210},
        {"syncedLyrics": english_line, "duration": 240},
    ]

    assert _pick_best_synced_lyrics(results, duration_seconds=212) == english_line


def test_pick_best_synced_lyrics_returns_none_when_every_candidate_is_wrong_script():
    thai_line = "[00:00.00]" + "สวัสดี" * 10
    results = [{"syncedLyrics": thai_line, "duration": 210}]

    assert _pick_best_synced_lyrics(results, duration_seconds=212) is None


def test_fetch_synced_lyrics_returns_words_on_a_good_match():
    response = Mock()
    response.json.return_value = [
        {"syncedLyrics": "[00:00.00]Hello world", "duration": 100},
    ]
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response) as mock_get:
        result = fetch_synced_lyrics("some song", duration_seconds=100)

    mock_get.assert_called_once()
    words, background_ranges = result
    # Single line, no following line to bound it, so the fallback
    # last-line duration (4s) applies, split evenly between the two
    # equal-length words.
    assert words == [
        {"word": "Hello", "start": 0.0, "end": 2.0, "line": 0},
        {"word": "world", "start": 2.0, "end": 4.0, "line": 0},
    ]
    assert background_ranges == []


def test_fetch_synced_lyrics_returns_background_vocal_ranges_alongside_words():
    response = Mock()
    response.json.return_value = [
        {
            "syncedLyrics": "[00:00.00]Real line\n[00:05.00](backing vocal)\n[00:09.00]Another line",
            "duration": 9,
        },
    ]
    with patch("audio_pipeline.lyrics_lookup.requests.get", return_value=response):
        result = fetch_synced_lyrics("some song", duration_seconds=9)

    words, background_ranges = result
    assert [w["word"] for w in words] == ["Real", "line", "Another", "line"]
    assert background_ranges == [(5.0, 9.0)]


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
