"""Tests for the faster-whisper lyrics transcription wrapper.

The pure `_flatten_words` reshaping logic is tested directly against
constructed faster-whisper types (no model needed). The end-to-end smoke
test uses a synthetic (non-speech) clip -- this only proves the pipeline
runs and produces valid JSON, not that transcription is accurate against
real singing (that's a manual check, see NOTES.md).
"""
import json
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from faster_whisper.transcribe import Segment, Word

from audio_pipeline.lyrics_extraction import (
    _align_synced_lyrics_to_audio,
    _detect_language,
    _find_first_vocal_onset,
    _find_last_vocal_activity,
    _fit_time_correction,
    _flatten_words,
    _repair_energetic_gaps,
    _transcribe_chunked,
    extract_lyrics,
)


def _word(word, start, end, probability=0.9):
    return Word(start=start, end=end, word=word, probability=probability)


def _segment(words):
    return Segment(
        id=0, seek=0, start=words[0].start, end=words[-1].end,
        text="".join(w.word for w in words), tokens=[], avg_logprob=0.0,
        compression_ratio=0.0, no_speech_prob=0.0, words=words, temperature=0.0,
    )


class _FakeModel:
    """Stands in for WhisperModel.detect_language's return shape."""

    def __init__(self, all_language_probs):
        self._all_language_probs = all_language_probs

    def detect_language(self, audio, vad_filter):
        top_lang, top_prob = self._all_language_probs[0]
        return top_lang, top_prob, self._all_language_probs


def test_detect_language_picks_yue_when_it_outranks_en():
    model = _FakeModel([("zh", 0.6), ("yue", 0.3), ("en", 0.05)])

    assert _detect_language(model, audio=None) == "yue"


def test_detect_language_picks_en_when_it_outranks_yue():
    model = _FakeModel([("en", 0.7), ("zh", 0.2), ("yue", 0.05)])

    assert _detect_language(model, audio=None) == "en"


def test_detect_language_never_returns_an_unsupported_language():
    model = _FakeModel([("ja", 0.9), ("ko", 0.05)])

    assert _detect_language(model, audio=None) in {"en", "yue"}


def test_flatten_words_extracts_stripped_text_timing_and_line():
    segments = [_segment([_word(" Hello", 0.0, 0.3), _word(" world", 0.3, 0.7)])]

    result = _flatten_words(segments)

    assert result == [
        {"word": "Hello", "start": 0.0, "end": 0.3, "line": 0},
        {"word": "world", "start": 0.3, "end": 0.7, "line": 0},
    ]


def test_flatten_words_assigns_sequential_line_indices_per_segment():
    segments = [
        _segment([_word(" First", 0.0, 0.3)]),
        _segment([_word(" Second", 1.0, 1.3), _word(" line", 1.3, 1.6)]),
    ]

    result = _flatten_words(segments)

    assert [w["line"] for w in result] == [0, 1, 1]


def test_flatten_words_skips_segments_without_words_and_blank_words():
    segments = [
        _segment([_word("  ", 0.0, 0.1), _word(" real", 0.1, 0.4)]),
        Segment(
            id=1, seek=0, start=1.0, end=1.5, text="", tokens=[], avg_logprob=0.0,
            compression_ratio=0.0, no_speech_prob=0.0, words=None, temperature=0.0,
        ),
    ]

    result = _flatten_words(segments)

    assert result == [{"word": "real", "start": 0.1, "end": 0.4, "line": 0}]


def test_flatten_words_does_not_reserve_a_line_number_for_empty_segments():
    segments = [
        _segment([_word(" First", 0.0, 0.3)]),
        Segment(
            id=1, seek=0, start=0.5, end=0.6, text="", tokens=[], avg_logprob=0.0,
            compression_ratio=0.0, no_speech_prob=0.0, words=None, temperature=0.0,
        ),
        _segment([_word(" Second", 1.0, 1.3)]),
    ]

    result = _flatten_words(segments)

    assert [w["line"] for w in result] == [0, 1]


def test_flatten_words_drops_zero_duration_words_as_hallucination_artifacts():
    segments = [
        _segment(
            [
                _word(" real", 0.0, 0.3),
                _word(" glitch", 0.3, 0.3),
                _word(" repeat", 0.3, 0.3),
            ]
        ),
    ]

    result = _flatten_words(segments)

    assert result == [{"word": "real", "start": 0.0, "end": 0.3, "line": 0}]


def test_flatten_words_drops_words_hallucinated_in_an_unsupported_script():
    # Observed in practice: even with transcription forced to "yue", a
    # low-confidence chunk can still come back with stray Thai text instead
    # of Cantonese -- whisper's `language=` argument only sets the decoder's
    # initial token, it doesn't hard-constrain every token after that.
    segments = [
        _segment(
            [
                _word(" real", 0.0, 0.3),
                _word(" สวัสดี", 0.3, 0.6),
                _word(" word", 0.6, 0.9),
            ]
        ),
    ]

    result = _flatten_words(segments)

    assert result == [
        {"word": "real", "start": 0.0, "end": 0.3, "line": 0},
        {"word": "word", "start": 0.6, "end": 0.9, "line": 0},
    ]


def test_flatten_words_keeps_cantonese_words():
    segments = [_segment([_word(" 愛你", 0.0, 0.3), _word(" 喇", 0.3, 0.6)])]

    result = _flatten_words(segments)

    assert result == [
        {"word": "愛你", "start": 0.0, "end": 0.3, "line": 0},
        {"word": "喇", "start": 0.3, "end": 0.6, "line": 0},
    ]


def test_flatten_words_does_not_reserve_a_line_number_for_all_zero_duration_segments():
    segments = [
        _segment([_word(" First", 0.0, 0.3)]),
        _segment([_word(" glitch", 0.5, 0.5), _word(" repeat", 0.5, 0.5)]),
        _segment([_word(" Second", 1.0, 1.3)]),
    ]

    result = _flatten_words(segments)

    assert [w["line"] for w in result] == [0, 1]


class _FakeTranscribeModel:
    """Stands in for WhisperModel.transcribe, returning one canned segment
    list per call, in order, and recording each call's audio length so
    tests can assert on the exact slice it was given.
    """

    def __init__(self, segments_per_call):
        self._segments_per_call = list(segments_per_call)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio_len": len(audio), **kwargs})
        return self._segments_per_call.pop(0), None


def test_transcribe_chunked_makes_a_single_call_for_short_audio():
    model = _FakeTranscribeModel([[_segment([_word(" hi", 0.0, 0.3)])]])

    result = _transcribe_chunked(model, audio=list(range(30)), sampling_rate=1, language="en")

    assert len(model.calls) == 1
    assert model.calls[0]["audio_len"] == 30
    assert model.calls[0]["language"] == "en"
    assert result == [{"word": "hi", "start": 0.0, "end": 0.3, "line": 0}]


def test_transcribe_chunked_splits_long_audio_and_offsets_timestamps():
    # sampling_rate=1 so seconds == sample count, keeping the test's fake
    # audio (a plain list) directly indexable by the times used below.
    # 80s total audio: with the module's 50s/5s chunk/overlap constants,
    # this yields exactly two windows -- core [0, 40) padded to [0, 45),
    # and core [40, 80) padded to [35, 80) (no overlap past the end for the
    # last window).
    chunk0_segments = [
        _segment(
            [
                _word(" a", 1.0, 2.0),  # inside chunk0's core -> kept
                _word(" b", 39.0, 40.0),  # still inside core (< 40) -> kept
                _word(" c", 42.0, 43.0),  # in chunk0's trailing pad -> dropped
            ]
        )
    ]
    chunk1_segments = [
        _segment(
            [
                # local 3-4 -> absolute 38-39, inside chunk1's leading pad
                # (before its core starts at 40) -> dropped, already covered
                # by chunk0's core.
                _word(" d", 3.0, 4.0),
                _word(" e", 6.0, 7.0),  # absolute 41-42, inside core -> kept
                # absolute 79-80: last chunk, so the usual core upper bound
                # is relaxed and this trailing word is still kept.
                _word(" f", 44.0, 45.0),
            ]
        )
    ]
    model = _FakeTranscribeModel([chunk0_segments, chunk1_segments])

    result = _transcribe_chunked(model, audio=list(range(80)), sampling_rate=1, language="en")

    assert len(model.calls) == 2
    assert model.calls[0]["audio_len"] == 45  # [0, 45)
    assert model.calls[1]["audio_len"] == 45  # [35, 80)
    assert result == [
        {"word": "a", "start": 1.0, "end": 2.0, "line": 0},
        {"word": "b", "start": 39.0, "end": 40.0, "line": 0},
        {"word": "e", "start": 41.0, "end": 42.0, "line": 1},
        {"word": "f", "start": 79.0, "end": 80.0, "line": 1},
    ]


def test_repair_energetic_gaps_recovers_words_dropped_into_a_singing_gap():
    words = [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]
    audio = np.zeros(20)
    audio[1:10] = 0.5  # real energy throughout the whole gap
    model = _FakeTranscribeModel([[_segment([_word(" inside", 4.0, 4.5), _word(" the", 4.5, 4.8)])]])

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "inside", "start": 4.0, "end": 4.5, "line": 8},
        {"word": "the", "start": 4.5, "end": 4.8, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]


def test_repair_energetic_gaps_leaves_a_silent_gap_untouched():
    words = [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]
    audio = np.zeros(20)  # no energy anywhere -- a real instrumental gap
    model = _FakeTranscribeModel([])  # would raise if called -- an untouched gap shouldn't call it

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == words
    assert model.calls == []


def test_repair_energetic_gaps_ignores_short_gaps_even_with_energy():
    words = [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "wont", "start": 2.0, "end": 3.0, "line": 8},  # 1s gap, below the threshold
    ]
    audio = np.full(5, 0.5)
    model = _FakeTranscribeModel([])

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == words
    assert model.calls == []


def test_repair_energetic_gaps_drops_recovered_words_that_just_duplicate_the_anchors():
    words = [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]
    audio = np.zeros(20)
    audio[1:10] = 0.5
    # The isolated re-transcription re-detects both anchor words again
    # (normal -- they're still in the padded clip) alongside one genuinely
    # new word; only the new one should survive.
    model = _FakeTranscribeModel(
        [[_segment([_word(" you", 0.2, 1.1), _word(" inside", 4.0, 4.5), _word(" wont", 9.8, 10.6)])]]
    )

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "inside", "start": 4.0, "end": 4.5, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]


def test_repair_energetic_gaps_treats_case_and_punctuation_variants_as_the_same_anchor():
    # Measured on the real test song: a re-detected anchor came back as
    # "die." and "Keep" instead of the original "die"/"keep" -- just
    # whisper's segment-boundary capitalization/punctuation, not a
    # different word -- and an exact-match dedup let both through as
    # confusing near-duplicate lines.
    words = [
        {"word": "die", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "keep", "start": 10.0, "end": 11.0, "line": 8},
    ]
    audio = np.zeros(20)
    audio[1:10] = 0.5
    model = _FakeTranscribeModel(
        [[_segment([_word(" die.", 0.2, 1.1), _word(" inside", 4.0, 4.5), _word(" Keep", 9.8, 10.6)])]]
    )

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == [
        {"word": "die", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "inside", "start": 4.0, "end": 4.5, "line": 8},
        {"word": "keep", "start": 10.0, "end": 11.0, "line": 8},
    ]


def test_repair_energetic_gaps_rejects_an_implausibly_long_recovered_word():
    # Measured on the real test song: the isolated re-transcription of a
    # gap sometimes hallucinates its own way, collapsing several real words
    # into one absurdly long single "word" span (3.84s, vs. 2.4s being the
    # longest genuinely-sung word measured across two real songs) --
    # inserting that would silently hide the words it actually swallowed.
    words = [
        {"word": "you", "start": 0.0, "end": 1.0, "line": 8},
        {"word": "wont", "start": 10.0, "end": 11.0, "line": 8},
    ]
    audio = np.zeros(20)
    audio[1:10] = 0.5
    model = _FakeTranscribeModel([[_segment([_word(" garbled", 2.0, 6.5)])]])  # 4.5s -- implausible

    result = _repair_energetic_gaps(model, audio=audio, sampling_rate=1, language="en", words=words)

    assert result == words  # left untouched rather than inserting the bad recovery


def _write_synthetic_wav(path, duration_s=2.0, samplerate=16000, freq_hz=220.0):
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, tone.astype(np.float32), samplerate)


def _write_wav_with_silence_then_tone(
    path, silence_s, tone_s, samplerate=16000, freq_hz=220.0, amplitude=0.3
):
    silence = np.zeros(int(silence_s * samplerate), dtype=np.float32)
    t = np.linspace(0, tone_s, int(tone_s * samplerate), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(path, np.concatenate([silence, tone]), samplerate)


def test_align_synced_lyrics_shifts_words_to_match_real_vocal_onset(tmp_path):
    # Reproduces the real bug: lrclib's fetched timestamps assume singing starts at 0.5s, but this
    # video's own vocal stem has no energy until 2.0s -- the lyrics should be re-anchored to the
    # audio's actual onset, not trusted as-is.
    vocal_path = tmp_path / "vocals.wav"
    _write_wav_with_silence_then_tone(vocal_path, silence_s=2.0, tone_s=1.0)
    words = [
        {"word": "early", "start": 0.5, "end": 0.8, "line": 0},
        {"word": "word", "start": 0.8, "end": 1.1, "line": 0},
    ]

    words_result, ranges_result = _align_synced_lyrics_to_audio(words, [(0.5, 0.8)], vocal_path)

    assert words_result[0]["start"] == pytest.approx(1.85, abs=1e-6)
    assert words_result[1]["start"] == pytest.approx(2.15, abs=1e-6)
    # relative spacing between words is preserved -- only the whole set shifts
    assert words_result[1]["start"] - words_result[0]["start"] == pytest.approx(
        words[1]["start"] - words[0]["start"], abs=1e-6
    )
    # background-vocal ranges shift by the same offset as the words
    assert ranges_result[0][0] == pytest.approx(1.85, abs=1e-6)
    assert ranges_result[0][1] == pytest.approx(2.15, abs=1e-6)


def test_align_synced_lyrics_leaves_small_mismatch_untouched(tmp_path):
    vocal_path = tmp_path / "vocals.wav"
    _write_wav_with_silence_then_tone(vocal_path, silence_s=0.0, tone_s=1.0)
    words = [{"word": "hi", "start": 0.1, "end": 0.4, "line": 0}]

    words_result, ranges_result = _align_synced_lyrics_to_audio(words, [], vocal_path)

    assert words_result == words
    assert ranges_result == []


def test_align_synced_lyrics_leaves_an_implausibly_large_offset_untouched(tmp_path):
    # An 18s implied shift is more likely a bad onset detection than a genuine intro mismatch --
    # safer to leave the fetched timestamps alone than risk a wrong correction.
    vocal_path = tmp_path / "vocals.wav"
    _write_wav_with_silence_then_tone(vocal_path, silence_s=18.0, tone_s=1.0)
    words = [{"word": "hi", "start": 0.0, "end": 0.3, "line": 0}]

    words_result, ranges_result = _align_synced_lyrics_to_audio(words, [], vocal_path)

    assert words_result == words
    assert ranges_result == []


def test_align_synced_lyrics_leaves_words_untouched_when_no_vocal_energy_found(tmp_path):
    vocal_path = tmp_path / "silence.wav"
    sf.write(vocal_path, np.zeros(int(2.0 * 16000), dtype=np.float32), 16000)
    words = [{"word": "hi", "start": 0.5, "end": 0.8, "line": 0}]

    words_result, ranges_result = _align_synced_lyrics_to_audio(words, [], vocal_path)

    assert words_result == words
    assert ranges_result == []


def test_fit_time_correction_scales_and_offsets_when_both_anchors_agree():
    # A real ~4% tempo drift across a long song: the last-word anchor implies a scale inside the
    # plausible band, so both anchors should be used rather than falling back to a pure offset.
    scale, offset = _fit_time_correction(
        first_reported=1.7, first_detected=2.0, last_reported=100.0, last_detected=103.0
    )

    assert scale == pytest.approx((103.0 - 2.0) / (100.0 - 1.7), abs=1e-6)
    assert offset == pytest.approx(2.0 - scale * 1.7, abs=1e-6)
    # Sanity: applying the fit to both anchors reproduces the detected times exactly.
    assert scale * 1.7 + offset == pytest.approx(2.0, abs=1e-6)
    assert scale * 100.0 + offset == pytest.approx(103.0, abs=1e-6)


def test_fit_time_correction_falls_back_to_offset_when_last_anchor_missing():
    scale, offset = _fit_time_correction(
        first_reported=1.7, first_detected=2.0, last_reported=100.0, last_detected=None
    )

    assert scale == 1.0
    assert offset == pytest.approx(2.0 - 1.7, abs=1e-6)


def test_fit_time_correction_falls_back_to_offset_when_implied_scale_is_implausible():
    # A last-detected activity implying a >10% tempo difference is more likely a bad end-of-song
    # detection (trailing noise, a mistimed last line) than a real drift -- distrust it.
    scale, offset = _fit_time_correction(
        first_reported=1.0, first_detected=1.0, last_reported=10.0, last_detected=15.0
    )

    assert scale == 1.0
    assert offset == pytest.approx(0.0, abs=1e-6)


def test_fit_time_correction_falls_back_to_offset_when_anchors_are_too_close():
    scale, offset = _fit_time_correction(
        first_reported=1.0, first_detected=1.2, last_reported=1.1, last_detected=1.3
    )

    assert scale == 1.0
    assert offset == pytest.approx(0.2, abs=1e-6)


def test_align_synced_lyrics_corrects_proportional_drift_not_just_offset(tmp_path):
    # Reproduces "the song is faster than the lyrics": a single constant shift can't fix this --
    # the gap between reported and real time has to grow across the song, i.e. a genuine scale
    # mismatch, not just a mistimed intro. Real singing: a burst around 20s (the "true" start) and
    # another around 100s (the "true" end), on a much longer bed of silence than the earlier
    # silence-then-tone tests use, so the two bursts are unambiguous, independent anchors.
    samplerate = 16000
    duration_s = 110.0
    audio = np.zeros(int(duration_s * samplerate), dtype=np.float32)

    def _write_tone(start_s, length_s, freq_hz=220.0, amplitude=0.3):
        t = np.linspace(0, length_s, int(length_s * samplerate), endpoint=False)
        tone = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        start_sample = int(start_s * samplerate)
        audio[start_sample : start_sample + len(tone)] = tone

    _write_tone(20.0, 1.0)
    _write_tone(100.0, 1.0)
    vocal_path = tmp_path / "vocals.wav"
    sf.write(vocal_path, audio, samplerate)

    # Independently confirm what the detection helpers actually find for this audio, rather than
    # hand-predicting the exact RMS-crossing edge (see the sibling silence-then-tone tests for that
    # style) -- this test is about the *fit* using whatever the real detectors report.
    detected_onset = _find_first_vocal_onset(audio, samplerate, search_seconds=30.0)
    detected_last = _find_last_vocal_activity(audio, samplerate, search_start_seconds=85.0)
    assert detected_onset is not None
    assert detected_last is not None

    # Reported (fetched) timestamps run increasingly early relative to the real audio -- a lead-in
    # of 0.6s at the first word growing to a much bigger lead by the last, i.e. the real audio is
    # "faster" than the reported timing.
    first_reported = detected_onset - 0.6
    last_reported = detected_last - 4.0
    words = [
        {"word": "start", "start": first_reported, "end": first_reported + 0.3, "line": 0},
        {"word": "end", "start": last_reported - 0.3, "end": last_reported, "line": 1},
    ]

    words_result, _ranges = _align_synced_lyrics_to_audio(words, [], vocal_path)

    expected_scale, expected_offset = _fit_time_correction(
        first_reported, detected_onset, last_reported, detected_last
    )
    assert expected_scale != 1.0  # confirms this test actually exercises the proportional-fit path
    assert words_result[0]["start"] == pytest.approx(
        expected_scale * first_reported + expected_offset, abs=1e-4
    )
    assert words_result[1]["end"] == pytest.approx(
        expected_scale * last_reported + expected_offset, abs=1e-4
    )
    # The whole point: the correction at the end must differ from the correction at the start,
    # which a constant-offset-only model could never produce.
    start_correction = words_result[0]["start"] - words[0]["start"]
    end_correction = words_result[1]["end"] - words[1]["end"]
    assert start_correction != pytest.approx(end_correction, abs=1e-3)


def test_extract_lyrics_produces_readable_json(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_wav(input_path)

    result = extract_lyrics(input_path, output_dir)

    assert result.lyrics_path.exists()
    assert result.lyrics_path.suffix == ".json"
    words = json.loads(result.lyrics_path.read_text())
    assert isinstance(words, list)
    for word in words:
        assert word.keys() == {"word", "start", "end", "line"}
        assert word["start"] < word["end"]


def test_extract_lyrics_rejects_an_unsupported_language(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    _write_synthetic_wav(input_path)

    with pytest.raises(ValueError, match="Unsupported language"):
        extract_lyrics(input_path, tmp_path / "out", language="fr")


def test_extract_lyrics_uses_online_synced_lyrics_when_a_query_is_given(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_wav(input_path)

    fake_words = [{"word": "Known", "start": 0.0, "end": 1.0, "line": 0}]
    with (
        patch(
            "audio_pipeline.lyrics_extraction.fetch_synced_lyrics",
            return_value=(fake_words, []),
        ) as mock_fetch,
        patch("audio_pipeline.lyrics_extraction.WhisperModel") as mock_model_cls,
    ):
        result = extract_lyrics(input_path, output_dir, lyrics_query="Some Song Title")

    mock_fetch.assert_called_once()
    assert mock_fetch.call_args.args[0] == "Some Song Title"
    mock_model_cls.assert_not_called()  # no need to load Whisper when the lookup succeeds
    assert json.loads(result.lyrics_path.read_text()) == fake_words
    assert result.background_vocal_ranges == []


def test_extract_lyrics_exposes_background_vocal_ranges_from_the_synced_lookup(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_wav(input_path)

    fake_words = [{"word": "Known", "start": 0.0, "end": 1.0, "line": 0}]
    fake_ranges = [(1.0, 2.0)]
    with (
        patch(
            "audio_pipeline.lyrics_extraction.fetch_synced_lyrics",
            return_value=(fake_words, fake_ranges),
        ),
        patch("audio_pipeline.lyrics_extraction.WhisperModel"),
    ):
        result = extract_lyrics(input_path, output_dir, lyrics_query="Some Song Title")

    # The synthetic clip has real energy from t=0, so no onset-alignment shift is expected --
    # the ranges should pass through unchanged, matching the words they came from.
    assert result.background_vocal_ranges == fake_ranges


def test_extract_lyrics_falls_back_to_transcription_when_lookup_finds_nothing(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    _write_synthetic_wav(input_path)

    with patch(
        "audio_pipeline.lyrics_extraction.fetch_synced_lyrics", return_value=None,
    ) as mock_fetch:
        result = extract_lyrics(input_path, output_dir, lyrics_query="Some Unmatched Title")

    mock_fetch.assert_called_once()
    words = json.loads(result.lyrics_path.read_text())
    assert isinstance(words, list)
