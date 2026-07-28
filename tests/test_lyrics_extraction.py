"""Tests for the faster-whisper lyrics transcription wrapper.

The pure `_flatten_words` reshaping logic is tested directly against
constructed faster-whisper types (no model needed). The end-to-end smoke
test uses a synthetic (non-speech) clip -- this only proves the pipeline
runs and produces valid JSON, not that transcription is accurate against
real singing (that's a manual check, see NOTES.md).
"""
import json

import numpy as np
import soundfile as sf
from faster_whisper.transcribe import Segment, Word

from audio_pipeline.lyrics_extraction import _flatten_words, extract_lyrics


def _word(word, start, end, probability=0.9):
    return Word(start=start, end=end, word=word, probability=probability)


def _segment(words):
    return Segment(
        id=0, seek=0, start=words[0].start, end=words[-1].end,
        text="".join(w.word for w in words), tokens=[], avg_logprob=0.0,
        compression_ratio=0.0, no_speech_prob=0.0, words=words, temperature=0.0,
    )


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


def test_extract_lyrics_produces_readable_json(tmp_path):
    input_path = tmp_path / "synthetic_vocals.wav"
    output_dir = tmp_path / "out"
    duration_s, samplerate, freq_hz = 2.0, 16000, 220.0
    t = np.linspace(0, duration_s, int(duration_s * samplerate), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(input_path, tone.astype(np.float32), samplerate)

    result = extract_lyrics(input_path, output_dir)

    assert result.lyrics_path.exists()
    assert result.lyrics_path.suffix == ".json"
    words = json.loads(result.lyrics_path.read_text())
    assert isinstance(words, list)
    for word in words:
        assert word.keys() == {"word", "start", "end", "line"}
        assert word["start"] < word["end"]
