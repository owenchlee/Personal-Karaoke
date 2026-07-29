"""Tests for the shared Latin/Han script-check used to catch wrong-language
lyrics from both the Whisper transcription and lrclib lookup paths."""
from audio_pipeline.text_script import has_unsupported_script, is_mostly_supported_script

_THAI_HELLO = "สวัสดี"
_CANTONESE_HELLO = "你好"


def test_has_unsupported_script_accepts_plain_english():
    assert has_unsupported_script("hello world") is False


def test_has_unsupported_script_accepts_contractions_and_punctuation():
    assert has_unsupported_script("don't stop, believin'!") is False


def test_has_unsupported_script_accepts_cantonese():
    assert has_unsupported_script(_CANTONESE_HELLO) is False


def test_has_unsupported_script_accepts_mixed_english_and_cantonese():
    assert has_unsupported_script("mixed" + _CANTONESE_HELLO + "word") is False


def test_has_unsupported_script_rejects_thai():
    assert has_unsupported_script(_THAI_HELLO) is True


def test_is_mostly_supported_script_accepts_pure_english():
    assert is_mostly_supported_script("the only thing that I know how to do") is True


def test_is_mostly_supported_script_accepts_pure_cantonese():
    assert is_mostly_supported_script(_CANTONESE_HELLO * 10) is True


def test_is_mostly_supported_script_accepts_cantonese_with_english_code_switching():
    # Real Cantopop lyrics commonly mix in English words/phrases -- this
    # should not be penalized as "wrong script".
    assert is_mostly_supported_script(_CANTONESE_HELLO * 5 + " baby " + _CANTONESE_HELLO * 5) is True


def test_is_mostly_supported_script_rejects_predominantly_thai_text():
    assert is_mostly_supported_script(_THAI_HELLO * 10) is False


def test_is_mostly_supported_script_tolerates_a_few_incidental_foreign_characters():
    # A handful of stray foreign characters (e.g. from an artist name in a
    # metadata line) shouldn't sink an otherwise-valid match.
    mostly_english = "a" * 95 + _THAI_HELLO[:5]

    assert is_mostly_supported_script(mostly_english) is True


def test_is_mostly_supported_script_treats_empty_text_as_supported():
    assert is_mostly_supported_script("") is True
    assert is_mostly_supported_script("   ") is True
