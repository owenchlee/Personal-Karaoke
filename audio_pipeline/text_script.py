"""Checks for whether lyric text is written in one of the two scripts this
app actually supports (Latin-script English or Han-script Cantonese).

Not a language classifier -- just Unicode-block membership. Used to catch
wrong-script text from two different failure modes:

- A local Whisper transcription forced to a specific language (``en``/
  ``yue``) can still hallucinate a run of text in a totally unrelated
  script on low-confidence audio (music bleeding through the vocal stem, a
  quiet passage) -- forcing ``language=`` only sets the decoder's initial
  token, it doesn't hard-constrain every token after that.
- An online lrclib.net lookup matches by title text only; a same/similar-
  titled song in a completely different language (e.g. Thai) can be the
  closest-duration match, and that lyrics text would otherwise be used
  verbatim with no per-word filter like Whisper's ever applied to it.
"""
import re

_ALLOWED_TEXT_PATTERN = re.compile(
    r"[\s\x20-\x7E"
    r"À-ɏ"  # accented Latin (Latin-1 Supplement + Latin Extended-A)
    r"‐-⁞"  # common punctuation: dashes, curly quotes, ellipsis, etc.
    r"　-〿"  # CJK punctuation
    r"㐀-䶿"  # CJK Unified Ideographs Extension A
    r"一-鿿"  # CJK Unified Ideographs (Cantonese)
    r"＀-￯"  # fullwidth forms
    r"]"
)


def has_unsupported_script(text: str) -> bool:
    """True if any character in ``text`` falls outside the supported
    scripts -- used to drop a single hallucinated transcription word."""
    return not all(_ALLOWED_TEXT_PATTERN.fullmatch(char) for char in text)


def is_mostly_supported_script(text: str, threshold: float = 0.7) -> bool:
    """True if at least ``threshold`` of ``text``'s non-whitespace
    characters fall inside the supported scripts -- used to validate a
    whole lyrics blob, where a few incidental foreign characters (e.g. an
    accented name) shouldn't cause a false rejection but a predominantly
    wrong-language blob should.
    """
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return True
    supported = sum(1 for char in chars if _ALLOWED_TEXT_PATTERN.fullmatch(char))
    return (supported / len(chars)) >= threshold
