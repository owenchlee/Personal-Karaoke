"""Vocal lyrics transcription via faster-whisper, CPU-only.

Given an isolated vocal stem, transcribe the sung lyrics with word-level
timestamps. This is a local transcript of audio the caller already has
(extracted from their own video, isolated by separation.py) -- not lyrics
sourced from any external database -- consistent with this project's fully
local, per-song offline processing.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

# Benchmarked "small", "medium", and "large-v3" against the real test song
# after a user-reported wrong first word ("Love and can hurt..." -- not
# grammatical). "small" and "medium" agreed on that same wrong wording;
# "large-v3" corrected it to "Loving can hurt..." (grammatical, and a much
# more plausible reading of what's actually sung). Costs real time (380s vs.
# 143s for medium on the 274s test song, CPU int8) but this is one-time
# offline per-song processing, and the accuracy difference was decisive, not
# marginal -- worth the wait. vad_filter=True suppresses hallucinated words
# in silent/instrumental sections, matching the same intent as the melody
# pipeline's own silence gate.
_MODEL_SIZE = "large-v3"


@dataclass
class LyricsResult:
    lyrics_path: Path


def _flatten_words(segments: list[Segment]) -> list[dict]:
    """Flatten basic-pitch-style segments into a single word list, tagging
    each word with a ``line`` index (its source segment's position among
    segments that actually produced words) so the frontend can group words
    back into displayable lines without re-deriving segmentation itself.
    """
    words = []
    line = 0
    for segment in segments:
        if not segment.words:
            continue
        line_words = []
        for word in segment.words:
            text = word.word.strip()
            if not text:
                continue
            line_words.append(
                {
                    "word": text,
                    "start": round(float(word.start), 4),
                    "end": round(float(word.end), 4),
                    "line": line,
                }
            )
        if line_words:
            words.extend(line_words)
            line += 1
    return words


def extract_lyrics(vocal_stem_path: str | Path, output_dir: str | Path) -> LyricsResult:
    """Transcribe ``vocal_stem_path`` with faster-whisper and save a
    word-level lyrics JSON (``[{"word", "start", "end", "line"}, ...]``)
    inside ``output_dir``.

    Returns a ``LyricsResult`` with the path to the saved file.
    """
    vocal_stem_path = Path(vocal_stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(vocal_stem_path), word_timestamps=True, vad_filter=True
    )

    words = _flatten_words(list(segments))

    lyrics_path = output_dir / f"{vocal_stem_path.stem}_lyrics.json"
    lyrics_path.write_text(json.dumps(words, indent=2))

    return LyricsResult(lyrics_path=lyrics_path)
