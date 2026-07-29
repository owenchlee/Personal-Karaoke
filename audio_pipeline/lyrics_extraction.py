"""Lyrics acquisition for a song: online lookup first, local transcription
as fallback.

Given an isolated vocal stem, produce word-level timestamped lyrics
(``[{"word", "start", "end", "line"}, ...]``). If a ``lyrics_query`` (e.g.
the song's title) is given, first tries ``lyrics_lookup.fetch_synced_lyrics``
-- known-correct lyrics text from lrclib.net, already time-matched at line
granularity. Falls back to local faster-whisper transcription (CPU-only) of
the vocal stem whenever no online match is found, so the app still works
fully offline / for songs not in that database.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment

from audio_pipeline.lyrics_lookup import fetch_synced_lyrics

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

# This project only supports English and Cantonese songs. Whisper's default
# language detection picks from ~100 languages, and singing (unusual prosody,
# music bleeding through the vocal stem) makes it a much less reliable signal
# than for plain speech -- Cantonese in particular is easily misdetected as
# Mandarin ("zh"), since large-v3 is the first Whisper release to separate
# them at all ("yue"). Restricting the choice to just these two candidates
# avoids the model wandering into an unsupported language entirely.
_SUPPORTED_LANGUAGES = ("en", "yue")


def _detect_language(model: WhisperModel, audio) -> str:
    """Detect whether ``audio`` is sung in English or Cantonese.

    Runs Whisper's normal language detection, then picks whichever of the
    two supported languages it favored -- rather than trusting its raw top
    pick, which could be any of the ~100 languages it knows.
    """
    _, _, all_language_probs = model.detect_language(audio=audio, vad_filter=True)
    probabilities = dict(all_language_probs)
    return max(_SUPPORTED_LANGUAGES, key=lambda lang: probabilities.get(lang, 0.0))


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
            start = round(float(word.start), 4)
            end = round(float(word.end), 4)
            # faster-whisper occasionally hallucinates a run of repeated
            # words that never advance past the current audio frame, so
            # their start/end timestamps collapse to the same value. Real
            # sung words always take non-zero time; treat a zero-length
            # word as a decoding artifact and drop it rather than show a
            # phantom repeated line.
            if end <= start:
                continue
            line_words.append(
                {
                    "word": text,
                    "start": start,
                    "end": end,
                    "line": line,
                }
            )
        if line_words:
            words.extend(line_words)
            line += 1
    return words


# A long song is decoded as a sequence of internal ~30s windows. Measured on
# two real songs: a single whole-song transcribe() call silently lost several
# seconds of real lyrics mid-song -- a run of words hallucinated with
# collapsed/frozen timestamps, later dropped by _flatten_words's zero-duration
# filter -- while transcribing that exact same audio region in isolation
# recovered it perfectly both times (confirmed by directly re-transcribing
# just that region: "inside the pocket of your ripped jeans..." on the test
# song, "my last hand, oh, hope it's worth it..." on a second song, both
# fully recovered with correct timing outside the full-song context).
# Disabling `condition_on_previous_text` alone did not fix it -- verified by
# rerunning the full song with it off and finding the exact same gap. This is
# whisper's long-form hallucination-collapse failure mode: something about
# decoding one very long continuous stretch lets a bad window corrupt
# everything after it, not just context bleed from the previous window's
# text. Splitting into short, independent windows resets that state before it
# can compound.
_CHUNK_SECONDS = 50.0
_CHUNK_OVERLAP_SECONDS = 5.0
_CHUNK_CORE_SECONDS = _CHUNK_SECONDS - 2 * _CHUNK_OVERLAP_SECONDS


def _transcribe_chunked(
    model: WhisperModel,
    audio,
    sampling_rate: int,
    language: str,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Transcribe ``audio`` in independent ~40s windows and stitch the
    results back together, rather than one long pass (see module-level
    comment above ``_CHUNK_SECONDS`` for why).

    Each window is padded with ``_CHUNK_OVERLAP_SECONDS`` of extra audio on
    each side, so a real word/phrase near a cut isn't decoded from a cold
    start with no lead-in -- but only words landing inside the window's own
    untrimmed "core" span are kept; the padding is trusted only from
    whichever neighboring window sees that same audio as its own core. This
    can occasionally split one real sung line across two windows (an early
    line-break in the display), but that's a minor cosmetic cost next to
    silently losing seconds of real lyrics.

    ``on_progress``, if given, is called with a 0-1 fraction of audio
    transcribed so far -- this is what dominates a job's total wall time, so
    it's the main source of a real (not just guessed) overall progress number.
    """
    total_seconds = len(audio) / sampling_rate
    if total_seconds <= _CHUNK_SECONDS:
        segments, _info = model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        words = _flatten_words(list(segments))
        if on_progress:
            on_progress(1.0)
        return words

    words: list[dict] = []
    next_line = -1
    core_start = 0.0
    while core_start < total_seconds:
        core_end = min(core_start + _CHUNK_CORE_SECONDS, total_seconds)
        is_last_chunk = core_end >= total_seconds
        pad_start = max(0.0, core_start - _CHUNK_OVERLAP_SECONDS)
        pad_end = min(total_seconds, core_end + _CHUNK_OVERLAP_SECONDS)

        chunk_audio = audio[int(pad_start * sampling_rate) : int(pad_end * sampling_rate)]
        segments, _info = model.transcribe(
            chunk_audio,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        chunk_words = _flatten_words(list(segments))

        last_chunk_line = None
        for word in chunk_words:
            absolute_start = round(word["start"] + pad_start, 4)
            absolute_end = round(word["end"] + pad_start, 4)
            in_core = absolute_start >= core_start and (is_last_chunk or absolute_start < core_end)
            if not in_core:
                continue
            if word["line"] != last_chunk_line:
                next_line += 1
                last_chunk_line = word["line"]
            words.append({"word": word["word"], "start": absolute_start, "end": absolute_end, "line": next_line})

        core_start = core_end
        if on_progress:
            on_progress(min(core_start / total_seconds, 1.0))

    return words


# Chunking (above) sharply reduces the hallucination-collapse dropout but
# doesn't eliminate it -- measured directly: even after chunking, a real
# song still had a 13.6s gap with clear singing energy underneath it,
# entirely inside a single ~40s core window, not at a chunk boundary. This
# is a last-resort repair pass: find gaps that still have real singing
# energy in the vocal stem underneath them (the same signal
# melody_extraction.py's own silence gate uses) and recover them the same
# way manual debugging did every time it was tried -- re-transcribe just
# that isolated stretch on its own, fresh context, and splice in whatever
# it finds.
_GAP_REPAIR_THRESHOLD_SECONDS = 2.5
_GAP_REPAIR_PAD_SECONDS = 3.0
_GAP_SINGING_RMS_GATE = 0.01  # same threshold family as melody_extraction._SILENCE_RMS_GATE

# The longest genuinely-sung word measured across both real test songs was
# 2.4s (a held note). A "recovered" word far past that is itself another
# hallucination artifact -- e.g. one seen in practice: a single word given a
# 3.84s span that had clearly swallowed several real words without
# transcribing them. Better to leave the surrounding gap as-is than insert
# text that looks plausible but silently drops what's actually sung inside it.
_MAX_PLAUSIBLE_WORD_DURATION_SECONDS = 3.0


def _rms(audio, sampling_rate: int, start_s: float, end_s: float) -> float:
    start_sample = max(0, int(start_s * sampling_rate))
    end_sample = min(len(audio), int(end_s * sampling_rate))
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(segment, dtype=np.float64) ** 2)))


def _normalize_word(text: str) -> str:
    """Lowercased, trailing-punctuation-stripped form used only to compare
    two transcriptions of "the same word" -- e.g. a re-detected anchor can
    come back as "Keep" or "die." instead of the original "keep"/"die",
    since segment-initial capitalization and end-of-phrase punctuation are
    just artifacts of where whisper happened to split a segment this time,
    not a different word.
    """
    return text.lower().strip(".,!?;:\"'")


def _repair_energetic_gaps(
    model: WhisperModel, audio, sampling_rate: int, language: str, words: list[dict]
) -> list[dict]:
    """Recover words dropped into gaps that still have real singing energy.

    Only re-transcribes gaps above ``_GAP_REPAIR_THRESHOLD_SECONDS`` whose
    vocal-stem RMS shows real energy, not silence -- a gap that's actually
    an instrumental break is left alone. Recovered words are assigned the
    surrounding gap's own line: both known real examples of this bug had
    the dropout happen in the middle of a single sung line, not across a
    line break.
    """
    if len(words) < 2:
        return words

    total_seconds = len(audio) / sampling_rate
    repaired = [words[0]]
    for prev_word, word in zip(words, words[1:]):
        gap_start, gap_end = prev_word["end"], word["start"]
        if gap_end - gap_start > _GAP_REPAIR_THRESHOLD_SECONDS and _rms(
            audio, sampling_rate, gap_start, gap_end
        ) >= _GAP_SINGING_RMS_GATE:
            pad_start = max(0.0, gap_start - _GAP_REPAIR_PAD_SECONDS)
            pad_end = min(total_seconds, gap_end + _GAP_REPAIR_PAD_SECONDS)
            clip = audio[int(pad_start * sampling_rate) : int(pad_end * sampling_rate)]
            segments, _info = model.transcribe(
                clip, language=language, word_timestamps=True, vad_filter=True,
                condition_on_previous_text=False,
            )
            for recovered in _flatten_words(list(segments)):
                absolute_start = round(recovered["start"] + pad_start, 4)
                absolute_end = round(recovered["end"] + pad_start, 4)
                if absolute_start < gap_start - 0.5 or absolute_start > gap_end + 0.5:
                    continue  # outside the gap -- re-detected padding context, not new
                if absolute_end - absolute_start > _MAX_PLAUSIBLE_WORD_DURATION_SECONDS:
                    continue  # another hallucination artifact, not a real recovered word
                recovered_norm = _normalize_word(recovered["word"])
                duplicates_prev = (
                    recovered_norm == _normalize_word(prev_word["word"])
                    and abs(absolute_start - prev_word["start"]) < 1.0
                )
                duplicates_next = (
                    recovered_norm == _normalize_word(word["word"])
                    and abs(absolute_start - word["start"]) < 1.0
                )
                if duplicates_prev or duplicates_next:
                    continue  # the anchor word itself, re-detected -- already in the list
                repaired.append(
                    {"word": recovered["word"], "start": absolute_start, "end": absolute_end, "line": prev_word["line"]}
                )
        repaired.append(word)
    return repaired


def extract_lyrics(
    vocal_stem_path: str | Path,
    output_dir: str | Path,
    language: str | None = None,
    lyrics_query: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> LyricsResult:
    """Produce word-level lyrics JSON (``[{"word", "start", "end", "line"},
    ...]``) for ``vocal_stem_path``, saved inside ``output_dir``.

    If ``lyrics_query`` (e.g. the song's title) is given, first tries an
    online synced-lyrics lookup; only transcribes locally when that finds
    no match. ``language`` picks the transcription language explicitly
    ("en" or "yue"/Cantonese) when local transcription is needed --
    otherwise it's auto-detected from the audio, restricted to those same
    two languages (see ``_detect_language``).

    ``on_progress``, if given, is called with a 0-1 fraction of this stage's
    own progress -- 1.0 right away for the fast online-lookup path, or
    tracking chunk-by-chunk transcription progress (reserving the last 5%
    for the gap-repair pass) when it falls back to local transcription.

    Returns a ``LyricsResult`` with the path to the saved file.
    """
    if language is not None and language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}; expected one of {_SUPPORTED_LANGUAGES}"
        )

    vocal_stem_path = Path(vocal_stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words = None
    if lyrics_query:
        duration_seconds = sf.info(str(vocal_stem_path)).duration
        words = fetch_synced_lyrics(lyrics_query, duration_seconds=duration_seconds)
        if words is not None and on_progress:
            on_progress(1.0)

    if words is None:
        model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
        audio = decode_audio(
            str(vocal_stem_path), sampling_rate=model.feature_extractor.sampling_rate
        )
        if language is None:
            language = _detect_language(model, audio)
        sampling_rate = model.feature_extractor.sampling_rate
        transcription_progress = (
            (lambda fraction: on_progress(fraction * 0.95)) if on_progress else None
        )
        words = _transcribe_chunked(
            model, audio, sampling_rate, language, on_progress=transcription_progress
        )
        words = _repair_energetic_gaps(model, audio, sampling_rate, language, words)
        if on_progress:
            on_progress(1.0)

    lyrics_path = output_dir / f"{vocal_stem_path.stem}_lyrics.json"
    lyrics_path.write_text(json.dumps(words, indent=2))

    return LyricsResult(lyrics_path=lyrics_path)
