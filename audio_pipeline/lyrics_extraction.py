"""Lyrics acquisition for a song: online lookup first, local transcription
as fallback.

Given an isolated vocal stem, produce word-level timestamped lyrics
(``[{"word", "start", "end", "line"}, ...]``). If a ``lyrics_query`` (e.g.
the song's title) is given, first tries ``lyrics_lookup.fetch_synced_lyrics``
-- known-correct lyrics text from lrclib.net, already time-matched at line
granularity. Falls back to local faster-whisper transcription of the vocal
stem whenever no online match is found, so the app still works fully offline
/ for songs not in that database. Runs on GPU when one is available (see
audio_pipeline.device), CPU otherwise.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment

from audio_pipeline.device import get_device
from audio_pipeline.forced_alignment import align_tokens
from audio_pipeline.lyrics_lookup import fetch_synced_lyrics
from audio_pipeline.text_script import has_unsupported_script

# Benchmarked "small", "medium", and "large-v3" against the real test song
# after a user-reported wrong first word ("Love and can hurt..." -- not
# grammatical). "small" and "medium" agreed on that same wrong wording;
# "large-v3" corrected it to "Loving can hurt..." (grammatical, and a much
# more plausible reading of what's actually sung). Costs real time (380s vs.
# 143s for medium on the 274s test song, CPU int8 -- GPU is substantially
# faster for the same model) but this is one-time offline per-song
# processing, and the accuracy difference was decisive, not marginal --
# worth the wait. vad_filter=True suppresses hallucinated words in
# silent/instrumental sections, matching the same intent as the melody
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

    Measured directly on a real Cantonese song: the model's own "yue" token
    scored effectively zero (didn't even appear in the returned
    probabilities) while generic "zh" scored 0.64 -- exactly the mislabeling
    the module comment above already warned about -- leaving "en" (a mere
    0.06) to "win" a naive comparison of just the two supported labels. Fold
    "zh" into "yue"'s candidate score before comparing, so a strong Chinese-
    family signal isn't lost just because the model didn't specifically use
    the "yue" token for it.
    """
    _, _, all_language_probs = model.detect_language(audio=audio, vad_filter=True)
    probabilities = dict(all_language_probs)
    yue_score = max(probabilities.get("yue", 0.0), probabilities.get("zh", 0.0))
    en_score = probabilities.get("en", 0.0)
    return "yue" if yue_score >= en_score else "en"


@dataclass
class LyricsResult:
    lyrics_path: Path
    # (start, end) spans that were pure backing/ad-lib vocals, stripped out of the lyrics text by
    # lyrics_lookup._strip_background_vocals -- only ever populated on the online synced-lyrics
    # path (local transcription has no such markup to detect). Callers use these to also drop the
    # corresponding note-highway notes, since that's not the line the player is meant to sing.
    background_vocal_ranges: list[tuple[float, float]] = field(default_factory=list)


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
            if has_unsupported_script(text):
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


# lrclib only gives line-level timestamps; word timing within a line is a linear character-
# weighted guess (see lyrics_lookup._distribute_words), and even the line timestamps themselves
# belong to whatever release lrclib matched by title, not necessarily this specific video (a
# different intro length, edit, or mix speed). An earlier approach tried to *correct* those
# reported timestamps by fitting an intro-offset + proportional-drift model against the vocal
# stem's own onset/last-activity energy -- workable for a pure linear mismatch, but it could only
# ever approximate a different release's timing, not measure this one. Force-aligning the same
# (known-correct) text directly against this audio via `forced_alignment.align_tokens` measures it
# instead, so this replaces that correction step entirely rather than adjusting it further.
_CJK_RE = re.compile("[一-鿿]")  # same range as lyrics_lookup._CJK_RE


def _language_of_words(words: list[dict]) -> str:
    """Cantonese lyrics are entirely CJK-script tokens (`lyrics_lookup._tokenize_line` splits them
    character-by-character); a single CJK character anywhere is enough to tell them apart from
    English, since a song's lyrics are never a mix of the two supported languages.
    """
    return "yue" if any(_CJK_RE.search(word["word"]) for word in words) else "en"


def _interpolate_background_range(
    old_start: float, old_end: float, old_words: list[dict], new_words: list[dict],
) -> tuple[float, float]:
    """Re-time a background-vocal range (originally in lrclib's own, now-distrusted time
    coordinates -- see the module comment above ``_CJK_RE``) using the *forced-aligned* neighbors
    immediately before/after it in the original word order, rather than trusting its raw lrclib
    timestamp. ``old_words``/``new_words`` are the same list before/after alignment, so a position
    match between them carries over directly -- no re-detection needed.
    """
    prev_end = None
    next_start = None
    for old_word, new_word in zip(old_words, new_words):
        if old_word["end"] <= old_start:
            prev_end = new_word["end"]
        if next_start is None and old_word["start"] >= old_end:
            next_start = new_word["start"]

    if prev_end is not None and next_start is not None:
        return prev_end, next_start
    span = old_end - old_start
    if prev_end is not None:
        return prev_end, prev_end + span
    if next_start is not None:
        return max(0.0, next_start - span), next_start
    return old_start, old_end  # no real-word anchor on either side -- nothing better to fall back to


def _force_align_synced_lyrics(
    words: list[dict],
    background_ranges: list[tuple[float, float]],
    vocal_stem_path: Path,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Replace lrclib-fetched ``words``' guessed timing with real per-word timestamps measured
    directly against this audio (see the module comment above ``_CJK_RE``), and re-time
    ``background_ranges`` (see ``lyrics_lookup.fetch_synced_lyrics``) from the newly-aligned
    neighboring words rather than their own now-distrusted raw timestamps.
    """
    if not words:
        return words, background_ranges

    tokens = [word["word"] for word in words]
    aligned = align_tokens(tokens, vocal_stem_path, _language_of_words(words))
    new_words = [{**word, **timing} for word, timing in zip(words, aligned)]
    new_ranges = [
        _interpolate_background_range(start, end, words, new_words)
        for start, end in background_ranges
    ]
    return new_words, new_ranges


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


def _find_energetic_gaps(
    audio, sampling_rate: int, words: list[dict], background_ranges: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Same gap-detection condition as `_repair_energetic_gaps` (long enough, and real singing
    energy in the vocal stem, not silence) but without needing a model -- lets a caller decide
    whether loading Whisper to repair anything is worth it before paying that cost. Skips any gap
    that overlaps a `background_ranges` span, since those are deliberately-stripped backing/ad-lib
    vocals, not something to recover.
    """
    if len(words) < 2:
        return []
    gaps = []
    for prev_word, word in zip(words, words[1:]):
        gap_start, gap_end = prev_word["end"], word["start"]
        if gap_end - gap_start <= _GAP_REPAIR_THRESHOLD_SECONDS:
            continue
        if any(gap_start < end and gap_end > start for start, end in background_ranges):
            continue
        if _rms(audio, sampling_rate, gap_start, gap_end) >= _GAP_SINGING_RMS_GATE:
            gaps.append((gap_start, gap_end))
    return gaps


# lrclib's line-level text has no defense at all against a real sung passage this specific video
# has that the fetched LRC just doesn't cover (e.g. an ad-lib/interlude the community transcription
# skipped) -- unlike the local-transcription path, which already recovers exactly this case via
# `_repair_energetic_gaps`. Measured directly on a real cached song
# (`priscilla-chan-night-flight-english-yale-romanization`): a 25.76s stretch had vocal-stem RMS
# (0.01-0.05) comparable to the song's own median non-silent RMS (~0.07) with zero lrclib coverage --
# the note highway (independently derived from the same vocal stem) correctly showed notes there the
# whole time, while the lyrics display had nothing to show, reading as frozen/stuck. Reuse the same
# repair here rather than leaving lrclib-sourced lyrics with no equivalent defense.
def _repair_synced_lyrics_gaps(
    words: list[dict],
    background_ranges: list[tuple[float, float]],
    vocal_stem_path: Path,
    language: str | None,
) -> list[dict]:
    """Fills real-singing gaps in lrclib-sourced ``words`` the same way ``_repair_energetic_gaps``
    already does for local transcription. Only loads Whisper at all if a cheap, model-free RMS scan
    (`_find_energetic_gaps`) finds at least one gap worth repairing, so a song whose lrclib lyrics
    already cover the whole song -- the common case -- stays on the fast lookup-only path.
    """
    if not words:
        return words

    audio, sampling_rate = sf.read(vocal_stem_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not _find_energetic_gaps(audio, sampling_rate, words, background_ranges):
        return words

    device = get_device()
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(_MODEL_SIZE, device=device, compute_type=compute_type)
    model_audio = decode_audio(
        str(vocal_stem_path), sampling_rate=model.feature_extractor.sampling_rate
    )
    resolved_language = language or _detect_language(model, model_audio)
    return _repair_energetic_gaps(
        model, model_audio, model.feature_extractor.sampling_rate, resolved_language, words
    )


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

    Returns a ``LyricsResult`` with the path to the saved file and any backing/ad-lib-vocal time
    ranges stripped out of it (see ``LyricsResult.background_vocal_ranges``).
    """
    if language is not None and language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}; expected one of {_SUPPORTED_LANGUAGES}"
        )

    vocal_stem_path = Path(vocal_stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words = None
    background_ranges: list[tuple[float, float]] = []
    if lyrics_query:
        duration_seconds = sf.info(str(vocal_stem_path)).duration
        fetched = fetch_synced_lyrics(lyrics_query, duration_seconds=duration_seconds)
        if fetched is not None:
            words, background_ranges = fetched
            words, background_ranges = _force_align_synced_lyrics(
                words, background_ranges, vocal_stem_path
            )
            words = _repair_synced_lyrics_gaps(words, background_ranges, vocal_stem_path, language)
            if on_progress:
                on_progress(1.0)

    if words is None:
        device = get_device()
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(_MODEL_SIZE, device=device, compute_type=compute_type)
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

    return LyricsResult(lyrics_path=lyrics_path, background_vocal_ranges=background_ranges)
