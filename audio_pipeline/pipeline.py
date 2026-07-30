"""End-to-end song processing: video -> audio -> stems -> melody + lyrics, with
caching.

Orchestrates video_extraction, separation, melody_extraction, and
lyrics_extraction into a single call, keyed by a per-song cache directory so a
song already processed is never reprocessed unnecessarily.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from audio_pipeline.lyrics_extraction import extract_lyrics
from audio_pipeline.melody_extraction import extract_melody
from audio_pipeline.separation import separate_stems
from audio_pipeline.video_extraction import extract_audio

_INSTRUMENTAL_FILENAME = "instrumental.wav"
_VOCALS_FILENAME = "vocals.wav"
_MIDI_FILENAME = "melody.mid"
_NOTES_FILENAME = "notes.json"
_LYRICS_FILENAME = "lyrics.json"
_META_FILENAME = "meta.json"


@dataclass
class SongAssets:
    instrumental_path: Path
    vocals_path: Path
    midi_path: Path
    notes_path: Path
    lyrics_path: Path


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "song"


def _cached_assets(song_cache_dir: Path) -> SongAssets | None:
    instrumental_path = song_cache_dir / _INSTRUMENTAL_FILENAME
    vocals_path = song_cache_dir / _VOCALS_FILENAME
    midi_path = song_cache_dir / _MIDI_FILENAME
    notes_path = song_cache_dir / _NOTES_FILENAME
    lyrics_path = song_cache_dir / _LYRICS_FILENAME

    if instrumental_path.exists() and notes_path.exists() and lyrics_path.exists():
        return SongAssets(
            instrumental_path=instrumental_path,
            vocals_path=vocals_path,
            midi_path=midi_path,
            notes_path=notes_path,
            lyrics_path=lyrics_path,
        )
    return None


# Notes right at a lyric line's boundary (e.g. a held note's tail ending a fraction of a
# second after the transcribed word's own reported end) are real continuations of that line,
# not humming -- measured directly on a real cached song
# (`priscilla-chan-night-flight-english-yale-romanization`): a note at 48.26-48.42s shares the
# same MIDI pitch (66) as the note ending exactly at the line's own reported end (48.18s).
# Padding each line's span before checking overlap keeps these while still dropping genuine
# no-lyrics stretches, which in that same song started/ended tens of seconds from any line
# (a hummed intro before the first line, and a hummed outro after the last). 0.5s matches the
# existing `_OFFSET_EXTEND_MAX_SECONDS` / gap-repair margin precedent already used elsewhere in
# this codebase for the identical "is this still the same utterance" judgment.
_LYRIC_COVERAGE_PADDING_SECONDS = 0.5


def _remove_notes_without_lyrics(notes_path: Path, lyrics_path: Path) -> None:
    """Drop note events that don't overlap any displayed lyric line (padded by
    ``_LYRIC_COVERAGE_PADDING_SECONDS`` on each side -- see the comment above that constant), so
    the note highway never shows a note for a stretch the player has no lyric to sing, e.g. a
    hummed intro/outro with no words at all.
    """
    words = json.loads(lyrics_path.read_text())
    notes = json.loads(notes_path.read_text())

    if not words:
        notes_path.write_text(json.dumps([], indent=2))
        return

    lines: dict[int, list[dict]] = {}
    for word in words:
        lines.setdefault(word["line"], []).append(word)
    spans = [
        (min(w["start"] for w in line_words), max(w["end"] for w in line_words))
        for line_words in lines.values()
    ]

    pad = _LYRIC_COVERAGE_PADDING_SECONDS
    filtered = [
        note
        for note in notes
        if any(note["onset"] < end + pad and note["offset"] > start - pad for start, end in spans)
    ]
    notes_path.write_text(json.dumps(filtered, indent=2))


def _remove_background_vocal_notes(notes_path: Path, background_ranges: list[tuple[float, float]]) -> None:
    """Drop note events that fall inside a backing/ad-lib-vocal lyric range (see
    ``lyrics_extraction.LyricsResult.background_vocal_ranges``), so the note highway only shows
    notes for the line the player is actually meant to sing.
    """
    notes = json.loads(notes_path.read_text())
    filtered = [
        note
        for note in notes
        if not any(note["onset"] < end and note["offset"] > start for start, end in background_ranges)
    ]
    notes_path.write_text(json.dumps(filtered, indent=2))


def is_cached(cache_dir: str | Path, song_id: str) -> bool:
    """Whether ``song_id`` already has a complete cached result in
    ``cache_dir`` -- lets a caller (e.g. the job server) skip expensive
    upstream work (like downloading) when the result already exists,
    without duplicating ``_cached_assets``'s file-presence check.
    """
    return _cached_assets(Path(cache_dir) / song_id) is not None


def _extract_and_filter(
    vocals_path: Path,
    song_cache_dir: Path,
    on_progress: Callable[[str, float], None] | None,
    language: str | None,
    lyrics_query: str | None,
):
    """Run melody extraction and lyrics acquisition concurrently against an already-isolated vocal
    stem, then apply the note-highway filters that depend on the resulting lyrics
    (``_remove_notes_without_lyrics``/``_remove_background_vocal_notes`` above). Shared by
    ``process_song`` (fresh from a video) and ``reprocess_melody_and_lyrics`` (from an
    already-cached vocal stem).

    extract_melody and extract_lyrics both only depend on vocals_path, not on each other, and
    lyrics transcription (large-v3, CPU) dominates wall time -- run them concurrently so melody
    extraction finishes inside that window instead of adding to it. basic-pitch has no progress
    hook to report sub-progress from, so "extracting_melody" just reports 0.0 -- the concurrent
    "transcribing_lyrics" fraction is what actually drives the progress bar for this whole window,
    since it's the longer-running of the two.
    """
    def report(stage: str, fraction: float = 0.0) -> None:
        if on_progress:
            on_progress(stage, fraction)

    report("extracting_melody")
    report("transcribing_lyrics")
    with ThreadPoolExecutor(max_workers=2) as executor:
        melody_future = executor.submit(extract_melody, vocals_path, song_cache_dir)
        lyrics_future = executor.submit(
            extract_lyrics, vocals_path, song_cache_dir,
            language=language, lyrics_query=lyrics_query,
            on_progress=(
                (lambda fraction: report("transcribing_lyrics", fraction)) if on_progress else None
            ),
        )
        melody = melody_future.result()
        lyrics = lyrics_future.result()

    _remove_notes_without_lyrics(melody.notes_path, lyrics.lyrics_path)
    if lyrics.background_vocal_ranges:
        _remove_background_vocal_notes(melody.notes_path, lyrics.background_vocal_ranges)

    return melody, lyrics


def process_song(
    video_path: str | Path,
    cache_dir: str | Path = Path("cache"),
    song_id: str | None = None,
    force: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    language: str | None = None,
    lyrics_query: str | None = None,
    separation_model: str = "htdemucs",
) -> SongAssets:
    """Process ``video_path`` end-to-end into cached instrumental audio and a
    note-event JSON, skipping reprocessing if a cached result already exists
    for this song (unless ``force`` is set).

    ``on_progress``, if given, is called with a stage name ("separating",
    "extracting_melody", "transcribing_lyrics") and a 0-1 fraction of that
    stage's own progress, so a caller (e.g. a job server) can report real
    progress rather than just which stage is active.

    ``language`` ("en" or "yue") and ``lyrics_query`` (e.g. the song's title,
    used for an online synced-lyrics lookup before falling back to local
    transcription) are passed straight through to ``extract_lyrics`` -- see
    there for details.

    ``separation_model`` ("htdemucs" or "htdemucs_ft") is passed straight
    through to ``separate_stems`` and recorded in ``meta.json`` so a later
    caller can tell what a cached song was actually separated with.
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    slug = slugify(song_id if song_id is not None else video_path.stem)
    song_cache_dir = cache_dir / slug

    if not force:
        cached = _cached_assets(song_cache_dir)
        if cached is not None:
            return cached

    song_cache_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, fraction: float = 0.0) -> None:
        if on_progress:
            on_progress(stage, fraction)

    report("separating")
    extracted_wav = extract_audio(video_path, song_cache_dir)
    vocals_path, instrumental_path = separate_stems(
        extracted_wav, song_cache_dir, model=separation_model,
        on_progress=(lambda fraction: report("separating", fraction)) if on_progress else None,
    )

    melody, lyrics = _extract_and_filter(
        vocals_path, song_cache_dir, on_progress, language, lyrics_query
    )
    extracted_wav.unlink(missing_ok=True)

    final_instrumental_path = song_cache_dir / _INSTRUMENTAL_FILENAME
    final_vocals_path = song_cache_dir / _VOCALS_FILENAME
    final_midi_path = song_cache_dir / _MIDI_FILENAME
    final_notes_path = song_cache_dir / _NOTES_FILENAME
    final_lyrics_path = song_cache_dir / _LYRICS_FILENAME

    instrumental_path.replace(final_instrumental_path)
    vocals_path.replace(final_vocals_path)
    melody.midi_path.replace(final_midi_path)
    melody.notes_path.replace(final_notes_path)
    lyrics.lyrics_path.replace(final_lyrics_path)

    meta = {
        "source_file": str(video_path),
        "song_id": slug,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "separation_model": separation_model,
    }
    (song_cache_dir / _META_FILENAME).write_text(json.dumps(meta, indent=2))

    return SongAssets(
        instrumental_path=final_instrumental_path,
        vocals_path=final_vocals_path,
        midi_path=final_midi_path,
        notes_path=final_notes_path,
        lyrics_path=final_lyrics_path,
    )


def reprocess_melody_and_lyrics(
    cache_dir: str | Path,
    song_id: str,
    on_progress: Callable[[str, float], None] | None = None,
    language: str | None = None,
    lyrics_query: str | None = None,
) -> SongAssets:
    """Re-run melody extraction and lyrics acquisition for an already-cached song, from its
    existing vocal stem -- no re-download, no re-running Demucs separation (``instrumental.wav``/
    ``vocals.wav`` are left untouched). For picking up a melody/lyrics-only pipeline improvement
    (e.g. the CTC-forced-alignment change, see NOTES.md's "Lyrics timing" entry) on songs whose
    original source video is no longer available to run a full ``process_song`` on.

    A fresh ``extract_melody`` run is required here, not just re-filtering the existing cached
    ``notes.json`` against new lyrics: the previous ``_remove_notes_without_lyrics`` pass already
    permanently dropped notes that didn't overlap the *old* (less accurate) lyric spans, and that
    can't be recovered by filtering an already-filtered file -- only a fresh melody extraction has
    the full, unfiltered note set to filter correctly against the new lyrics.

    Raises ``FileNotFoundError`` if the song has no cached vocal/instrumental stems.
    """
    song_cache_dir = Path(cache_dir) / song_id
    vocals_path = song_cache_dir / _VOCALS_FILENAME
    instrumental_path = song_cache_dir / _INSTRUMENTAL_FILENAME
    if not vocals_path.exists() or not instrumental_path.exists():
        raise FileNotFoundError(f"No cached vocal/instrumental stems found under {song_cache_dir}")

    melody, lyrics = _extract_and_filter(
        vocals_path, song_cache_dir, on_progress, language, lyrics_query
    )

    final_midi_path = song_cache_dir / _MIDI_FILENAME
    final_notes_path = song_cache_dir / _NOTES_FILENAME
    final_lyrics_path = song_cache_dir / _LYRICS_FILENAME
    melody.midi_path.replace(final_midi_path)
    melody.notes_path.replace(final_notes_path)
    lyrics.lyrics_path.replace(final_lyrics_path)

    meta_path = song_cache_dir / _META_FILENAME
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"song_id": song_id}
    meta["processed_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))

    return SongAssets(
        instrumental_path=instrumental_path,
        vocals_path=vocals_path,
        midi_path=final_midi_path,
        notes_path=final_notes_path,
        lyrics_path=final_lyrics_path,
    )
