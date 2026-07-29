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


def process_song(
    video_path: str | Path,
    cache_dir: str | Path = Path("cache"),
    song_id: str | None = None,
    force: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    language: str | None = None,
    lyrics_query: str | None = None,
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
        extracted_wav, song_cache_dir,
        on_progress=(lambda fraction: report("separating", fraction)) if on_progress else None,
    )

    # extract_melody and extract_lyrics both only depend on vocals_path, not on
    # each other, and lyrics transcription (large-v3, CPU) dominates wall time --
    # run them concurrently so melody extraction finishes inside that window
    # instead of adding to it. basic-pitch has no progress hook to report
    # sub-progress from, so "extracting_melody" just reports 0.0 -- the
    # concurrent "transcribing_lyrics" fraction is what actually drives the
    # progress bar for this whole window, since it's the longer-running of
    # the two.
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
    extracted_wav.unlink(missing_ok=True)

    if lyrics.background_vocal_ranges:
        _remove_background_vocal_notes(melody.notes_path, lyrics.background_vocal_ranges)

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
    }
    (song_cache_dir / _META_FILENAME).write_text(json.dumps(meta, indent=2))

    return SongAssets(
        instrumental_path=final_instrumental_path,
        vocals_path=final_vocals_path,
        midi_path=final_midi_path,
        notes_path=final_notes_path,
        lyrics_path=final_lyrics_path,
    )
