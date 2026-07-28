"""End-to-end song processing: video -> audio -> stems -> melody + lyrics, with
caching.

Orchestrates video_extraction, separation, melody_extraction, and
lyrics_extraction into a single call, keyed by a per-song cache directory so a
song already processed is never reprocessed unnecessarily.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


def _slugify(name: str) -> str:
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


def process_song(
    video_path: str | Path,
    cache_dir: str | Path = Path("cache"),
    song_id: str | None = None,
    force: bool = False,
) -> SongAssets:
    """Process ``video_path`` end-to-end into cached instrumental audio and a
    note-event JSON, skipping reprocessing if a cached result already exists
    for this song (unless ``force`` is set).
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    slug = _slugify(song_id if song_id is not None else video_path.stem)
    song_cache_dir = cache_dir / slug

    if not force:
        cached = _cached_assets(song_cache_dir)
        if cached is not None:
            return cached

    song_cache_dir.mkdir(parents=True, exist_ok=True)

    extracted_wav = extract_audio(video_path, song_cache_dir)
    vocals_path, instrumental_path = separate_stems(extracted_wav, song_cache_dir)
    melody = extract_melody(vocals_path, song_cache_dir)
    lyrics = extract_lyrics(vocals_path, song_cache_dir)
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
    }
    (song_cache_dir / _META_FILENAME).write_text(json.dumps(meta, indent=2))

    return SongAssets(
        instrumental_path=final_instrumental_path,
        vocals_path=final_vocals_path,
        midi_path=final_midi_path,
        notes_path=final_notes_path,
        lyrics_path=final_lyrics_path,
    )
