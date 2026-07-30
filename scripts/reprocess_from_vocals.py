"""Re-run melody extraction + lyrics acquisition for already-cached songs, from
their existing vocal stem -- no re-download, no re-running Demucs separation.

For picking up a melody/lyrics-only pipeline improvement (e.g. the CTC-forced-
alignment change, see NOTES.md's "Lyrics timing" entry) on songs whose
original source video is no longer available to run a full process_song on.
Republishes each reprocessed song's frontend-facing assets afterward.

Usage:
    venv/Scripts/python.exe scripts/reprocess_from_vocals.py --all
    venv/Scripts/python.exe scripts/reprocess_from_vocals.py song-slug-1 song-slug-2
        [--cache-dir cache] [--public-dir frontend/public/cache]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline.pipeline import reprocess_melody_and_lyrics  # noqa: E402
from publish_song import publish_song  # noqa: E402


def _cached_song_ids(cache_dir: Path) -> list[str]:
    if not cache_dir.exists():
        return []
    return sorted(
        entry.name for entry in cache_dir.iterdir()
        if entry.is_dir() and (entry / "vocals.wav").exists()
    )


def _title_for(cache_dir: Path, song_id: str) -> str | None:
    """Best-effort lyrics_query for a cached song -- its own previously-recorded title if known,
    matching what the original processing (job server / process_song CLI) would have used. Songs
    with no recorded title were originally processed with no lyrics_query either (e.g. a local
    file run directly through process_song.py), so leaving it None here reproduces the same
    Whisper-transcription path they already used, not a regression.
    """
    meta_path = cache_dir / song_id / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return meta.get("title")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song_ids", nargs="*", help="Specific cache slugs to reprocess")
    parser.add_argument("--all", action="store_true", help="Reprocess every cached song")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"),
        help="Directory containing per-song cached assets (default: ./cache)",
    )
    parser.add_argument(
        "--public-dir", type=Path, default=Path("frontend/public/cache"),
        help="Directory to publish frontend-facing assets into (default: ./frontend/public/cache)",
    )
    args = parser.parse_args()

    if not args.all and not args.song_ids:
        print("Specify song slugs, or pass --all to reprocess every cached song.", file=sys.stderr)
        return 1

    song_ids = _cached_song_ids(args.cache_dir) if args.all else args.song_ids

    failures = []
    for i, song_id in enumerate(song_ids, start=1):
        title = _title_for(args.cache_dir, song_id)
        print(f"[{i}/{len(song_ids)}] {song_id} (lyrics_query={title!r})")
        start = time.perf_counter()
        try:
            reprocess_melody_and_lyrics(args.cache_dir, song_id, lyrics_query=title)
            publish_song(song_id, args.cache_dir, args.public_dir)
        except Exception as exc:  # noqa: BLE001 - reported per-song, doesn't abort the batch
            failures.append((song_id, str(exc)))
            print(f"  FAILED: {exc}")
            continue
        print(f"  done in {time.perf_counter() - start:.1f}s")

    if failures:
        print(f"\n{len(failures)}/{len(song_ids)} song(s) failed:", file=sys.stderr)
        for song_id, error in failures:
            print(f"  {song_id}: {error}", file=sys.stderr)
        return 1

    print(f"\nReprocessed {len(song_ids)} song(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
