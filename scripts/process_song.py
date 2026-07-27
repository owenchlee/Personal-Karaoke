"""End-to-end song processing CLI: video -> cached instrumental audio + note JSON.

Usage:
    venv/Scripts/python.exe scripts/process_song.py path/to/video.mp4 [--cache-dir cache]
        [--song-id my-song] [--force]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.pipeline import process_song  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to a video (or audio) file")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"),
        help="Directory to store/read per-song cached assets (default: ./cache)",
    )
    parser.add_argument(
        "--song-id", type=str, default=None,
        help="Explicit cache key for this song (default: derived from the input filename)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if cached assets already exist for this song",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    assets = process_song(
        args.input_path, cache_dir=args.cache_dir, song_id=args.song_id, force=args.force,
    )
    elapsed = time.perf_counter() - start

    print(f"Instrumental: {assets.instrumental_path}")
    print(f"Vocals: {assets.vocals_path}")
    print(f"MIDI: {assets.midi_path}")
    print(f"Notes JSON: {assets.notes_path}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
