"""Publish a processed song's frontend-facing assets for local dev serving.

Copies notes.json + instrumental.wav from cache/<song-id>/ into
frontend/public/cache/<song-id>/ so Vite's dev server can serve them (with
correct HTTP Range support for <audio> seeking) without a custom backend.

Usage:
    venv/Scripts/python.exe scripts/publish_song.py test-song
        [--cache-dir cache] [--public-dir frontend/public/cache]
"""
import argparse
import shutil
import sys
from pathlib import Path

_ASSETS_TO_PUBLISH = ("notes.json", "instrumental.wav")


def publish_song(song_id: str, cache_dir: str | Path, public_dir: str | Path) -> Path:
    """Copy a cached song's frontend-facing assets into ``public_dir``.

    Returns the destination directory.
    """
    source_dir = Path(cache_dir) / song_id
    dest_dir = Path(public_dir) / song_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in _ASSETS_TO_PUBLISH:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Expected cached asset not found: {source_path}")
        shutil.copyfile(source_path, dest_dir / filename)

    return dest_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song_id", help="Cache slug of a previously processed song")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"),
        help="Directory containing per-song cached assets (default: ./cache)",
    )
    parser.add_argument(
        "--public-dir", type=Path, default=Path("frontend/public/cache"),
        help="Directory to publish frontend-facing assets into (default: ./frontend/public/cache)",
    )
    args = parser.parse_args()

    source_dir = args.cache_dir / args.song_id
    if not source_dir.exists():
        print(f"No cached song found at: {source_dir}", file=sys.stderr)
        return 1

    dest_dir = publish_song(args.song_id, args.cache_dir, args.public_dir)
    print(f"Published to: {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
