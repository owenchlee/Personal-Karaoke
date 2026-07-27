"""Isolate the vocal stem and instrumental track from an audio file.

Usage:
    venv/Scripts/python.exe scripts/separate_stems.py path/to/song.wav [--output-dir output]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.separation import separate_stems  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to an audio file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to save the vocal stem and instrumental track (default: ./output)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    vocals_path, instrumental_path = separate_stems(args.input_path, args.output_dir)
    elapsed = time.perf_counter() - start

    print(f"Vocal stem saved to: {vocals_path}")
    print(f"Instrumental saved to: {instrumental_path}")
    print(f"Separation took {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
