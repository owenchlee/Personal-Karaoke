"""Phase 1 deliverable CLI: transcribe lyrics from an isolated vocal stem.

Usage:
    venv/Scripts/python.exe scripts/extract_lyrics.py path/to/vocals.wav [--output-dir output]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.lyrics_extraction import extract_lyrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to an isolated vocal stem wav file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to save the transcribed lyrics JSON (default: ./output)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    result = extract_lyrics(args.input_path, args.output_dir)
    elapsed = time.perf_counter() - start

    print(f"Lyrics JSON saved to: {result.lyrics_path}")
    print(f"Transcription took {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
