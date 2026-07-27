"""Phase 1 deliverable CLI: extract the vocal melody from an isolated vocal stem.

Usage:
    venv/Scripts/python.exe scripts/extract_melody.py path/to/vocals.wav [--output-dir output]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.melody_extraction import extract_melody  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to an isolated vocal stem wav file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to save the extracted melody MIDI file (default: ./output)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    result = extract_melody(args.input_path, args.output_dir)
    elapsed = time.perf_counter() - start

    print(f"MIDI saved to: {result.midi_path}")
    print(f"Notes JSON saved to: {result.notes_path}")
    print(f"Extraction took {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
