"""Extract the audio track from a video file.

Usage:
    venv/Scripts/python.exe scripts/extract_audio.py path/to/video.mp4 [--output-dir output]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.video_extraction import extract_audio  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to a video (or audio) file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to save the extracted wav file (default: ./output)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    wav_path = extract_audio(args.input_path, args.output_dir)
    elapsed = time.perf_counter() - start

    print(f"Audio saved to: {wav_path}")
    print(f"Extraction took {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
