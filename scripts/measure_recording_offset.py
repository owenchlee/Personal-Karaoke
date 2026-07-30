"""One-off calibration: measures the constant start-offset between a raw
two-track recording's vocal and instrumental tracks, so
audio_pipeline/mastering.py's _RECORDING_OFFSET_SECONDS can be set from a
real measurement instead of a guess (see that constant's docstring, and the
"Recording start offset" section of
docs/superpowers/specs/2026-07-30-auto-balance-recording-design.md).

Procedure:
1. Temporarily record a take where you clap once, sharply, right on the
   song's first strong instrumental hit (e.g. the first beat after any
   count-in).
2. Get the two raw (pre-mastering) tracks for that take as wav files -- e.g.
   temporarily add two `shutil.copy(...)` lines in
   scripts/server.py's render_recording_mp3, right after it writes
   vocal_path/instrumental_path, copying them out to a scratch directory
   before mastering runs; remove those lines again once done.
3. Run:
       venv/Scripts/python.exe scripts/measure_recording_offset.py <vocal.wav> <instrumental.wav>
4. Update _RECORDING_OFFSET_SECONDS in audio_pipeline/mastering.py with the
   printed value.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pipeline.mastering import measure_start_offset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vocal_wav", type=Path)
    parser.add_argument("instrumental_wav", type=Path)
    args = parser.parse_args()

    offset = measure_start_offset(args.vocal_wav, args.instrumental_wav)
    print(f"Measured offset: {offset:+.3f}s")
    print("(positive = vocal lags instrumental -- set _RECORDING_OFFSET_SECONDS in")
    print(" audio_pipeline/mastering.py to this value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
