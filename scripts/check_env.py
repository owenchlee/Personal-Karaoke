"""Phase 0 environment check: confirm every pipeline dependency is importable
and that the system ffmpeg/ffprobe binaries (not just the Python wrapper) are
on PATH. Run with: venv/Scripts/python.exe scripts/check_env.py
"""
import shutil
import sys


def check_imports() -> list[str]:
    modules = ["ffmpeg", "demucs", "basic_pitch", "librosa", "soundfile"]
    errors = []
    for name in modules:
        try:
            __import__(name)
            print(f"OK   import {name}")
        except ImportError as exc:
            errors.append(f"FAIL import {name}: {exc}")
    return errors


def check_ffmpeg_binary() -> list[str]:
    errors = []
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if path:
            print(f"OK   system binary {binary} -> {path}")
        else:
            errors.append(f"FAIL system binary {binary} not found on PATH")
    return errors


def main() -> int:
    errors = check_imports() + check_ffmpeg_binary()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\n{len(errors)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll environment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
