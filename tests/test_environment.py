import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_env import check_ffmpeg_binary, check_imports  # noqa: E402


def test_all_pipeline_packages_import_cleanly():
    errors = check_imports()
    assert errors == []


def test_system_ffmpeg_binaries_on_path():
    errors = check_ffmpeg_binary()
    assert errors == []
