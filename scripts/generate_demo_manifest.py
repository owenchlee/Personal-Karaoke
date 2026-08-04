"""Generate the static song manifest for the public demo build.

The demo build (`npm run build:demo`, see frontend/.env.demo) has no job
server behind it, so SongLibrary reads the "cached songs" list from
frontend/public/demo-cache/manifest.json instead of GET /api/songs. This
script produces that file with the same {slug, title, processed_at, starred}
shape as the live endpoint (see list_songs() in scripts/server.py).

demo-cache/ (not cache/) is the target on purpose: frontend/public/cache/ is
.gitignored because it accumulates whatever a local user has processed --
frequently copyrighted tracks not safe to publish. demo-cache/ is git-tracked
(see the exception in .gitignore) and should only ever contain songs you have
the rights to redistribute publicly.

Usage:
    venv/Scripts/python.exe scripts/publish_song.py <song-slug> --public-dir frontend/public/demo-cache
    venv/Scripts/python.exe scripts/generate_demo_manifest.py
        [--cache-dir cache] [--public-dir frontend/public/demo-cache]
"""
import argparse
import json
from pathlib import Path


def generate_demo_manifest(cache_dir: Path, public_dir: Path) -> dict:
    songs = []
    if public_dir.exists():
        for entry in sorted(public_dir.iterdir()):
            if not entry.is_dir() or not (entry / "notes.json").exists():
                continue
            title = entry.name
            processed_at = None
            meta_path = cache_dir / entry.name / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                title = meta.get("title") or entry.name
                processed_at = meta.get("processed_at")
            songs.append({
                "slug": entry.name,
                "title": title,
                "processed_at": processed_at,
                # Star/delete are hidden in demo mode (no server to persist them to), so every
                # manifest entry is unstarred -- see DEMO_MODE handling in SongLibrary.tsx.
                "starred": False,
            })
    return {"songs": songs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"),
        help="Directory containing per-song cached assets/meta.json (default: ./cache)",
    )
    parser.add_argument(
        "--public-dir", type=Path, default=Path("frontend/public/demo-cache"),
        help="Directory of published frontend-facing song assets (default: ./frontend/public/demo-cache)",
    )
    args = parser.parse_args()

    manifest = generate_demo_manifest(args.cache_dir, args.public_dir)
    out_path = args.public_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path} ({len(manifest['songs'])} song(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
