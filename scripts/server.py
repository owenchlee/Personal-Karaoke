"""Local job server: lets the frontend load a song straight from a link.

Exposes two endpoints the frontend polls against:

    POST /api/jobs        {"url": "..."}  -> {"job_id": "..."}
    GET  /api/jobs/{id}   -> {"status": ..., "progress": ..., "slug": ..., "error": ...}

Each job downloads audio from the given URL (yt-dlp), runs it through the
existing offline pipeline (separate/extract melody/transcribe lyrics), then
publishes the result to frontend/public/cache/<slug>/ -- the same steps a
human would otherwise run by hand via process_song.py + publish_song.py.
Runs in a background thread per job so the HTTP request returns immediately;
the frontend polls for progress since a real song takes 1.5-7 minutes.

Usage:
    venv/Scripts/python.exe scripts/server.py
"""
import json
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline.download import download_audio, probe_title  # noqa: E402
from audio_pipeline.pipeline import is_cached, process_song, slugify  # noqa: E402
from publish_song import publish_song  # noqa: E402

CACHE_DIR = Path("cache")
PUBLIC_DIR = Path("frontend/public/cache")


@dataclass
class Job:
    id: str
    url: str
    language: str | None = None
    status: str = "queued"
    progress: float = 0.0
    slug: str | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# Overall 0-1 job progress each stage spans, so the frontend can show one
# continuous, accurate percentage instead of jumping between 6 equal-sized
# stage buckets. Weighted from measured timings on a real ~4.5min song (see
# NOTES.md): downloading and publishing are quick, separating is a real but
# secondary cost (~80s), and transcribing_lyrics (large-v3, CPU) dominates
# wall time (~270-380s) whenever it falls back from the fast online lookup --
# extracting_melody runs concurrently with it and has no progress signal of
# its own, so it shares transcribing_lyrics's span and lets that fraction
# drive the bar.
_STAGE_SPANS: dict[str, tuple[float, float]] = {
    "queued": (0.0, 0.0),
    "downloading": (0.0, 0.05),
    "separating": (0.05, 0.30),
    "extracting_melody": (0.30, 0.95),
    "transcribing_lyrics": (0.30, 0.95),
    "publishing": (0.95, 1.0),
    "done": (1.0, 1.0),
}


def _overall_progress(stage: str, fraction: float) -> float:
    start, end = _STAGE_SPANS[stage]
    return start + (end - start) * min(max(fraction, 0.0), 1.0)


_JOBS: dict[str, Job] = {}

# Guards against two jobs processing the same song concurrently. Without
# this, two requests for the same URL that overlap in time both run the full
# pipeline against the same deterministic intermediate filenames inside the
# same cache/<slug>/ dir (extract_audio/separate_stems name their outputs
# from the source video id, not per-job), and whichever finishes first
# renames those shared files out from under the other -- surfacing as a
# "[WinError 2] ... instrumental.wav" FileNotFoundError partway through the
# second job. Serializing per slug means a second concurrent request simply
# waits, then hits the is_cached() fast path below once the first finishes.
_SLUG_LOCKS: dict[str, threading.Lock] = {}
_SLUG_LOCKS_GUARD = threading.Lock()


def _lock_for_slug(slug: str) -> threading.Lock:
    with _SLUG_LOCKS_GUARD:
        return _SLUG_LOCKS.setdefault(slug, threading.Lock())


def _song_dir(base: Path, slug: str) -> Path:
    """Resolve `slug` under `base`, rejecting anything (`../etc`, absolute
    paths, etc.) that would resolve outside of it -- the slug in DELETE
    /api/songs/{slug} comes straight from the URL path.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / slug).resolve()
    if candidate.parent != base_resolved:
        raise HTTPException(status_code=400, detail="Invalid slug")
    return candidate


def _sync_meta_title(song_cache_dir: Path, title: str) -> None:
    """Keep meta.json's human-readable title in sync with the most recently
    seen source title -- used by GET /api/songs to display "Bohemian
    Rhapsody" instead of the cache slug "bohemian-rhapsody". A no-op if the
    song hasn't been processed yet (meta.json doesn't exist).
    """
    meta_path = song_cache_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("title") != title:
        meta["title"] = title
        meta_path.write_text(json.dumps(meta, indent=2))


def _run_job(job: Job) -> None:
    def on_progress(stage: str, fraction: float = 0.0) -> None:
        with job.lock:
            job.status = stage
            job.progress = _overall_progress(stage, fraction)

    try:
        title = probe_title(job.url)
        slug = slugify(title)

        on_progress("queued")
        with _lock_for_slug(slug):
            on_progress("downloading")

            if not is_cached(CACHE_DIR, slug):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    downloaded_path, title = download_audio(
                        job.url, Path(tmp_dir),
                        on_progress=lambda fraction: on_progress("downloading", fraction),
                    )
                    slug = slugify(title)

                    process_song(
                        downloaded_path, cache_dir=CACHE_DIR, song_id=slug, on_progress=on_progress,
                        language=job.language, lyrics_query=title,
                    )

            _sync_meta_title(CACHE_DIR / slug, title)

        on_progress("publishing")
        publish_song(slug, CACHE_DIR, PUBLIC_DIR)
        # publish_song re-reads from CACHE_DIR by slug -- this just confirms
        # the cached/just-processed result actually has what it expects.
        assert (CACHE_DIR / slug / "notes.json").exists()

        with job.lock:
            job.status = "done"
            job.progress = 1.0
            job.slug = slug
    except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
        with job.lock:
            job.status = "error"
            job.error = str(exc)


class CreateJobRequest(BaseModel):
    url: str
    language: Literal["en", "yue"] | None = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/jobs")
def create_job(request: CreateJobRequest) -> dict:
    job_id = uuid.uuid4().hex
    job = Job(id=job_id, url=request.url, language=request.language)
    _JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    with job.lock:
        return {
            "status": job.status,
            "progress": job.progress,
            "slug": job.slug,
            "error": job.error,
        }


@app.get("/api/songs")
def list_songs() -> dict:
    """List already-processed, playable songs, for a "pick from your
    library" UI that reloads a song instantly with no processing at all.
    """
    songs = []
    if PUBLIC_DIR.exists():
        for entry in PUBLIC_DIR.iterdir():
            if not entry.is_dir() or not (entry / "notes.json").exists():
                continue
            title = entry.name
            processed_at = None
            meta_path = CACHE_DIR / entry.name / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                title = meta.get("title") or entry.name
                processed_at = meta.get("processed_at")
            songs.append({"slug": entry.name, "title": title, "processed_at": processed_at})
    return {"songs": songs}


@app.delete("/api/songs/{slug}")
def delete_song(slug: str) -> dict:
    """Remove a cached song from the library -- both the playable copy under
    PUBLIC_DIR and the raw pipeline output under CACHE_DIR, so a re-add
    reprocesses from scratch rather than resurrecting stale intermediates.
    """
    public_dir = _song_dir(PUBLIC_DIR, slug)
    if not public_dir.exists():
        raise HTTPException(status_code=404, detail="Unknown song")
    shutil.rmtree(public_dir)

    cache_dir = _song_dir(CACHE_DIR, slug)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    return {"deleted": slug}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
