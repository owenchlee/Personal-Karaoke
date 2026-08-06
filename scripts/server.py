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
import re
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline.device import get_device_info  # noqa: E402
from audio_pipeline.download import download_audio, probe_title  # noqa: E402
from audio_pipeline.mastering import master_recording  # noqa: E402
from audio_pipeline.pipeline import is_cached, process_song, slugify  # noqa: E402
from audio_pipeline.transcode import transcode_to_mp3  # noqa: E402
from publish_song import publish_song  # noqa: E402

CACHE_DIR = Path("cache")
PUBLIC_DIR = Path("frontend/public/cache")
RECORDINGS_DIR = Path("recordings")
SCORES_DIR = Path("scores")


@dataclass
class Job:
    id: str
    url: str
    language: str | None = None
    separation_model: str = "htdemucs"
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


def _cached_separation_model(song_cache_dir: Path) -> str:
    """The separation model recorded in `song_cache_dir`'s meta.json, or the
    default ("htdemucs") for a cached song processed before this field
    existed.
    """
    meta_path = song_cache_dir / "meta.json"
    if not meta_path.exists():
        return "htdemucs"
    meta = json.loads(meta_path.read_text())
    return meta.get("separation_model", "htdemucs")


def _needs_reprocessing(slug: str, requested_model: str) -> bool:
    if not is_cached(CACHE_DIR, slug):
        return True
    return _cached_separation_model(CACHE_DIR / slug) != requested_model


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

            if _needs_reprocessing(slug, job.separation_model):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    downloaded_path, title = download_audio(
                        job.url, Path(tmp_dir),
                        on_progress=lambda fraction: on_progress("downloading", fraction),
                    )
                    slug = slugify(title)
                    # Re-check against the *real* (downloaded) title's slug -- probe_title's
                    # guess and the real title can differ, and each has its own cache entry.
                    force = is_cached(CACHE_DIR, slug) and (
                        _cached_separation_model(CACHE_DIR / slug) != job.separation_model
                    )

                    process_song(
                        downloaded_path, cache_dir=CACHE_DIR, song_id=slug, on_progress=on_progress,
                        language=job.language, lyrics_query=title,
                        separation_model=job.separation_model, force=force,
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
    separation_model: Literal["htdemucs", "htdemucs_ft", "bs_roformer"] = "htdemucs"


class SubmitScoreRequest(BaseModel):
    song_id: str
    score: int


class SetStarredRequest(BaseModel):
    starred: bool


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
    job = Job(
        id=job_id, url=request.url, language=request.language,
        separation_model=request.separation_model,
    )
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


@app.get("/api/gpu-status")
def get_gpu_status() -> dict:
    """Whether processing (separation/melody/lyrics) will run on an
    NVIDIA/CUDA GPU or fall back to CPU -- lets the frontend show "NVIDIA GPU
    detected" next to the separation-model picker instead of leaving users to
    guess.
    """
    return get_device_info()


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
            starred = False
            meta_path = CACHE_DIR / entry.name / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                title = meta.get("title") or entry.name
                processed_at = meta.get("processed_at")
                starred = bool(meta.get("starred", False))
            songs.append({
                "slug": entry.name,
                "title": title,
                "processed_at": processed_at,
                "starred": starred,
            })
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


@app.put("/api/songs/{slug}/starred")
def set_song_starred(slug: str, request: SetStarredRequest) -> dict:
    """Star/unstar a cached song so it can be pinned to the top of the "My
    songs" library list (sorting is left to the frontend, same convention as
    GET /api/songs). Persisted in meta.json alongside the title, creating it
    if the song was published before meta.json existed (see
    test_list_songs_falls_back_to_slug_when_meta_is_missing).
    """
    public_dir = _song_dir(PUBLIC_DIR, slug)
    if not public_dir.exists():
        raise HTTPException(status_code=404, detail="Unknown song")

    cache_dir = _song_dir(CACHE_DIR, slug)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["starred"] = request.starred
    meta_path.write_text(json.dumps(meta, indent=2))

    return {"slug": slug, "starred": request.starred}


def _recording_title(slug: str) -> str:
    """Human-readable title for a recording's song, read from the same
    meta.json `_sync_meta_title` above keeps in sync -- falls back to the
    raw slug if the song has no cached title (or was deleted since).
    """
    meta_path = CACHE_DIR / slug / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("title"):
            return meta["title"]
    return slug


def _parse_recording_filename(path: Path) -> tuple[str, int] | None:
    """Recordings are saved on disk as `<slug>__<epoch-seconds>.mp3` -- parses
    that back into its parts, or None for a file that doesn't match (e.g. a
    stray file dropped into RECORDINGS_DIR by hand).
    """
    slug, separator, epoch_part = path.stem.rpartition("__")
    if not separator or not epoch_part.isdigit():
        return None
    return slug, int(epoch_part)


def _recording_download_name(slug: str, epoch_seconds: int) -> str:
    """The name to hand the browser for a recording download -- the song's
    title followed by when it was recorded, per the requested "song name,
    then the timestamp" format. Sanitized for filesystem-unsafe characters
    since real song titles (e.g. from a YouTube title) can contain them.
    """
    title = _recording_title(slug)
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title).strip() or slug
    recorded_at = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return f"{safe_title}-{recorded_at.strftime('%Y-%m-%d-%H%M%S')}.mp3"


def _recording_path(filename: str) -> Path:
    """Resolve `filename` under RECORDINGS_DIR, rejecting anything (`../etc`,
    absolute paths, etc.) that would resolve outside of it -- same guard as
    `_song_dir` above, since this filename comes straight from a URL path.
    """
    base_resolved = RECORDINGS_DIR.resolve()
    candidate = (base_resolved / filename).resolve()
    if candidate.parent != base_resolved:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


@app.post("/api/recordings/mp3")
def render_recording_mp3(
    vocal: UploadFile = File(...),
    instrumental: UploadFile = File(...),
    song_id: str = Form("recording"),
    calibration_offset_seconds: float | None = Form(None),
) -> FileResponse:
    """Master (denoise/balance/align -- see audio_pipeline/mastering.py) and
    transcode a recorded take -- the browser's separately-recorded vocal and
    instrumental MediaRecorder tracks, see frontend/src/hooks/useRecording.ts
    -- into an mp3, save it under RECORDINGS_DIR so it shows up in the "My
    recordings" list later, and return it directly so the browser can also
    download it immediately.

    ``calibration_offset_seconds``, when the browser sent one, is that
    player's own mic-latency calibration (frontend/src/game/calibration.ts,
    already measured for live-pitch sync and re-run whenever they switch
    mics) -- passed straight through to `master_recording` so the vocal/
    instrumental start-alignment uses real per-device latency instead of one
    fixed guess. Omitted (``None``) for a player who hasn't calibrated yet,
    in which case `master_recording` falls back to its own module-level
    default.

    Deliberately a plain (not async) def: this does real, sequential ffmpeg
    work (master_recording/transcode_to_mp3) that would otherwise block the
    single-process event loop for the whole request, freezing every other
    route (e.g. GET /api/jobs/{id} polling) -- FastAPI runs sync handlers in
    its threadpool automatically, same fix already applied to _run_job's
    background thread for process_song/download_audio.
    """
    vocal_bytes = vocal.file.read()
    instrumental_bytes = instrumental.file.read()
    if not vocal_bytes or not instrumental_bytes:
        raise HTTPException(status_code=400, detail="Empty recording")

    slug = slugify(song_id)
    recorded_at = datetime.now(timezone.utc)
    epoch_seconds = int(recorded_at.timestamp())
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = RECORDINGS_DIR / f"{slug}__{epoch_seconds}.mp3"

    with tempfile.TemporaryDirectory() as tmp_dir:
        vocal_path = Path(tmp_dir) / "vocal.webm"
        instrumental_path = Path(tmp_dir) / "instrumental.webm"
        vocal_path.write_bytes(vocal_bytes)
        instrumental_path.write_bytes(instrumental_bytes)
        try:
            mastered_path = master_recording(
                vocal_path, instrumental_path, Path(tmp_dir) / "mastered", calibration_offset_seconds
            )
            mp3_path = transcode_to_mp3(mastered_path, Path(tmp_dir))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        shutil.copy(mp3_path, saved_path)

    return FileResponse(
        saved_path,
        media_type="audio/mpeg",
        filename=_recording_download_name(slug, epoch_seconds),
    )


@app.get("/api/recordings")
def list_recordings() -> dict:
    """List saved recordings for the "My recordings" screen -- sorting is
    left to the frontend (same convention as GET /api/songs above).
    """
    recordings = []
    if RECORDINGS_DIR.exists():
        for entry in RECORDINGS_DIR.glob("*.mp3"):
            parsed = _parse_recording_filename(entry)
            if parsed is None:
                continue
            slug, epoch_seconds = parsed
            recordings.append({
                "filename": entry.name,
                "slug": slug,
                "title": _recording_title(slug),
                "recorded_at": datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat(),
            })
    return {"recordings": recordings}


@app.get("/api/recordings/{filename}")
def download_recording(filename: str) -> FileResponse:
    path = _recording_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown recording")

    parsed = _parse_recording_filename(path)
    download_name = _recording_download_name(*parsed) if parsed else path.name
    return FileResponse(path, media_type="audio/mpeg", filename=download_name)


@app.delete("/api/recordings/{filename}")
def delete_recording(filename: str) -> dict:
    """Remove a saved recording from RECORDINGS_DIR (the "My recordings"
    screen's delete action) -- same not-found/path-escape guards as
    download_recording above, via the shared `_recording_path` helper.
    """
    path = _recording_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown recording")
    path.unlink()
    return {"deleted": filename}


def _score_path(slug: str) -> Path:
    """Resolve scores/<slug>.json, rejecting anything (`../etc`, absolute
    paths, etc.) that would resolve outside of SCORES_DIR -- same guard as
    `_song_dir`/`_recording_path` above, since slug comes from the POST
    /api/scores request body.
    """
    base_resolved = SCORES_DIR.resolve()
    candidate = (base_resolved / f"{slug}.json").resolve()
    if candidate.parent != base_resolved:
        raise HTTPException(status_code=400, detail="Invalid slug")
    return candidate


@app.post("/api/scores")
def submit_score(request: SubmitScoreRequest) -> dict:
    """Record a completed play-through's score against `scores/<slug>.json`,
    updating the running best if beaten. Returns the updated record plus
    `is_new_best` and `previous_best` so the frontend can render the
    "New high score! (was X%)" banner in one round trip, without a second
    GET to fetch what the old best used to be.
    """
    slug = slugify(request.song_id)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = _score_path(slug)
    now = datetime.now(timezone.utc).isoformat()

    # Serialize read-modify-write per slug -- same lock dict _run_job uses to
    # avoid two jobs stomping on the same cache/<slug>/, reused here since a
    # double-submit (e.g. a flaky retry, or two tabs open on the same song)
    # racing on scores/<slug>.json has the identical shape of problem.
    with _lock_for_slug(slug):
        if path.exists():
            record = json.loads(path.read_text())
        else:
            record = {
                "slug": slug,
                "best_score": None,
                "best_achieved_at": None,
                "play_count": 0,
                "last_played_at": None,
            }

        previous_best = record["best_score"]
        is_new_best = previous_best is None or request.score > previous_best

        record["play_count"] += 1
        record["last_played_at"] = now
        if is_new_best:
            record["best_score"] = request.score
            record["best_achieved_at"] = now

        path.write_text(json.dumps(record, indent=2))

    return {**record, "is_new_best": is_new_best, "previous_best": previous_best}


@app.get("/api/scores")
def list_scores() -> dict:
    """List saved scores for the "High Scores" screen -- sorting and badge
    computation are left to the frontend (same convention as GET /api/songs
    / GET /api/recordings above). Title is resolved fresh from
    cache/<slug>/meta.json each time, not stored in the score file itself, so
    a later title edit (_sync_meta_title) never leaves a score record stale.
    """
    scores = []
    if SCORES_DIR.exists():
        for entry in SCORES_DIR.glob("*.json"):
            slug = entry.stem
            record = json.loads(entry.read_text())
            scores.append({
                "slug": slug,
                "title": _recording_title(slug),
                "best_score": record.get("best_score"),
                "best_achieved_at": record.get("best_achieved_at"),
                "play_count": record.get("play_count", 0),
                "last_played_at": record.get("last_played_at"),
            })
    return {"scores": scores}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
