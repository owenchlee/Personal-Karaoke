"""Tests for the job server's GET/DELETE /api/songs endpoints (the "song
library").

``scripts/server.py`` has no package ``__init__.py``, so it's imported here
the same way it resolves its own sibling imports at runtime: by inserting
``scripts/`` onto ``sys.path`` first. Driven through FastAPI's ``TestClient``
rather than spinning up a real HTTP server.
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import server  # noqa: E402

client = TestClient(server.app)


def test_get_gpu_status_reports_the_detected_device(monkeypatch):
    monkeypatch.setattr(
        server, "get_device_info", lambda: {"available": True, "name": "NVIDIA GeForce RTX 3060"}
    )

    response = client.get("/api/gpu-status")

    assert response.status_code == 200
    assert response.json() == {"available": True, "name": "NVIDIA GeForce RTX 3060"}


def test_get_gpu_status_reports_cpu_fallback(monkeypatch):
    monkeypatch.setattr(server, "get_device_info", lambda: {"available": False, "name": None})

    response = client.get("/api/gpu-status")

    assert response.status_code == 200
    assert response.json() == {"available": False, "name": None}


def test_list_songs_returns_empty_when_nothing_published(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(server, "PUBLIC_DIR", tmp_path / "public")

    response = client.get("/api/songs")

    assert response.status_code == 200
    assert response.json() == {"songs": []}


def test_list_songs_skips_unpublished_and_reads_title_from_meta(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    published_dir = public_dir / "my-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")

    # Has a public dir but never finished publishing (no notes.json yet) --
    # shouldn't show up as playable.
    unpublished_dir = public_dir / "half-done"
    unpublished_dir.mkdir(parents=True)

    meta_dir = cache_dir / "my-song"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        json.dumps(
            {
                "title": "My Song",
                "song_id": "my-song",
                "processed_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )

    response = client.get("/api/songs")

    assert response.status_code == 200
    assert response.json()["songs"] == [
        {
            "slug": "my-song",
            "title": "My Song",
            "processed_at": "2026-07-28T00:00:00+00:00",
            "starred": False,
        }
    ]


def test_list_songs_falls_back_to_slug_when_meta_is_missing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    # A song published via the old process_song.py/publish_song.py CLI path,
    # from before meta.json gained a "title" field.
    published_dir = public_dir / "old-cli-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")

    response = client.get("/api/songs")

    assert response.json()["songs"] == [
        {"slug": "old-cli-song", "title": "old-cli-song", "processed_at": None, "starred": False}
    ]


def test_delete_song_removes_public_and_cache_dirs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    published_dir = public_dir / "my-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")
    meta_dir = cache_dir / "my-song"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"title": "My Song"}))

    response = client.delete("/api/songs/my-song")

    assert response.status_code == 200
    assert response.json() == {"deleted": "my-song"}
    assert not published_dir.exists()
    assert not meta_dir.exists()
    assert client.get("/api/songs").json()["songs"] == []


def test_delete_song_is_idempotent_about_a_missing_cache_dir(tmp_path, monkeypatch):
    """The raw CACHE_DIR copy can be absent (e.g. already cleaned up by hand)
    even though the song is still published -- deleting shouldn't 500."""
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    published_dir = public_dir / "my-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")

    response = client.delete("/api/songs/my-song")

    assert response.status_code == 200
    assert not published_dir.exists()


def test_delete_song_404s_for_unknown_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(server, "PUBLIC_DIR", tmp_path / "public")

    response = client.delete("/api/songs/does-not-exist")

    assert response.status_code == 404


def test_set_song_starred_persists_in_meta_json(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    published_dir = public_dir / "my-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")
    meta_dir = cache_dir / "my-song"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"title": "My Song"}))

    response = client.put("/api/songs/my-song/starred", json={"starred": True})

    assert response.status_code == 200
    assert response.json() == {"slug": "my-song", "starred": True}
    meta = json.loads((meta_dir / "meta.json").read_text())
    assert meta == {"title": "My Song", "starred": True}
    assert client.get("/api/songs").json()["songs"][0]["starred"] is True


def test_set_song_starred_creates_meta_json_when_missing(tmp_path, monkeypatch):
    """A song published via the old CLI path has no meta.json yet -- starring
    it should create one rather than 404ing or crashing."""
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)

    published_dir = public_dir / "old-cli-song"
    published_dir.mkdir(parents=True)
    (published_dir / "notes.json").write_text("[]")

    response = client.put("/api/songs/old-cli-song/starred", json={"starred": True})

    assert response.status_code == 200
    meta = json.loads((cache_dir / "old-cli-song" / "meta.json").read_text())
    assert meta == {"starred": True}


def test_set_song_starred_404s_for_unknown_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(server, "PUBLIC_DIR", tmp_path / "public")

    response = client.put("/api/songs/does-not-exist/starred", json={"starred": True})

    assert response.status_code == 404


def _seed_cached_song(cache_dir: Path, slug: str) -> None:
    song_dir = cache_dir / slug
    song_dir.mkdir(parents=True)
    (song_dir / "instrumental.wav").write_bytes(b"original wav data")
    (song_dir / "notes.json").write_text(json.dumps([{"pitch_midi": 60}]))


def test_transpose_song_key_404s_for_unknown_song(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")

    response = client.post("/api/songs/does-not-exist/transpose", json={"semitones": 3})

    assert response.status_code == 404


def test_transpose_song_key_at_zero_semitones_points_back_at_the_original_without_generating(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    _seed_cached_song(cache_dir, "my-song")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called for a zero-semitone request")

    monkeypatch.setattr(server, "transpose_song", _fail_if_called)

    response = client.post("/api/songs/my-song/transpose", json={"semitones": 0})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "instrumental_url": "/api/songs/my-song/transpose/0/instrumental.wav",
        "notes_url": "/api/songs/my-song/transpose/0/notes.json",
    }

    instrumental_response = client.get(body["instrumental_url"])
    assert instrumental_response.status_code == 200
    assert instrumental_response.content == b"original wav data"


def test_transpose_song_key_generates_and_caches_a_shifted_variant(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    _seed_cached_song(cache_dir, "my-song")

    calls = []

    def fake_transpose_song(song_cache_dir, semitones, output_dir):
        calls.append(semitones)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "instrumental.wav").write_bytes(b"shifted wav data")
        (output_dir / "notes.json").write_text(json.dumps([{"pitch_midi": 63}]))
        from audio_pipeline.transpose import TransposeResult
        return TransposeResult(
            instrumental_path=output_dir / "instrumental.wav", notes_path=output_dir / "notes.json"
        )

    monkeypatch.setattr(server, "transpose_song", fake_transpose_song)

    response = client.post("/api/songs/my-song/transpose", json={"semitones": 3})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "instrumental_url": "/api/songs/my-song/transpose/3/instrumental.wav",
        "notes_url": "/api/songs/my-song/transpose/3/notes.json",
    }

    instrumental_response = client.get(body["instrumental_url"])
    assert instrumental_response.content == b"shifted wav data"
    notes_response = client.get(body["notes_url"])
    assert notes_response.json() == [{"pitch_midi": 63}]

    # A second request for the identical (song, semitones) pair is a cache hit -- doesn't
    # regenerate.
    client.post("/api/songs/my-song/transpose", json={"semitones": 3})
    assert calls == [3]


def test_transpose_song_key_rejects_a_shift_outside_the_allowed_range(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    _seed_cached_song(cache_dir, "my-song")

    response = client.post("/api/songs/my-song/transpose", json={"semitones": 13})

    assert response.status_code == 422


def test_transpose_song_key_returns_500_when_transposing_fails(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    _seed_cached_song(cache_dir, "my-song")

    def fake_transpose_song(song_cache_dir, semitones, output_dir):
        raise RuntimeError("rubberband is not on PATH")

    monkeypatch.setattr(server, "transpose_song", fake_transpose_song)

    response = client.post("/api/songs/my-song/transpose", json={"semitones": 3})

    assert response.status_code == 500


def test_get_transposed_instrumental_404s_when_not_yet_generated(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    _seed_cached_song(cache_dir, "my-song")

    response = client.get("/api/songs/my-song/transpose/5/instrumental.wav")

    assert response.status_code == 404


def test_song_dir_rejects_a_slug_that_resolves_above_base(tmp_path):
    """Unit-level check for the guard `delete_song` relies on: a literal "/"
    in the slug can't even reach that route (FastAPI's default path
    converter only matches one segment, so Starlette 404s those before our
    handler runs), but ".." alone *is* a single segment, and a raw,
    non-normalizing HTTP client could send it verbatim -- unlike httpx,
    which collapses ".." out of the URL client-side before we'd ever see it,
    making that specific case untestable through the route itself.
    """
    base = tmp_path / "public"
    base.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        server._song_dir(base, "..")

    assert exc_info.value.status_code == 400


def test_song_dir_rejects_a_windows_backslash_escape(tmp_path):
    """FastAPI's path converter only blocks a literal "/", not "\\" -- on
    Windows, pathlib treats a backslash as a separator too, so a slug like
    "..\\sibling" is one URL path segment (reaches the route fine) but two
    filesystem path components (escapes `base`) unless guarded against.
    """
    base = tmp_path / "public"
    base.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        server._song_dir(base, "..\\sibling")

    assert exc_info.value.status_code == 400


def test_run_job_forces_reprocessing_when_the_cached_model_differs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    # Pre-seed a fully cached song, separated with htdemucs (the default).
    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"cached wav data")
    (song_cache_dir / "notes.json").write_text("[]")
    (song_cache_dir / "lyrics.json").write_text("[]")
    (song_cache_dir / "meta.json").write_text(
        json.dumps({
            "song_id": "my-song", "processed_at": "2026-07-28T00:00:00+00:00",
            "separation_model": "htdemucs",
        })
    )

    process_song_calls = []

    def fake_probe_title(url):
        return "My Song"

    def fake_download_audio(url, output_dir, on_progress=None):
        return Path(output_dir) / "video.mp4", "My Song"

    def fake_process_song(
        video_path, cache_dir, song_id, on_progress, language, lyrics_query, separation_model,
        force,
    ):
        process_song_calls.append({"separation_model": separation_model, "force": force})
        song_cache_dir = Path(cache_dir) / song_id
        song_cache_dir.mkdir(parents=True, exist_ok=True)
        (song_cache_dir / "instrumental.wav").write_bytes(b"new wav data")
        (song_cache_dir / "notes.json").write_text("[]")
        (song_cache_dir / "lyrics.json").write_text("[]")
        (song_cache_dir / "meta.json").write_text(
            json.dumps({
                "song_id": song_id, "processed_at": "2026-07-29T00:00:00+00:00",
                "separation_model": separation_model,
            })
        )

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", fake_download_audio)
    monkeypatch.setattr(server, "process_song", fake_process_song)

    job = server.Job(
        id="job-1", url="https://example.com/watch?v=abc", separation_model="htdemucs_ft",
    )
    server._run_job(job)

    assert process_song_calls == [{"separation_model": "htdemucs_ft", "force": True}]
    assert job.status == "done"


def test_run_job_skips_reprocessing_when_the_cached_model_matches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    song_cache_dir = cache_dir / "my-song"
    song_cache_dir.mkdir(parents=True)
    (song_cache_dir / "instrumental.wav").write_bytes(b"cached wav data")
    (song_cache_dir / "notes.json").write_text("[]")
    (song_cache_dir / "lyrics.json").write_text("[]")
    (song_cache_dir / "meta.json").write_text(
        json.dumps({
            "song_id": "my-song", "processed_at": "2026-07-28T00:00:00+00:00",
            "separation_model": "htdemucs",
        })
    )

    def fake_probe_title(url):
        return "My Song"

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called when the cache already matches")

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", _fail_if_called)
    monkeypatch.setattr(server, "process_song", _fail_if_called)

    job = server.Job(
        id="job-1", url="https://example.com/watch?v=abc", separation_model="htdemucs",
    )
    server._run_job(job)

    assert job.status == "done"
    assert job.slug == "my-song"


def test_run_job_serializes_concurrent_requests_for_the_same_song(tmp_path, monkeypatch):
    """Regression test for a real bug: two /api/jobs requests for the same
    song that overlap in time both used to run the full pipeline against the
    same deterministic intermediate filenames inside the same cache/<slug>/
    dir (extract_audio/separate_stems name their outputs from the source
    video id, not per-job) -- whichever finished first renamed those shared
    files to their final cache names out from under the other, crashing it
    with "[WinError 2] ... instrumental.wav" partway through its own rename.
    _run_job now serializes per slug so a second overlapping request waits,
    then finds the cache already warm and skips reprocessing entirely.
    """
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    call_count = {"n": 0}
    entered_process_song = threading.Event()
    release_process_song = threading.Event()

    def fake_probe_title(url):
        return "My Song"

    def fake_download_audio(url, output_dir, on_progress=None):
        return Path(output_dir) / "video.mp4", "My Song"

    def fake_process_song(
        video_path, cache_dir, song_id, on_progress, language, lyrics_query, separation_model,
        force,
    ):
        call_count["n"] += 1
        entered_process_song.set()
        assert release_process_song.wait(timeout=5), "test deadlocked waiting to be released"
        song_cache_dir = Path(cache_dir) / song_id
        song_cache_dir.mkdir(parents=True, exist_ok=True)
        (song_cache_dir / "instrumental.wav").write_bytes(b"fake wav data")
        (song_cache_dir / "notes.json").write_text("[]")
        (song_cache_dir / "lyrics.json").write_text("[]")
        (song_cache_dir / "meta.json").write_text(
            json.dumps({"song_id": song_id, "processed_at": "2026-07-28T00:00:00+00:00"})
        )

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", fake_download_audio)
    monkeypatch.setattr(server, "process_song", fake_process_song)

    job1 = server.Job(id="job-1", url="https://example.com/watch?v=abc")
    job2 = server.Job(id="job-2", url="https://example.com/watch?v=abc")

    thread1 = threading.Thread(target=server._run_job, args=(job1,))
    thread1.start()
    assert entered_process_song.wait(timeout=5), "job1 never reached process_song"

    thread2 = threading.Thread(target=server._run_job, args=(job2,))
    thread2.start()
    # Give job2 every chance to (wrongly) start its own concurrent
    # process_song call while job1 is still mid-flight, before releasing it.
    thread2.join(timeout=0.3)
    assert call_count["n"] == 1, "job2 started processing concurrently instead of waiting"

    release_process_song.set()
    thread1.join(timeout=5)
    thread2.join(timeout=5)

    assert call_count["n"] == 1  # job2 found the cache warm and skipped reprocessing
    assert job1.status == "done"
    assert job2.status == "done"
    assert job1.slug == job2.slug == "my-song"


def test_submit_score_creates_a_new_record_on_first_play(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SCORES_DIR", tmp_path / "scores")
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    response = client.post("/api/scores", json={"song_id": "My Song", "score": 72})

    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "my-song"
    assert body["best_score"] == 72
    assert body["play_count"] == 1
    assert body["is_new_best"] is True
    assert body["previous_best"] is None


def test_submit_score_updates_best_when_beaten(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SCORES_DIR", tmp_path / "scores")
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    client.post("/api/scores", json={"song_id": "my-song", "score": 60})
    response = client.post("/api/scores", json={"song_id": "my-song", "score": 85})

    assert response.status_code == 200
    body = response.json()
    assert body["best_score"] == 85
    assert body["previous_best"] == 60
    assert body["play_count"] == 2
    assert body["is_new_best"] is True


def test_submit_score_keeps_previous_best_when_not_beaten(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SCORES_DIR", tmp_path / "scores")
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    client.post("/api/scores", json={"song_id": "my-song", "score": 85})
    response = client.post("/api/scores", json={"song_id": "my-song", "score": 60})

    assert response.status_code == 200
    body = response.json()
    assert body["best_score"] == 85
    assert body["previous_best"] == 85
    assert body["play_count"] == 2
    assert body["is_new_best"] is False


def test_list_scores_resolves_title_from_meta_json(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    scores_dir = tmp_path / "scores"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "SCORES_DIR", scores_dir)

    meta_dir = cache_dir / "my-song"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"title": "My Song"}))

    scores_dir.mkdir(parents=True)
    (scores_dir / "my-song.json").write_text(
        json.dumps(
            {
                "slug": "my-song",
                "best_score": 91,
                "best_achieved_at": "2026-07-28T00:00:00+00:00",
                "play_count": 3,
                "last_played_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )

    response = client.get("/api/scores")

    assert response.status_code == 200
    assert response.json()["scores"] == [
        {
            "slug": "my-song",
            "title": "My Song",
            "best_score": 91,
            "best_achieved_at": "2026-07-28T00:00:00+00:00",
            "play_count": 3,
            "last_played_at": "2026-07-28T00:00:00+00:00",
        }
    ]


def test_list_scores_falls_back_to_slug_when_meta_is_missing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    scores_dir = tmp_path / "scores"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "SCORES_DIR", scores_dir)

    scores_dir.mkdir(parents=True)
    (scores_dir / "old-cli-song.json").write_text(
        json.dumps(
            {
                "slug": "old-cli-song",
                "best_score": 50,
                "best_achieved_at": "2026-07-28T00:00:00+00:00",
                "play_count": 1,
                "last_played_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )

    response = client.get("/api/scores")

    assert response.json()["scores"][0]["title"] == "old-cli-song"


def test_score_path_rejects_a_slug_that_resolves_above_base(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SCORES_DIR", tmp_path / "scores")

    with pytest.raises(HTTPException) as exc_info:
        server._score_path("../escape")

    assert exc_info.value.status_code == 400


def test_delete_recording_removes_the_file(tmp_path, monkeypatch):
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    monkeypatch.setattr(server, "RECORDINGS_DIR", recordings_dir)

    recording_path = recordings_dir / "my-song__1700000000.mp3"
    recording_path.write_bytes(b"fake mp3 data")

    response = client.delete(f"/api/recordings/{recording_path.name}")

    assert response.status_code == 200
    assert response.json() == {"deleted": recording_path.name}
    assert not recording_path.exists()
    assert client.get("/api/recordings").json()["recordings"] == []


def test_delete_recording_404s_for_unknown_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    response = client.delete("/api/recordings/does-not-exist__1700000000.mp3")

    assert response.status_code == 404


def test_render_recording_mp3_masters_both_uploaded_tracks_and_saves_the_take(tmp_path, monkeypatch):
    recordings_dir = tmp_path / "recordings"
    monkeypatch.setattr(server, "RECORDINGS_DIR", recordings_dir)

    calls = []

    def fake_master_recording(vocal_path, instrumental_path, output_dir, calibration_offset_seconds=None):
        calls.append((Path(vocal_path).read_bytes(), Path(instrumental_path).read_bytes(), calibration_offset_seconds))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mastered_path = output_dir / "mastered.wav"
        t = np.linspace(0, 0.5, 22050, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(mastered_path, tone, 44100)
        return mastered_path

    monkeypatch.setattr(server, "master_recording", fake_master_recording)

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"fake vocal bytes", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert calls == [(b"fake vocal bytes", b"fake instrumental bytes", None)]

    saved = list(recordings_dir.glob("my-song__*.mp3"))
    assert len(saved) == 1


def test_render_recording_mp3_forwards_the_browser_s_calibration_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    calls = []

    def fake_master_recording(vocal_path, instrumental_path, output_dir, calibration_offset_seconds=None):
        calls.append(calibration_offset_seconds)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mastered_path = output_dir / "mastered.wav"
        t = np.linspace(0, 0.5, 22050, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(mastered_path, tone, 44100)
        return mastered_path

    monkeypatch.setattr(server, "master_recording", fake_master_recording)

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"fake vocal bytes", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song", "calibration_offset_seconds": "0.318"},
    )

    assert response.status_code == 200
    assert calls == [pytest.approx(0.318)]


def test_render_recording_mp3_rejects_an_empty_track(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 400


def test_render_recording_mp3_returns_500_when_mastering_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path / "recordings")

    def fake_master_recording(vocal_path, instrumental_path, output_dir, calibration_offset_seconds=None):
        raise RuntimeError("ffmpeg failed to clean vocal: boom")

    monkeypatch.setattr(server, "master_recording", fake_master_recording)

    response = client.post(
        "/api/recordings/mp3",
        files={
            "vocal": ("vocal.webm", b"fake vocal bytes", "audio/webm"),
            "instrumental": ("instrumental.webm", b"fake instrumental bytes", "audio/webm"),
        },
        data={"song_id": "My Song"},
    )

    assert response.status_code == 500


def test_overall_progress_maps_stage_fraction_into_its_weighted_span():
    assert server._overall_progress("separating", 0.0) == pytest.approx(0.05)
    assert server._overall_progress("separating", 1.0) == pytest.approx(0.30)
    assert server._overall_progress("transcribing_lyrics", 0.5) == pytest.approx(0.625)
    assert server._overall_progress("done", 0.0) == pytest.approx(1.0)


def test_overall_progress_clamps_out_of_range_fractions():
    assert server._overall_progress("separating", -1.0) == pytest.approx(0.05)
    assert server._overall_progress("separating", 5.0) == pytest.approx(0.30)


def test_run_job_reports_real_progress_and_reaches_1_on_success(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    public_dir = tmp_path / "public"
    monkeypatch.setattr(server, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(server, "_SLUG_LOCKS", {})

    def fake_probe_title(url):
        return "My Song"

    def fake_download_audio(url, output_dir, on_progress=None):
        if on_progress:
            on_progress(0.5)
        return Path(output_dir) / "video.mp4", "My Song"

    def fake_process_song(
        video_path, cache_dir, song_id, on_progress, language, lyrics_query, separation_model,
        force,
    ):
        on_progress("separating", 1.0)
        on_progress("transcribing_lyrics", 0.5)
        song_cache_dir = Path(cache_dir) / song_id
        song_cache_dir.mkdir(parents=True, exist_ok=True)
        (song_cache_dir / "instrumental.wav").write_bytes(b"fake wav data")
        (song_cache_dir / "notes.json").write_text("[]")
        (song_cache_dir / "lyrics.json").write_text("[]")
        (song_cache_dir / "meta.json").write_text(
            json.dumps({"song_id": song_id, "processed_at": "2026-07-28T00:00:00+00:00"})
        )

    monkeypatch.setattr(server, "probe_title", fake_probe_title)
    monkeypatch.setattr(server, "download_audio", fake_download_audio)
    monkeypatch.setattr(server, "process_song", fake_process_song)
    monkeypatch.setattr(server, "publish_song", lambda slug, cache_dir, public_dir: None)

    job = server.Job(id="job-1", url="https://example.com/watch?v=abc")
    server._run_job(job)

    assert job.status == "done"
    assert job.progress == 1.0
    assert job.slug == "my-song"
