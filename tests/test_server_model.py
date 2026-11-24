"""HTTP surface tests (tiny model)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def client():
    from dataclasses import replace

    import overheard.server as srv
    from overheard.server import create_app

    orig_load = srv.load_settings

    def patched():
        s = orig_load()
        return replace(s, whisper_model="tiny", diar_backend="light", cpu_threads=4)

    srv.load_settings = patched
    try:
        app = create_app()
        with TestClient(app) as c:
            yield c
    finally:
        srv.load_settings = orig_load


def test_healthz(client):
    assert client.get("/healthz").status_code == 200


def test_status(client):
    body = client.get("/v1/status").json()
    assert body["whisper_model"] == "tiny"
    assert body["version"]


def test_sync_transcribe(client, jfk_bytes):
    r = client.post("/v1/transcribe", files={"file": ("jfk.wav", jfk_bytes, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "ask not" in body["transcript"].lower()
    assert body["segments"][0]["embedding"]
    assert body["topics"]


def test_async_job_flow(client, jfk_bytes):
    r = client.post(
        "/v1/transcribe?async=true",
        files={"file": ("jfk.wav", jfk_bytes, "audio/wav")},
        data={"embeddings": "false", "topics": "false"},
    )
    assert r.status_code == 202 and "job_id" in r.json()
    job_id = r.json()["job_id"]
    deadline = time.time() + 120
    rec = None
    while time.time() < deadline:
        rec = client.get(f"/v1/jobs/{job_id}").json()
        if rec["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    assert rec["status"] == "succeeded", rec
    result = client.get(f"/v1/jobs/{job_id}/result").json()
    assert "ask not" in result["transcript"].lower()
    assert "embedding" not in result["segments"][0]


def test_openai_verbose_and_srt(client, jfk_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("jfk.wav", jfk_bytes, "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "ask not" in body["text"].lower()
    assert body["segments"][0]["words"]
    assert body["duration"] > 5

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("jfk.wav", jfk_bytes, "audio/wav")},
        data={"response_format": "srt"},
    )
    assert r.status_code == 200
    assert "-->" in r.text
    assert "SPEAKER" in r.text or "ask not" in r.text


def test_rejects_empty_upload(client):
    r = client.post("/v1/transcribe", files={"file": ("x.wav", b"", "audio/wav")})
    assert r.status_code == 422


def test_rejects_garbage_audio(client):
    r = client.post("/v1/transcribe", files={"file": ("x.bin", b"not audio at all", "audio/wav")})
    assert r.status_code == 422


def test_auth_enforced(monkeypatch, jfk_bytes):

    monkeypatch.setenv("OVERHEARD_API_KEY", "secret123")
    from overheard.server import create_app

    with TestClient(create_app()) as c:
        assert c.get("/v1/status").status_code == 401
        assert c.get("/healthz").status_code == 200
        r = c.get("/v1/status", headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200
