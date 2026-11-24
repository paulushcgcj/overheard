"""Queue semantics with a stubbed engine: no downloads."""

from __future__ import annotations

import json
import time

import pytest

from overheard.config import Settings
from overheard.jobs import JobManager
from overheard.pipeline import JobParams


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def process(self, data, params, progress=None):
        self.calls += 1
        if progress:
            progress(0.5, "fake")
        if b"boom" in data:
            raise RuntimeError("boom happened")
        return {"transcript": "ok", "segments": [], "params_echo": params.__dict__}


@pytest.fixture()
def jm(tmp_path):
    s = Settings(model_dir=tmp_path / "m", data_dir=tmp_path / "j")
    eng = FakeEngine()
    return JobManager(eng, s), eng


def wait(jm: JobManager, job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = jm.get(job_id)
        if rec["status"] in ("succeeded", "failed", "canceled"):
            return rec
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {rec['status']}")


def test_submit_succeeds_and_persists(jm):
    mgr, eng = jm
    job_id = mgr.submit(b"audio", JobParams(), "a.wav")
    rec = wait(mgr, job_id)
    assert rec["status"] == "succeeded"
    assert mgr.result(job_id)["transcript"] == "ok"
    assert mgr._path(job_id).exists()
    assert not (mgr.dir / f"{job_id}.audio").exists(), "audio should be cleaned up"


def test_failure_records_error(jm):
    mgr, _ = jm
    job_id = mgr.submit(b"boom", JobParams(), "bad.wav")
    rec = wait(mgr, job_id)
    assert rec["status"] == "failed"
    assert "boom" in rec["error"]


def test_restart_marks_orphans_failed(jm, tmp_path):
    mgr, _ = jm
    job_id = mgr.submit(b"audio", JobParams(), "a.wav")
    wait(mgr, job_id)
    # forge a "running" record, then reboot
    rec = mgr._path(job_id).read_text()
    data = json.loads(rec)
    data["status"] = "running"
    mgr._path(job_id).write_text(json.dumps(data))
    mgr2 = JobManager(FakeEngine(), mgr.settings)
    assert mgr2.get(job_id)["status"] == "failed"


def test_delete_and_list(jm):
    mgr, _ = jm
    a = mgr.submit(b"x", JobParams(), None)
    wait(mgr, a)
    assert any(j["id"] == a for j in mgr.list())
    assert mgr.delete(a)
    assert mgr.get(a) is None
    assert mgr.result(a) is None
