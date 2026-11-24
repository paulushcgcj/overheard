"""File-backed async job queue over the pipeline engine.

One worker by default: this box is CPU-bound and a 2-core laptop does better
running jobs serially than thrashing. Job metadata + results survive restarts
(orphaned ``running`` jobs are marked failed on boot).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import Settings
from .pipeline import Engine, JobParams

log = logging.getLogger("overheard.jobs")

TERMINAL = ("succeeded", "failed", "canceled")


class JobManager:
    def __init__(self, engine: Engine, settings: Settings):
        self.engine = engine
        self.settings = settings
        self.dir = settings.data_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.worker_threads), thread_name_prefix="ovh"
        )
        self._load_existing()

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Shared bounded pool so sync and async requests never oversubscribe
        the CPU."""
        return self._executor

    # -- persistence ----------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _result_path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.result.json"

    def _write(self, rec: dict) -> None:
        tmp = self._path(rec["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(self._path(rec["id"]))

    def _load_existing(self) -> None:
        cutoff = time.time() - self.settings.retention_days * 86_400
        for p in sorted(self.dir.glob("*.json")):
            if p.name.endswith(".result.json") or p.name.endswith(".tmp"):
                continue
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if rec.get("created_ts", 0) < cutoff:
                p.unlink(missing_ok=True)
                self._result_path(rec["id"]).unlink(missing_ok=True)
                continue
            if rec.get("status") == "running":
                rec.update(
                    status="failed", error="interrupted by server restart", updated_ts=time.time()
                )
                self._write(rec)
            self._jobs[rec["id"]] = rec

    # -- lifecycle --------------------------------------------------------------

    def submit(self, data: bytes, params: JobParams, filename: str | None) -> str:
        job_id = uuid.uuid4().hex[:12]
        rec = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "filename": filename,
            "params": params.__dict__,
            "created_ts": time.time(),
            "updated_ts": time.time(),
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = rec
            self._write(rec)
        # bytes handed off to worker; copied to disk so memory stays lean
        payload = self.dir / f"{job_id}.audio"
        payload.write_bytes(data)
        self._executor.submit(self._run, job_id, payload)
        return job_id

    def _progress_cb(self, job_id: str):
        last = {"t": 0.0}

        def cb(p: float, stage: str) -> None:
            now = time.monotonic()
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None or rec["status"] == "canceled":
                    return
                if now - last["t"] < 1.0 and p < 1.0:
                    rec["progress"], rec["stage"] = round(p, 3), stage
                    return
                last["t"] = now
                rec.update(progress=round(p, 3), stage=stage, updated_ts=time.time())
                self._write(rec)

        return cb

    def _run(self, job_id: str, payload: Path) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            if rec["status"] == "canceled":
                payload.unlink(missing_ok=True)
                return
            rec.update(status="running", stage="starting", updated_ts=time.time())
            self._write(rec)
        try:
            params = JobParams(**rec["params"])
            result = self.engine.process(
                payload.read_bytes(), params, progress=self._progress_cb(job_id)
            )
            self._result_path(job_id).write_text(json.dumps(result, ensure_ascii=False))
            with self._lock:
                rec = self._jobs[job_id]
                if rec["status"] != "canceled":
                    rec.update(
                        status="succeeded", stage="done", progress=1.0, updated_ts=time.time()
                    )
                    self._write(rec)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            with self._lock:
                rec = self._jobs[job_id]
                rec.update(status="failed", error=str(exc)[:400], updated_ts=time.time())
                self._write(rec)
        finally:
            payload.unlink(missing_ok=True)

    # -- queries ------------------------------------------------------------------

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return dict(rec) if rec else None

    def result(self, job_id: str) -> dict | None:
        p = self._result_path(job_id)
        return json.loads(p.read_text()) if p.exists() else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            recs = sorted(self._jobs.values(), key=lambda r: r["created_ts"], reverse=True)
            return [dict(r) for r in recs[:limit]]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or rec["status"] in TERMINAL:
                return False
            rec.update(status="canceled", stage="canceled", updated_ts=time.time())
            self._write(rec)
            return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.pop(job_id, None)
        if rec is None:
            return False
        self._path(job_id).unlink(missing_ok=True)
        self._result_path(job_id).unlink(missing_ok=True)
        (self.dir / f"{job_id}.audio").unlink(missing_ok=True)
        return True
