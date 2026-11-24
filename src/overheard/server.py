"""FastAPI server: sync endpoint, async job queue, and an OpenAI-compatible
`/v1/audio/transcriptions` (verbose_json carries speaker turns)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .config import load_settings
from .jobs import JobManager
from .pipeline import Engine, JobParams

log = logging.getLogger("overheard.server")

MAX_BYTES_DEFAULT = None  # resolved from settings at startup


def _fmt_ts(t: float, comma: bool) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def _to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        who = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines += [
            str(i),
            f"{_fmt_ts(seg['start'], True)} --> {_fmt_ts(seg['end'], True)}",
            f"{who}{seg['text']}",
            "",
        ]
    return "\n".join(lines)


def _to_vtt(segments: list[dict]) -> str:
    body = ["WEBVTT", ""]
    for seg in segments:
        who = f"<v {seg['speaker']}> " if seg.get("speaker") else ""
        body += [
            f"{_fmt_ts(seg['start'], False)} --> {_fmt_ts(seg['end'], False)}",
            f"{who}{seg['text']}",
            "",
        ]
    return "\n".join(body)


def create_app() -> FastAPI:
    settings = load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = Engine(settings)
        app.state.jobs = JobManager(app.state.engine, settings)
        log.info(
            "overheard %s ready (whisper=%s, diar=%s)",
            __version__,
            settings.whisper_model,
            settings.diar_backend,
        )
        yield

    app = FastAPI(title="overheard", version=__version__, lifespan=lifespan)

    async def auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not settings.api_key:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip() or (x_api_key or "")
        if supplied != settings.api_key:
            raise HTTPException(401, "missing or invalid API key")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/v1/status", dependencies=[Depends(auth)])
    async def status() -> dict:
        jm: JobManager = app.state.jobs
        pending = [j for j in jm.list(200) if j["status"] in ("queued", "running")]
        return {
            "version": __version__,
            "whisper_model": settings.whisper_model,
            "compute_type": settings.compute_type,
            "cpu_threads": settings.cpu_threads,
            "diarization_backend": settings.diar_backend,
            "text_embed_model": settings.text_embed_model,
            "model_dir": str(settings.model_dir),
            "jobs_active": len(pending),
        }

    def _params(language, diarize, num_speakers, word_timestamps, embeddings, topics) -> JobParams:
        return JobParams(
            language=language,
            word_timestamps=word_timestamps,
            diarize=diarize if diarize is not None else None,
            num_speakers=num_speakers,
            embeddings=embeddings,
            topics=topics,
        )

    async def _read_upload(file: UploadFile) -> bytes:
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read()
        if len(data) == 0:
            raise HTTPException(422, "empty upload")
        if len(data) > limit:
            raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB limit")
        return data

    @app.post("/v1/transcribe", dependencies=[Depends(auth)])
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
        diarize: bool | None = Form(default=None),
        num_speakers: int | None = Form(default=None),
        word_timestamps: bool = Form(default=False),
        embeddings: bool = Form(default=True),
        topics: bool = Form(default=True),
        is_async: bool = Query(default=False, alias="async"),
    ) -> JSONResponse:
        data = await _read_upload(file)
        params = _params(language, diarize, num_speakers, word_timestamps, embeddings, topics)
        if is_async:
            job_id = app.state.jobs.submit(data, params, file.filename)
            return JSONResponse(
                {"job_id": job_id, "status_url": f"/v1/jobs/{job_id}"}, status_code=202
            )
        result = await _run_sync(app, data, params)
        return JSONResponse(result)

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
    async def openai_transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default="whisper-1"),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="verbose_json"),
        temperature: float = Form(default=0.0),
        diarize: bool | None = Form(default=None),
        num_speakers: int | None = Form(default=None),
    ):
        data = await _read_upload(file)
        params = JobParams(
            language=language,
            diarize=diarize,
            num_speakers=num_speakers,
            word_timestamps=response_format == "verbose_json",
            embeddings=False,
            topics=False,
        )
        result = await _run_sync(app, data, params)
        segs = result["segments"]
        if response_format == "text":
            return PlainTextResponse("\n".join(s["text"] for s in segs))
        if response_format == "srt":
            return PlainTextResponse(_to_srt(segs))
        if response_format == "vtt":
            return PlainTextResponse(_to_vtt(segs))
        body = {
            "task": "transcribe",
            "language": result["language"],
            "duration": result["audio_duration_s"],
            "text": result["transcript"],
        }
        if response_format == "json":
            return JSONResponse(body)
        body.update(
            {
                "segments": segs,
                "speakers": result["speakers"],
                "diarization_backend": result["diarization_backend"],
            }
        )
        return JSONResponse(body)

    @app.get("/v1/jobs", dependencies=[Depends(auth)])
    async def jobs(limit: int = Query(default=50, le=200)) -> dict:
        return {"jobs": app.state.jobs.list(limit)}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    async def job_status(job_id: str) -> dict:
        rec = app.state.jobs.get(job_id)
        if rec is None:
            raise HTTPException(404, "no such job")
        return rec

    @app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(auth)])
    async def job_result(job_id: str):
        rec = app.state.jobs.get(job_id)
        if rec is None:
            raise HTTPException(404, "no such job")
        if rec["status"] != "succeeded":
            raise HTTPException(409, f"job is {rec['status']}")
        return JSONResponse(app.state.jobs.result(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
    async def job_cancel(job_id: str) -> dict:
        if not app.state.jobs.cancel(job_id):
            raise HTTPException(409, "job already finished or unknown")
        return {"id": job_id, "status": "canceled"}

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    async def job_delete(job_id: str) -> dict:
        if not app.state.jobs.delete(job_id):
            raise HTTPException(404, "no such job")
        return {"id": job_id, "deleted": True}

    return app


async def _run_sync(app, data: bytes, params: JobParams) -> dict:
    """Run on the shared single-worker pool (never oversubscribes the CPU)
    without blocking the event loop."""
    import asyncio

    engine: Engine = app.state.engine
    jm: JobManager = app.state.jobs
    t0 = time.monotonic()
    future = jm.executor.submit(engine.process, data, params)
    try:
        result = await asyncio.wrap_future(future)
    except ValueError as exc:  # decode failures surface as 4xx, not a server fault
        raise HTTPException(422, str(exc)) from exc
    result["server_time_s"] = round(time.monotonic() - t0, 2)
    return result


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    host = __import__("os").environ.get("OVERHEARD_HOST", "0.0.0.0")
    port = int(__import__("os").environ.get("OVERHEARD_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
