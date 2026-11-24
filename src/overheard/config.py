"""Environment-driven configuration.

Every knob is a plain environment variable so the same image runs unchanged
in Docker. Values are parsed once at startup into a frozen ``Settings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _default_model_dir() -> Path:
    override = _env("OVERHEARD_MODEL_DIR")
    if override:
        return Path(override)
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return data / "models"
    return Path.home() / ".cache" / "overheard"


@dataclass(frozen=True)
class Settings:
    # --- models -------------------------------------------------------------
    whisper_model: str = field(default_factory=lambda: _env("OVERHEARD_WHISPER_MODEL", "base"))
    compute_type: str = field(default_factory=lambda: _env("OVERHEARD_COMPUTE_TYPE", "int8"))
    cpu_threads: int = field(default_factory=lambda: _env_int("OVERHEARD_CPU_THREADS", 0))
    text_embed_model: str = field(
        default_factory=lambda: _env("OVERHEARD_TEXT_EMBED_MODEL", "Xenova/all-MiniLM-L6-v2")
    )
    model_dir: Path = field(default_factory=_default_model_dir)

    # --- diarization ----------------------------------------------------------
    diarize: bool = field(default_factory=lambda: _env_bool("OVERHEARD_DIARIZE", True))
    # auto: pyannote when HF_TOKEN present and importable, else light
    diar_backend: str = field(default_factory=lambda: _env("OVERHEARD_DIAR_BACKEND", "auto"))
    cluster_threshold: float = field(
        default_factory=lambda: _env_float("OVERHEARD_CLUSTER_THRESHOLD", 0.62)
    )
    min_speakers: int | None = field(default_factory=lambda: _opt_int("OVERHEARD_MIN_SPEAKERS"))
    max_speakers: int | None = field(default_factory=lambda: _opt_int("OVERHEARD_MAX_SPEAKERS"))

    # --- VAD ------------------------------------------------------------------
    vad_threshold: float = field(default_factory=lambda: _env_float("OVERHEARD_VAD_THRESHOLD", 0.5))
    vad_min_speech_ms: int = field(
        default_factory=lambda: _env_int("OVERHEARD_VAD_MIN_SPEECH_MS", 250)
    )
    vad_min_silence_ms: int = field(
        default_factory=lambda: _env_int("OVERHEARD_VAD_MIN_SILENCE_MS", 400)
    )

    # --- server ----------------------------------------------------------------
    api_key: str = field(default_factory=lambda: _env("OVERHEARD_API_KEY"))
    data_dir: Path = field(
        default_factory=lambda: Path(
            _env("OVERHEARD_DATA_DIR", str(_default_model_dir().parent / "jobs"))
        )
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("OVERHEARD_MAX_UPLOAD_MB", 512))
    retention_days: int = field(default_factory=lambda: _env_int("OVERHEARD_RETENTION_DAYS", 7))
    worker_threads: int = field(default_factory=lambda: _env_int("OVERHEARD_WORKER_THREADS", 1))
    # include 384-d vectors inline in responses when requested; controls the
    # float rounding written into JSON payloads
    embedding_precision: int = field(
        default_factory=lambda: _env_int("OVERHEARD_EMBED_PRECISION", 5)
    )

    @property
    def hf_cache_dir(self) -> Path:
        return self.model_dir / "hf"

    @property
    def whisper_dir(self) -> Path:
        return self.model_dir / "whisper"


def _opt_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def load_settings() -> Settings:
    s = Settings()
    if s.cpu_threads <= 0:
        # dataclass is frozen; rebuild with detected core count
        s = Settings(**{**s.__dict__, "cpu_threads": (os.cpu_count() or 4)})
    return s
