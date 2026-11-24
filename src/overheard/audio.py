"""Audio decoding to 16 kHz mono float32.

Uses PyAV through faster-whisper's own decoder so mp3/flac/m4a/ogg/wav all
land on one code path with ffmpeg bundled in the wheel (no system ffmpeg).
"""

from __future__ import annotations

import io

import numpy as np

TARGET_SR = 16_000


def decode(data: bytes) -> np.ndarray:
    """Decode arbitrary container audio into 16 kHz mono float32."""
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=TARGET_SR)
    except Exception as exc:  # noqa: BLE001 - normalize to a clear error for API users
        raise ValueError(f"could not decode audio file: {exc}") from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:  # multi-channel folded by decode_audio, but be safe
        audio = audio.mean(axis=1).astype(np.float32)
    if audio.size == 0 or np.all(np.abs(audio) < 1e-6):
        raise ValueError("audio track is empty or silent")
    return audio


def duration(audio: np.ndarray) -> float:
    return round(float(audio.size) / TARGET_SR, 3)
