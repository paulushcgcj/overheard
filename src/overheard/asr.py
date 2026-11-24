"""Whisper transcription and speech-region detection (both torch-free)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .config import Settings
from .speakers import Region

log = logging.getLogger("overheard.asr")


def speech_regions(audio: np.ndarray, s: Settings) -> list[Region]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(
        audio,
        vad_options=VadOptions(
            threshold=s.vad_threshold,
            min_speech_duration_ms=s.vad_min_speech_ms,
            min_silence_duration_ms=s.vad_min_silence_ms,
            max_speech_duration_s=15.0,
            speech_pad_ms=100,
        ),
    )
    # returns begin/end in SAMPLES regardless of legacy kwargs; convert here
    return [Region(round(t["start"] / 16_000, 3), round(t["end"] / 16_000, 3)) for t in stamps]


class Transcriber:
    def __init__(self, s: Settings):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            s.whisper_model,
            device="cpu",
            compute_type=s.compute_type,
            cpu_threads=s.cpu_threads,
            download_root=str(s.whisper_dir),
        )

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        progress: Callable[[float], None] | None = None,
    ) -> tuple[list[dict], str, float]:
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            beam_size=5,
            condition_on_previous_text=True,
        )
        total = max(1.0, float(np.asarray(audio).size) / 16_000)
        out: list[dict] = []
        for seg in segments_iter:
            item = {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip(),
                "confidence": round(float(np.exp(seg.avg_logprob)), 4),
            }
            if word_timestamps and seg.words:
                item["words"] = [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(float(w.probability), 4),
                    }
                    for w in seg.words
                ]
            if progress is not None:
                progress(min(1.0, item["end"] / total))
            out.append(item)
        return out, (info.language or "unknown"), round(total, 3)
