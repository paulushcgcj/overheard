"""Pipeline orchestration: audio bytes in -> transcript + speakers + vectors out."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import audio as audio_mod
from . import store
from .asr import Transcriber, speech_regions
from .config import Settings
from .embed import TextEmbedder, split_topics
from .speakers import (
    Region,
    SpeakerEmbedder,
    assign_to_segments,
    cluster,
    embed_regions,
    merge_turns,
    relabel_first_appearance,
)

log = logging.getLogger("overheard.pipeline")


@dataclass
class JobParams:
    language: str | None = None
    word_timestamps: bool = False
    diarize: bool | None = None  # None -> settings default
    num_speakers: int | None = None
    embeddings: bool = True
    topics: bool = True
    topic_min_sim: float = 0.45

    @classmethod
    def from_form(cls, form: dict) -> JobParams:
        def b(key, default=False):
            v = form.get(key)
            if v is None:
                return default
            return str(v).lower() in {"1", "true", "yes", "on"}

        ns = form.get("num_speakers")
        return cls(
            language=(form.get("language") or None),
            word_timestamps=b("word_timestamps"),
            diarize=b("diarize", True) if "diarize" in form else None,
            num_speakers=int(ns) if ns not in (None, "", 0, "0") else None,
            embeddings=b("embeddings", True),
            topics=b("topics", True),
            topic_min_sim=float(form.get("topic_min_sim", 0.45)),
        )


class Engine:
    """Loads models once; safe to share across requests (inference is serialized
    by the single-worker job queue)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        t0 = time.monotonic()
        self.asr = Transcriber(self.settings)
        self._speakers: SpeakerEmbedder | None = None
        self._text: TextEmbedder | None = None
        self._pyannote = None
        self._pyannote_failed = False
        self.ready_at = round(time.monotonic() - t0, 2)
        log.info("ASR model '%s' loaded in %.1fs", self.settings.whisper_model, self.ready_at)

    # -- lazy components ------------------------------------------------------

    def speaker_embedder(self) -> SpeakerEmbedder:
        if self._speakers is None:
            path = store.ecapa_model(self.settings)
            self._speakers = SpeakerEmbedder(str(path), threads=self.settings.cpu_threads)
        return self._speakers

    def text_embedder(self) -> TextEmbedder:
        if self._text is None:
            onnx, tok = store.text_model_paths(self.settings)
            self._text = TextEmbedder(str(onnx), str(tok), threads=self.settings.cpu_threads)
        return self._text

    def _get_pyannote(self):
        if self._pyannote is None and not self._pyannote_failed:
            try:
                import os

                import torch
                from pyannote.audio import Pipeline

                token = os.environ.get("HF_TOKEN")
                if not token:
                    raise RuntimeError("HF_TOKEN not set")
                pipe = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1", token=token
                )
                self._pyannote = pipe.to(torch.device("cpu"))
            except Exception as exc:  # noqa: BLE001
                self._pyannote_failed = True
                log.warning("pyannote backend unavailable (%s); using light backend", exc)
        return self._pyannote

    # -- main entry ------------------------------------------------------------

    def process(
        self, data: bytes, params: JobParams, progress: Callable[[float, str], None] | None = None
    ) -> dict:
        s = self.settings
        t_start = time.monotonic()

        def stage(p: float, name: str) -> None:
            if progress:
                progress(p, name)

        stage(0.0, "decoding")
        wav = audio_mod.decode(data)
        dur = audio_mod.duration(wav)

        stage(0.05, "transcribing")
        segments, language, _ = self.asr.transcribe(
            wav,
            language=params.language,
            word_timestamps=params.word_timestamps,
            vad_filter=True,
            progress=lambda frac: stage(0.05 + 0.55 * frac, "transcribing"),
        )

        speakers_used = "none"
        turns: list[tuple[float, float, int]] = []
        do_diar = s.diarize if params.diarize is None else params.diarize
        if do_diar and segments:
            stage(0.62, "diarizing")
            turns, speakers_used = self._diarize(wav, params)
            if turns:
                assign_to_segments(turns, segments)
        else:
            for seg in segments:
                seg["speaker"] = None

        stage(0.82, "embedding")
        topics = []
        if params.embeddings:
            te = self.text_embedder()
            vecs = te.embed([seg["text"] for seg in segments])
            for seg, v in zip(segments, vecs, strict=True):
                seg["_embedding"] = v
            if params.topics and segments:
                topics = split_topics(segments, min_sim=params.topic_min_sim)

        stage(0.92, "finalizing")
        result = self._assemble(
            segments, topics, turns, language, dur, speakers_used, t_start, params
        )
        stage(1.0, "done")
        return result

    def _diarize(self, wav: np.ndarray, params: JobParams):
        s = self.settings
        want_pyannote = s.diar_backend in ("auto", "pyannote")
        if want_pyannote:
            pipe = self._get_pyannote()
            if pipe is not None:
                try:
                    import torch as _torch

                    audio_in = {
                        "waveform": _torch.from_numpy(wav.copy()).unsqueeze(0),
                        "sample_rate": 16_000,
                    }
                    ann = (
                        pipe(audio_in, num_speakers=params.num_speakers)
                        if params.num_speakers
                        else pipe(audio_in)
                    )
                    turns = []
                    for seg, _, lab in ann.itertracks(yield_label=True):
                        turns.append(
                            (
                                round(seg.start, 3),
                                round(seg.end, 3),
                                int(str(lab).rsplit("_", 1)[-1]),
                            )
                        )
                    turns.sort(key=lambda t: t[0])
                    # pyannote labels are already SPEAKER_xx style; normalize order
                    labels = sorted({t[2] for t in turns})
                    remap = {old: new for new, old in enumerate(labels)}
                    turns = [(a, b, remap[c]) for a, b, c in turns]
                    return merge_turns_turns(turns), "pyannote-community-1"
                except Exception as exc:  # noqa: BLE001
                    log.warning("pyannote run failed (%s); falling back to light backend", exc)

        # light backend
        regions: list[Region] = speech_regions(wav, s)
        regions = [r for r in regions if r.duration >= 0.3]
        if not regions:
            return [], "light"
        emb = self.speaker_embedder()
        vectors = embed_regions(emb, wav, regions)
        min_s = max_s = params.num_speakers
        labels = cluster(
            vectors, threshold=s.cluster_threshold, min_speakers=min_s, max_speakers=max_s
        )
        labels = relabel_first_appearance(labels)
        order = sorted(range(len(regions)), key=lambda i: regions[i].start)
        r_sorted = [regions[i] for i in order]
        t_sorted = labels[order]
        return merge_turns(r_sorted, t_sorted), "light"

    def _assemble(self, segments, topics, turns, language, duration, backend, t_start, params):
        s = self.settings
        prec = s.embedding_precision
        for seg in segments:
            vec = seg.pop("_embedding", None)
            if params.embeddings and vec is not None:
                seg["embedding"] = [round(float(x), prec) for x in vec]
        for topic in topics:
            cent = topic.pop("_centroid", None)
            if cent is not None:
                topic["embedding"] = [round(float(x), prec) for x in cent]
        speaker_ids = sorted({t[2] for t in turns})
        speakers = []
        for lab in speaker_ids:
            spans = [(a, b) for a, b, c in turns if c == lab]
            speakers.append(
                {
                    "id": f"SPEAKER_{lab:02d}",
                    "total_speech_s": round(sum(b - a for a, b in spans), 2),
                    "turns": len(spans),
                }
            )
        return {
            "language": language,
            "audio_duration_s": duration,
            "processing_time_s": round(time.monotonic() - t_start, 2),
            "diarization_backend": backend if turns or backend != "light" else backend,
            "speakers": speakers,
            "segments": segments,
            "topics": topics,
            "transcript": " ".join(seg["text"] for seg in segments).strip(),
        }


def merge_turns_turns(turns: list[tuple[float, float, int]], max_gap_s: float = 0.75):
    """merge_turns variant operating on already-materialized turns."""
    out: list[list] = []
    for start, end, lab in turns:
        if out and out[-1][2] == lab and start - out[-1][1] <= max_gap_s:
            out[-1][1] = end
        else:
            out.append([start, end, lab])
    return [(a, b, c) for a, b, c in out]
