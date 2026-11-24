"""Light (torch-free) speaker diarization: ECAPA-ONNX embeddings + clustering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import features as feat

log = logging.getLogger("overheard.diar")

MAX_WINDOW_S = 4.0  # long regions are split into windows, embeddings averaged
MIN_WINDOW_S = 0.75  # windows shorter than this are merged into the previous one


@dataclass
class Region:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerEmbedder:
    """192-d utterance embeddings from the ECAPA-TDNN ONNX graph."""

    def __init__(self, model_path: str, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(threads, 1)
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def embed_window(self, audio: np.ndarray) -> np.ndarray:
        span = np.asarray(audio, dtype=np.float32)
        if span.size < 960:  # ECAPA's dilated reflect-pad convs need >= 5-6 frames
            span = np.pad(span, (0, 960 - span.size))
        f = feat.features(span)
        out = self._sess.run(
            None,
            {
                "features": np.ascontiguousarray(f[None]),
                "feature_lens": np.array([f.shape[0]], np.float32),
            },
        )[0][0]
        return np.asarray(out, np.float32)

    def embed_region(self, audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        sr = 16_000
        lo, hi = int(start_s * sr), min(int(end_s * sr), audio.size)
        span = audio[lo:hi]
        dur = span.size / sr
        if dur <= MAX_WINDOW_S:
            vec = self.embed_window(span)
        else:
            win = int(MAX_WINDOW_S * sr)
            # center-aligned windows, no more than needed
            n = max(2, int(round(dur / (0.75 * MAX_WINDOW_S))))
            step = max(1, (span.size - win) // n)
            vecs = [
                self.embed_window(span[i : i + win]) for i in range(0, span.size - win + 1, step)
            ]
            if not vecs:
                vecs = [self.embed_window(span)]
            vec = np.mean(np.stack(vecs), axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / (norm if norm > 1e-9 else 1.0)


def embed_regions(
    embedder: SpeakerEmbedder, audio: np.ndarray, regions: list[Region]
) -> np.ndarray:
    if not regions:
        return np.zeros((0, 192), np.float32)
    return np.stack([embedder.embed_region(audio, r.start, r.end) for r in regions])


def cluster(
    embs: np.ndarray,
    threshold: float,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> np.ndarray:
    """Average-linkage agglomerative over cosine distance -> 0-based labels."""
    n = embs.shape[0]
    if n == 1:
        return np.zeros(1, np.int32)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sims = embs @ embs.T
    dist = np.clip(1.0 - sims, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=threshold, criterion="distance") - 1

    # honor explicit speaker count when the distance cut disagrees
    want = min_speakers if min_speakers and min_speakers == max_speakers else None
    if want and want > 1 and labels.max() + 1 != want:
        labels = fcluster(Z, t=want, criterion="maxclust") - 1
    elif max_speakers and labels.max() + 1 > max_speakers:
        labels = fcluster(Z, t=max_speakers, criterion="maxclust") - 1
    elif min_speakers and labels.max() + 1 < min_speakers:
        labels = fcluster(Z, t=min_speakers, criterion="maxclust") - 1
    return labels.astype(np.int32)


def relabel_first_appearance(labels: np.ndarray) -> np.ndarray:
    """Renumber speakers 0..k in order of first appearance (labels time-sorted)."""
    seen: dict[int, int] = {}
    out = []
    for lab in labels:
        seen.setdefault(int(lab), len(seen))
        out.append(seen[int(lab)])
    return np.asarray(out, np.int32)


def merge_turns(
    regions: list[Region], labels: np.ndarray, max_gap_s: float = 0.75
) -> list[tuple[float, float, int]]:
    """Adjacent same-speaker regions become one turn; tiny gaps bridged."""
    turns: list[list] = []
    for reg, lab in sorted(zip(regions, labels, strict=True), key=lambda t: t[0].start):
        lab = int(lab)
        if turns and turns[-1][2] == lab and reg.start - turns[-1][1] <= max_gap_s:
            turns[-1][1] = reg.end
        else:
            turns.append([reg.start, reg.end, lab])
    return [(start, end, speaker) for start, end, speaker in turns]


def assign_to_segments(turns: list[tuple[float, float, int]], segments: list[dict]) -> None:
    """Stamp each ASR segment with the dominant speaker over its overlap."""
    for seg in segments:
        best_lab, best_ov = None, 0.0
        for s, e, lab in turns:
            ov = min(seg["end"], e) - max(seg["start"], s)
            if ov > best_ov:
                best_ov, best_lab = ov, lab
        if best_lab is None and turns:
            # fall back to nearest turn in time
            mid = (seg["start"] + seg["end"]) / 2
            s, e, lab = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))
            best_lab = lab
        seg["speaker"] = f"SPEAKER_{best_lab:02d}" if best_lab is not None else None
