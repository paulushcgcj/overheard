"""Text embeddings + cheap semantic topic segmentation (MiniLM ONNX)."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("overheard.embed")


class TextEmbedder:
    def __init__(self, onnx_path: str, tokenizer_path: str, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(threads, 1)
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tok.enable_truncation(max_length=512)
        self.dim = 384

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        outs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = self._tok.encode_batch(chunk)
            input_ids = np.stack([e.ids for e in enc]).astype(np.int64)
            attention = np.stack([e.attention_mask for e in enc]).astype(np.int64)
            feeds = {"input_ids": input_ids, "attention_mask": attention}
            # only pass token_type_ids if the graph declares them
            names = {inp.name for inp in self._sess.get_inputs()}
            if "token_type_ids" in names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            result = self._sess.run(None, feeds)
            hidden = np.asarray(result[0], np.float32)  # (B, T, H)
            mask = attention[..., None].astype(np.float32)
            summed = (hidden * mask).sum(axis=1)
            counts = np.maximum(mask.sum(axis=1), 1e-9)
            vecs = summed / counts
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
            outs.append(vecs)
        return np.concatenate(outs, axis=0).astype(np.float32)


def split_topics(segments: list[dict], min_sim: float = 0.45, min_segments: int = 2) -> list[dict]:
    """Group consecutive utterances into topics via cosine similarity of
    their embeddings; boundaries are local minima below ``min_sim``."""
    vecs = [seg["_embedding"] for seg in segments if "_embedding" in seg]
    if len(segments) < min_segments or len(vecs) != len(segments):
        return (
            [
                {
                    "start": segments[0]["start"] if segments else 0.0,
                    "end": segments[-1]["end"] if segments else 0.0,
                    "text": " ".join(s["text"] for s in segments),
                }
            ]
            if segments
            else []
        )
    boundaries = [0]
    for i in range(1, len(segments)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        dur = segments[i]["start"] - segments[i - 1]["end"]
        if sim < min_sim or dur > 30.0:
            boundaries.append(i)
    boundaries.append(len(segments))
    topics = []
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=False):
        text = " ".join(s["text"] for s in segments[a:b]).strip()
        if not text:
            continue
        centroid = np.mean(np.stack(vecs[a:b]), axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        topics.append(
            {
                "start": segments[a]["start"],
                "end": segments[b - 1]["end"],
                "speakers": sorted({s.get("speaker") for s in segments[a:b] if s.get("speaker")}),
                "text": text,
                "_centroid": centroid.astype(np.float32),
            }
        )
    return topics
