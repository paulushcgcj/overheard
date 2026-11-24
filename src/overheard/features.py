"""SpeechBrain-compatible fbank features, pure numpy (no torch).

Reproduces ``speechbrain/spkrec-ecapa-voxceleb``'s ComputeFeatures exactly:

* framing: 400-sample Hann window (periodic), 160 hop, constant (zero) padding
* FFT: rfft(n=400) -> 201 bins, power spectrum |X|**2
* mel: frozen 80x201 matrix (asset, measured from speechbrain's runtime
  filterbank), ``mel = power @ M.T``
* dB floor: ``10 * log10(max(mel, 1e-10))``
* per-utterance sentence mean subtraction

Verified against upstream speechbrain embeddings: cosine 0.97-0.99 per clip,
cross-speaker separability within 0.005 of the reference (see tests).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

N_FFT = 400
HOP = 160
N_BINS = N_FFT // 2 + 1  # 201
_ASSET = Path(__file__).parent / "assets" / "fbank_speechbrain_201x80_f32.bin"


@lru_cache(maxsize=1)
def mel_matrix() -> np.ndarray:
    return np.fromfile(_ASSET, dtype=np.float32).reshape(N_BINS, 80)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    k = np.arange(N_FFT)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / N_FFT)).astype(np.float64)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """(T, 80) log-mel dB frames for 16 kHz mono float32 input."""
    x = np.asarray(audio, dtype=np.float64).ravel()
    padded = np.pad(x, N_FFT // 2, mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP] * _window()
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = power @ mel_matrix()
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def features(audio: np.ndarray) -> np.ndarray:
    """Normalized fbank input for the ECAPA ONNX: log-mel minus sentence mean."""
    db = log_mel(audio)
    return (db - db.mean(axis=0)).astype(np.float32)
