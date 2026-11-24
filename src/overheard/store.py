"""Model asset resolution with an on-disk cache.

Whisper models are fetched by faster-whisper itself (``download_root``);
everything else downloads through ``huggingface_hub`` into a stable folder.
The image bundles no model weights: first request on a fresh volume pays a
one-off download (kept warm by mounting the data dir).
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings

# Vendored ONNX export of speechbrain ECAPA-TDNN (Apache-2.0).
ECAPA_REPO = "vedk00/ecapa-voxceleb-speaker-embedding-onnx"
ECAPA_FILE = "model/ecapa-speaker-v1.onnx"

TEXT_REPO = "Xenova/all-MiniLM-L6-v2"
TEXT_ONNX = "onnx/model.onnx"
TEXT_TOKENIZER = "tokenizer.json"


def _download(repo: str, filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
    )


def ecapa_model(settings: Settings) -> Path:
    return _download(ECAPA_REPO, ECAPA_FILE, settings.hf_cache_dir)


def text_model_paths(settings: Settings) -> tuple[Path, Path]:
    onnx = _download(settings.text_embed_model, TEXT_ONNX, settings.hf_cache_dir)
    tok = _download(settings.text_embed_model, TEXT_TOKENIZER, settings.hf_cache_dir)
    return onnx, tok
