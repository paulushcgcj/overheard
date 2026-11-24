# overheard

**Post audio. Get back *who said what, when* — plus vectors for every turn.**

A self-hosted FastAPI endpoint that does **transcription**, **speaker
diarization** and **text embeddings** in one container — no GPU required,
no cloud calls, no torch in the default image. Built and benchmarked on an
old 2-core/4-thread laptop server (Intel i7-7500U).

> Why not summaries too? Intentionally left out: this endpoint returns clean
> structured text, and any agent or LLM you already run does the summary/
> extraction step far better than a co-located tiny model on this hardware.
> Pipe `/v1/jobs/{id}/result` into your agent of choice.

![tests](https://github.com/paulushcgcj/overheard/actions/workflows/ci.yml/badge.svg)
![docker](https://github.com/paulushcgcj/overheard/actions/workflows/docker.yml/badge.svg)

## What it returns

| Block | Contents |
| --- | --- |
| `transcript` | plain merged text |
| `segments` | `{start, end, text, speaker, confidence, embedding?, words?}` |
| `speakers` | per-`SPEAKER_xx` speech time and turn count |
| `topics` | semantic groups (embedding-similarity boundaries) with centroids |
| `language` / `audio_duration_s` / timings | detection + processing metadata |

## Quick start (Docker)

```bash
docker compose up -d          # uses the example compose file, port 8005
# first request warms the model cache into the volume (~200-600 MB, once)

curl -F "file=@meeting.mp3" http://localhost:8005/v1/transcribe | jq .transcript
```

OpenAI-compatible drop-in (works with any `OPENAI_BASE_URL` client):

```bash
curl -F "file=@meeting.mp3" -F "response_format=verbose_json" \
  http://localhost:8005/v1/audio/transcriptions | jq '.segments[0]'
# {"start":0.0,"end":8.0,"text":"And so, my fellow Americans...","speaker":"SPEAKER_00","words":[...]}
```

Long recordings — async queue (recommended over 10 min of audio):

```bash
job=$(curl -s -F "file=@3h.m4a" "http://localhost:8005/v1/transcribe?async=true" | jq -r .job_id)
watch -n 5 "curl -s http://localhost:8005/v1/jobs/$job | jq '{status, stage, progress}'"
curl -s http://localhost:8005/v1/jobs/$job/result | jq .
```

## Images

| Tag | Diarization backend | Size | Notes |
| --- | --- | --- | --- |
| `:latest` | light: Silero VAD → ECAPA-ONNX → agglomerative clustering | ~600 MB | torch-free, no tokens needed |
| `:latest-accurate` | pyannote `community-1` (falls back to light) | ~3 GB | needs `HF_TOKEN` + agreeing to model conditions |

Both run CPU-only (CTranslate2 int8). Pull from `ghcr.io/paulushcgcj/overheard`.

## Measured speed (i7-7500U, 4 threads, int8)

| Model | Realtime factor | 1 h audio | RAM |
| --- | --- | --- | --- |
| `tiny` | ~12.6× | ~5 min | low |
| `base` | ~7.1× | ~8 min | low |
| `small` | ~2.4× | ~25 min | ~1.5 GB |

Diarization adds roughly the same order as one extra `base` pass; embeddings
are negligible. Expect ~30-45 min wall clock end-to-end for an hour of audio
with `small` on that class of machine. (An NVIDIA Maxwell 940MX/CC-5.x GPU is
**not** useful here: CUDA 13 and cuDNN 9 dropped pre-Turing architectures,
and 580 was the final driver branch — the CPU path is the only living one.)

## Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `OVERHEARD_WHISPER_MODEL` | `base` | tiny / base / small / medium / large-v3 / distil-* |
| `OVERHEARD_CPU_THREADS` | all cores | CTranslate2 threads (set to physical cores) |
| `OVERHEARD_DIARIZE` | `true` | run speaker diarization |
| `OVERHEARD_DIAR_BACKEND` | `auto` | `light` \| `pyannote` \| `auto` (pyannote if token & import) |
| `OVERHEARD_CLUSTER_THRESHOLD` | `0.62` | cosine distance cut for speaker clusters |
| `OVERHEARD_VAD_THRESHOLD` | `0.5` | Silero speech threshold |
| `OVERHEARD_MODEL_DIR` | `/data/models` | model cache (mount a volume!) |
| `OVERHEARD_DATA_DIR` | `/data/jobs` | job records and results |
| `OVERHEARD_API_KEY` | *(off)* | require `Authorization: Bearer` / `X-API-Key` |
| `OVERHEARD_MAX_UPLOAD_MB` | `512` | reject larger uploads |
| `OVERHEARD_RETENTION_DAYS` | `7` | prune job records/results |
| `HF_TOKEN` | | gated pyannote weights (accurate image only) |

Per-request knobs (form fields): `language`, `diarize`, `num_speakers`,
`word_timestamps`, `embeddings`, `topics`, `topic_min_sim`.

## API surface

| Endpoint | Description |
| --- | --- |
| `GET /healthz` | liveness, no auth |
| `GET /v1/status` | loaded models/queue depth |
| `POST /v1/transcribe` | upload; `?async=true` → `202 {job_id}` |
| `GET /v1/jobs` / `GET /v1/jobs/{id}` | list / status+progress |
| `GET /v1/jobs/{id}/result` | full JSON (409 while unfinished) |
| `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | lifecycle |
| `POST /v1/audio/transcriptions` | OpenAI-compatible; `json`/`text`/`srt`/`vtt`/`verbose_json` with `speaker` on segments |

Audio containers handled by PyAV (bundled ffmpeg): wav/mp3/m4a/flac/ogg/opus…

## Development

```bash
uv venv --python 3.12 && . .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q -m "not model"     # instant, no downloads
pytest -q                    # + tiny Whisper & ONNX models (~350 MB, cached)
ruff check src tests
```

The torch-free speaker front-end (`src/overheard/features.py`) reproduces
SpeechBrain's ECAPA fbank exactly (constant pad, periodic Hann, `10·log10`,
sentence-mean subtraction); the mel matrix is vendored as a generated asset
and validated to cosine ≥ 0.97 against the reference torch pipeline.
Third-party model credits: [assets/NOTICE.md](src/overheard/assets/NOTICE.md).

## Roadmap ideas

- VAD-aware streaming ingest for live microphones
- Speaker voiceprint enrollment (label `SPEAKER_00` → real names)
- Word-level diarization for fast interruption turns

## License

GPL-3.0-or-later. Model weights have their own licenses
(see NOTICE; Whisper/Silero MIT, ECAPA/MiniLM Apache-2.0, pyannote MIT-with-conditions).
