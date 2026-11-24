# syntax=docker/dockerfile:1
#
# Two build targets:
#   light    (default) — torch-free: CTranslate2 + ONNX. ~600 MB image.
#   accurate           — adds CPU torch + pyannote community-1 diarization
#                        (needs HF_TOKEN at runtime). ~2.5-3 GB image.
#
# docker build --target light    -t overheard:latest .
# docker build --target accurate -t overheard:latest-accurate .

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OVERHEARD_MODEL_DIR=/data/models \
    OVERHEARD_DATA_DIR=/data/jobs \
    OVERHEARD_HOST=0.0.0.0 \
    OVERHEARD_PORT=8000
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

FROM base AS light
RUN --mount=type=cache,target=/root/.cache/pip pip install "."
RUN groupadd -r overheard \
 && useradd -r -g overheard -d /data -s /usr/sbin/nologin overheard \
 && mkdir -p /data/models /data/jobs \
 && chown -R overheard:overheard /data /app
USER overheard
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"
CMD ["python", "-m", "overheard.server"]

FROM base AS accurate
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[pyannote]"
RUN groupadd -r overheard \
 && useradd -r -g overheard -d /data -s /usr/sbin/nologin overheard \
 && mkdir -p /data/models /data/jobs \
 && chown -R overheard:overheard /data /app
USER overheard
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"
CMD ["python", "-m", "overheard.server"]
