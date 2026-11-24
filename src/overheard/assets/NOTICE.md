# Third-party model attribution

overheard bundles or downloads the following models at runtime. No model
weights are redistributed in this repository except `assets/fbank_speechbrain_201x80_f32.bin`,
an 80-band mel filterbank matrix (numerical metadata, not model weights)
generated against SpeechBrain's ECAPA recipe (see below).

| Component | Upstream | License |
| --- | --- | --- |
| ASR models (Whisper via CTranslate2) | SYSTRAN/faster-whisper, OpenAI Whisper | MIT |
| VAD (Silero VAD v6, ONNX) | snakers4/silero-vad | MIT |
| Speaker embeddings (ECAPA-TDNN ONNX, downloaded first run) | vedk00/ecapa-voxceleb-speaker-embedding-onnx, converted from speechbrain/spkrec-ecapa-voxceleb | Apache-2.0 |
| fbank matrix asset | reproduced from speechbrain/processing/features.py Filterbank | Apache-2.0 (SpeechBrain) |
| Text embeddings (all-MiniLM-L6-v2 ONNX, downloaded first run) | Xenova/all-MiniLM-L6-v2, from sentence-transformers | Apache-2.0 |
| Optional accurate diarization | pyannote/speaker-diarization-community-1 | MIT (gated, needs HF token) |

The fbank asset was produced by probing the SpeechBrain `spkrec-ecapa-voxceleb`
pipeline's actual runtime filterbank with unit-energy one-hot spectra, then
inverting the dB step; the resulting matrix reproduces upstream features to
cosine >= 0.97 per clip against the reference torch implementation.
