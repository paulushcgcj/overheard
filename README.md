# overheard

> Post audio. Get back *who said what, when*, with vector embeddings for every turn.

`overheard` is a self-hosted FastAPI endpoint that runs entirely on CPU:

- **Transcription** — Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8)
- **Speaker diarization** — torch-free `Silero VAD → ECAPA-TDNN ONNX → hierarchical clustering` by default,
  or [pyannote](https://pyannote.ai) `community-1` in the `-accurate` image variant
- **Text embeddings** — `all-MiniLM-L6-v2` ONNX vectors per utterance + semantic topic grouping

No GPU required. Built to run on a 2-core/4-thread laptop CPU.

*(full README written at docs commit)*
