"""overheard — self-hosted audio endpoint: transcription, diarization, embeddings.

Runs entirely on CPU: Whisper via CTranslate2 (int8), speaker diarization via
Silero VAD + ECAPA-TDNN ONNX + hierarchical clustering (or pyannote
community-1 in the ``-accurate`` image variant), text embeddings via
all-MiniLM-L6-v2 ONNX.
"""

__version__ = "0.1.0"
