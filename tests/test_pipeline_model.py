"""End-to-end pipeline tests with real models (tiny Whisper, ONNX diarizer/embedder)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="session")
def engine(settings):
    # force tiny model + light backend regardless of ambient config
    from dataclasses import replace

    from overheard.pipeline import Engine

    s = replace(settings, whisper_model="tiny", diar_backend="light", cpu_threads=4)
    return Engine(s)


def test_single_speaker_transcription(engine, jfk_bytes):
    from overheard.pipeline import JobParams

    result = engine.process(jfk_bytes, JobParams(embeddings=True, topics=True))
    assert "ask not" in result["transcript"].lower()
    assert result["language"] == "en"
    assert result["audio_duration_s"] > 5
    assert len(result["segments"]) >= 1
    assert result["diarization_backend"] == "light"
    # one speaker only
    sp = {s["speaker"] for s in result["segments"] if s["speaker"]}
    assert sp in ({"SPEAKER_00"}, set())
    # embeddings present and normalized
    vecs = np.array([s["embedding"] for s in result["segments"]])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
    assert result["topics"]


def test_two_speaker_diarization(engine, conversation_bytes):
    from overheard.pipeline import JobParams

    result = engine.process(
        conversation_bytes, JobParams(num_speakers=2, embeddings=False, topics=False)
    )
    speakers = {s["speaker"] for s in result["segments"] if s["speaker"]}
    assert len(speakers) == 2, f"expected 2 speakers, got {speakers}"

    def speaker_at(t: float) -> str | None:
        for seg in result["segments"]:
            if seg["start"] <= t <= seg["end"]:
                return seg.get("speaker")
        return None

    # fixture layout: JFK 0-8s, other speaker 8.5-14.5s, JFK again 15-19s
    intro, mid, tail = speaker_at(2.0), speaker_at(11.0), speaker_at(17.0)
    assert None not in (intro, mid, tail)
    assert mid != intro, "middle block must be a different speaker"
    assert tail == intro, "final JFK block should match the intro speaker"
    assert len(result["speakers"]) == 2
    for spk in result["speakers"]:
        assert spk["total_speech_s"] > 1.0
