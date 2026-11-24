"""Pure-math unit tests: no model downloads."""

from __future__ import annotations

import numpy as np

from overheard import features as feat
from overheard.speakers import Region, assign_to_segments, cluster, merge_turns


def test_mel_asset_loads():
    m = feat.mel_matrix()
    assert m.shape == (201, 80)
    assert m.dtype == np.float32
    assert float(m.max()) <= 1.0 and float(m.min()) >= 0.0


def test_log_mel_shape_and_sanity():
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.1, 16_000)).astype(np.float32)  # 1 s of noise
    db = feat.log_mel(audio)
    assert db.shape == (101, 80)  # 16000/160 + 1 frames incl. center pad alignment
    assert np.isfinite(db).all()
    assert -80.0 < float(db.mean()) < 60.0


def test_features_zero_mean_per_dim():
    rng = np.random.default_rng(1)
    audio = (rng.normal(0, 0.2, 32_000)).astype(np.float32)
    f = feat.features(audio)
    assert f.dtype == np.float32
    np.testing.assert_allclose(f.mean(axis=0), 0.0, atol=1e-4)


def test_cluster_recovers_two_groups():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=192), rng.normal(size=192)
    embs = np.stack([a, a, b, b] + [a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    labels = cluster(embs.astype(np.float32), threshold=0.5).tolist()
    assert labels[0] == labels[1] == labels[4]
    assert labels[2] == labels[3] == labels[5]
    assert labels[0] != labels[2]


def test_cluster_respects_num_speakers():
    rng = np.random.default_rng(3)
    embs = rng.normal(size=(8, 192))
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    labels = cluster(embs.astype(np.float32), threshold=0.99, min_speakers=3, max_speakers=3)
    assert set(labels) == {0, 1, 2}


def test_merge_and_assign():
    regions = [Region(0.0, 2.0), Region(2.1, 4.0), Region(5.0, 7.0)]
    labels = np.array([0, 0, 1])
    turns = merge_turns(regions, labels)
    assert turns[0] == (0.0, 4.0, 0)
    assert turns[1] == (5.0, 7.0, 1)
    segments = [{"start": 0.1, "end": 3.9, "text": "hi"}, {"start": 5.2, "end": 6.8, "text": "yo"}]
    assign_to_segments(turns, segments)
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_01"


def test_assign_falls_back_to_nearest():
    turns = [(0.0, 1.0, 1)]
    segments = [{"start": 9.0, "end": 9.5, "text": "x"}]
    assign_to_segments(turns, segments)
    assert segments[0]["speaker"] == "SPEAKER_01"
