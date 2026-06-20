"""Tests for Matching Engine Module 3 Phase 2 inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from src.matching_engine.components.chunk_sampler import ChunkSampler
from src.matching_engine.components.embedding_cache import EmbeddingCache
from src.matching_engine.components.hierarchical_scorer import HierarchicalScorer
from src.matching_engine.components.track_candidate_builder import TrackCandidateBuilder
from src.matching_engine.config import (
    CacheConfig,
    CandidateConfig,
    ChunkConfig,
    MatchingEngineConfig,
    TrackScoringConfig,
)
from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import (
    MatchingEngineRequest,
    QueryMetadata,
    QueryUnderstandingPayload,
    TrackCandidate,
    TrackletChunk,
    TrackletMetadata,
    TrackletPayloadInput,
    VectorSearchPayload,
)


def test_schema_accepts_tracklets_and_timeline_fields() -> None:
    """Request schema should consume many Module 2 tracklet chunks."""

    request = MatchingEngineRequest(
        query=make_query("red shirt"),
        tracklets=[
            TrackletPayloadInput(
                track_id=1,
                status="ready",
                images=["red"],
                metadata=TrackletMetadata(
                    frame_ids=[10],
                    timeline_frame_ids=[10],
                    timeline_bboxes=[[1, 2, 3, 4]],
                ),
            )
        ],
        video_id="video-a",
    )

    assert request.tracklets[0].metadata.frame_ids == [10]
    assert request.tracklets[0].metadata.timeline_bboxes == [[1, 2, 3, 4]]
    assert request.video_id == "video-a"


def test_empty_query_and_empty_tracklets_return_statuses() -> None:
    """Pipeline should fail fast on invalid query and missing candidates."""

    pipeline = make_pipeline()
    invalid = pipeline.run(
        MatchingEngineRequest(
            query=make_query(""),
            tracklets=[make_payload(1, ["red"], [1])],
        )
    )
    empty = pipeline.run(
        MatchingEngineRequest(
            query=make_query("red shirt"),
            tracklets=[],
        )
    )

    assert invalid.status == "invalid_query"
    assert empty.status == "no_candidate"


def test_candidate_builder_groups_chunks_without_flattening() -> None:
    """Multiple payload chunks with the same track ID become one candidate."""

    builder = TrackCandidateBuilder()
    candidates = builder.build(
        [
            make_payload(7, ["red-a", "red-b"], [0, 1], first_seen=0.0, last_seen=1.0),
            make_payload(3, ["blue-a"], [0], first_seen=0.0, last_seen=0.0),
            make_payload(7, ["red-c", "red-d"], [2, 3], first_seen=2.0, last_seen=3.0),
        ]
    )
    by_track = {candidate.track_id: candidate for candidate in candidates}

    assert set(by_track) == {3, 7}
    assert isinstance(by_track[7], TrackCandidate)
    assert len(by_track[7].chunks) == 2
    assert [len(chunk.images) for chunk in by_track[7].chunks] == [2, 2]
    assert by_track[7].timeline_frame_ids == [0, 1, 2, 3]
    assert by_track[7].first_seen == 0.0
    assert by_track[7].last_seen == 3.0


def test_chunk_sampler_samples_quality_even_and_preserves_alignment() -> None:
    """Sampler should sample within one chunk and keep metadata aligned."""

    chunk = TrackletChunk(
        track_id=9,
        chunk_id=0,
        images=[f"crop-{index}" for index in range(10)],
        frame_ids=list(range(10)),
        bboxes=[[index, index, index + 1, index + 2] for index in range(10)],
        timestamps=[float(index) for index in range(10)],
        confidence_scores=[0.1, 0.2, 0.3, 0.8, 0.2, 0.1, 0.3, 0.99, 0.2, 0.1],
    )
    original_images = list(chunk.images)
    sampled = ChunkSampler(samples_per_chunk=3, max_samples_per_chunk=5).sample(chunk)
    small_sampled = ChunkSampler(samples_per_chunk=3).sample(
        chunk.model_copy(update={"images": ["a", "b"], "frame_ids": [1, 2]})
    )

    assert len(sampled.images) == 3
    assert 7 in sampled.sampled_indices
    assert sampled.frame_ids == sampled.sampled_indices
    assert len(sampled.bboxes) == 3
    assert chunk.images == original_images
    assert small_sampled.images == ["a", "b"]


def test_hierarchical_scorer_topk_and_empty_cases() -> None:
    """Chunk and track top-k means should handle short and empty inputs."""

    scorer = HierarchicalScorer()
    query = torch.tensor([[1.0, 0.0]])
    images = torch.tensor([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]])

    score, best_index = scorer.score_chunk(query, images, crop_topk=5)
    empty_score, empty_index = scorer.score_chunk(query, torch.empty(0, 2))
    track_score = scorer.score_track(
        [
            make_chunk_score(0, 0.1),
            make_chunk_score(1, 0.9),
            make_chunk_score(2, 0.5),
        ],
        top_chunks=5,
    )

    assert score == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)
    assert best_index == 0
    assert empty_score == 0.0
    assert empty_index is None
    assert track_score == pytest.approx((0.9 + 0.5 + 0.1) / 3.0)


def test_pipeline_ranks_track_ids_and_returns_selected_timeline() -> None:
    """Pipeline should rank unique track IDs, not payload chunks."""

    pipeline = make_pipeline()
    request = MatchingEngineRequest(
        query=make_query("red shirt"),
        tracklets=[
            make_payload(7, ["red-a", "red-b"], [10, 11], first_seen=1.0, last_seen=2.0),
            make_payload(3, ["blue-a", "blue-b"], [20, 21], first_seen=1.0, last_seen=2.0),
            make_payload(7, ["red-c"], [12], first_seen=3.0, last_seen=3.0),
        ],
        video_id="video-1",
        session_id="session-1",
    )

    response = pipeline.run(request)

    assert response.status == "success"
    assert response.best_track_id == 7
    assert [result.track_id for result in response.ranking] == [7, 3]
    assert len(response.ranking) == 2
    assert response.selected_track is not None
    assert response.selected_track.frame_ids == [10, 11, 12]
    assert response.ranking[0].num_chunks == 2


def test_pipeline_cache_reuses_chunk_embeddings_for_second_query() -> None:
    """A second query on the same video/session should not re-encode images."""

    encoder = FakeEncoder()
    pipeline = make_pipeline(encoder=encoder)
    request = MatchingEngineRequest(
        query=make_query("red shirt"),
        tracklets=[
            make_payload(7, ["red-a", "red-b"], [1, 2]),
            make_payload(7, ["red-c"], [3]),
            make_payload(3, ["blue-a"], [4]),
        ],
        video_id="video-cache",
        session_id="session-cache",
    )

    first = pipeline.run(request)
    calls_after_first = encoder.image_encode_calls
    second = pipeline.run(request)

    assert first.best_track_id == 7
    assert second.best_track_id == 7
    assert calls_after_first == 3
    assert encoder.image_encode_calls == calls_after_first


def test_pipeline_marks_tiny_score_margin_as_ambiguous() -> None:
    """Near-tied tracks should be reported as ambiguous."""

    pipeline = make_pipeline(encoder=CollapsedEncoder())
    request = MatchingEngineRequest(
        query=make_query("red shirt"),
        tracklets=[
            make_payload(7, ["crop-a"], [1]),
            make_payload(3, ["crop-b"], [2]),
        ],
        video_id="ambiguous-video",
    )

    response = pipeline.run(request)

    assert response.status == "success"
    assert "ambiguous" in response.message


def test_cuda_cache_fails_fast_without_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA requests should not silently fall back to CPU."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        EmbeddingCache(enabled=True, dtype="fp16", device="cuda")

    cache = EmbeddingCache(
        enabled=True,
        dtype="fp16",
        device="cuda",
        allow_cpu_fallback=True,
    )
    assert cache.device.type == "cpu"


class FakeEncoder:
    """Deterministic retrieval encoder for CPU pipeline tests."""

    def __init__(self) -> None:
        self.image_encode_calls = 0

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._embedding(text) for text in texts], dim=0)

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        self.image_encode_calls += 1
        return torch.stack([self._embedding(str(image)) for image in images], dim=0)

    @staticmethod
    def _embedding(value: str) -> torch.Tensor:
        lowered = value.lower()
        if "red" in lowered:
            vector = torch.tensor([1.0, 0.0, 0.0])
        elif "blue" in lowered:
            vector = torch.tensor([0.0, 1.0, 0.0])
        else:
            vector = torch.tensor([0.0, 0.0, 1.0])
        return F.normalize(vector, dim=0)


class CollapsedEncoder:
    """Encoder that intentionally collapses all inputs to test diagnostics."""

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._embedding() for _text in texts], dim=0)

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        return torch.stack([self._embedding() for _image in images], dim=0)

    @staticmethod
    def _embedding() -> torch.Tensor:
        return torch.tensor([1.0, 0.0, 0.0])


def make_pipeline(encoder: FakeEncoder | None = None) -> MatchingEnginePipeline:
    config = MatchingEngineConfig(
        candidate=CandidateConfig(min_total_images=1, min_total_chunks=1),
        chunk=ChunkConfig(samples_per_chunk=3, max_samples_per_chunk=5),
        cache=CacheConfig(enabled=True, dtype="fp32", device="cpu"),
        track_scoring=TrackScoringConfig(crop_topk=2, top_chunks=5),
    )
    return MatchingEnginePipeline(
        encoder=encoder or FakeEncoder(),
        config=config,
        cache=EmbeddingCache(enabled=True, dtype="fp32", device="cpu"),
    )


def make_query(text: str) -> QueryUnderstandingPayload:
    return QueryUnderstandingPayload(
        metadata=QueryMetadata(status="success"),
        vector_search_payload=VectorSearchPayload(normalized_text=text),
    )


def make_payload(
    track_id: int,
    images: list[str],
    frame_ids: list[int],
    *,
    first_seen: float | None = None,
    last_seen: float | None = None,
) -> TrackletPayloadInput:
    return TrackletPayloadInput(
        track_id=track_id,
        status="ready",
        images=images,
        metadata=TrackletMetadata(
            frame_ids=frame_ids,
            bboxes=[[frame_id, 1, frame_id + 10, 20] for frame_id in frame_ids],
            timestamps=[float(frame_id) for frame_id in frame_ids],
            confidence_scores=[0.9 for _ in frame_ids],
            first_seen=first_seen,
            last_seen=last_seen,
        ),
    )


def make_chunk_score(chunk_id: int, score: float):
    from src.matching_engine.schema import ChunkMatchResult

    return ChunkMatchResult(
        track_id=1,
        chunk_id=chunk_id,
        score=score,
        num_samples=1,
    )
