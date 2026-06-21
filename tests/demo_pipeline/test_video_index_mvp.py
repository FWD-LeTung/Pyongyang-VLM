"""Tests for the shared video index/query/render MVP helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.demo_pipeline import query_index
from src.demo_pipeline.video_indexing import build_index_payload
from src.demo_pipeline.video_renderer import get_track_timeline
from src.matching_engine.schema import (
    QueryMetadata,
    QueryUnderstandingPayload,
    TrackCandidate,
    TrackletChunk,
    VectorSearchPayload,
)


def test_query_video_index_accepts_in_memory_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The UI path should query a gr.State index without loading a .pt file."""

    index_data = make_index_data()

    monkeypatch.setattr(
        query_index,
        "run_module1",
        lambda raw_query: QueryUnderstandingPayload(
            metadata=QueryMetadata(original_query=raw_query),
            vector_search_payload=VectorSearchPayload(normalized_text="red shirt"),
        ),
    )
    monkeypatch.setattr(
        query_index,
        "encode_text",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0]]),
    )

    result = query_index.query_video_index(
        index_data=index_data,
        raw_query="person in red",
        score_topk=1,
    )

    assert result["index_data"] is index_data
    assert result["best_track_id"] == 1
    assert result["ranking"][0]["best_frame_id"] == 10


def test_query_video_index_requires_path_or_data() -> None:
    with pytest.raises(ValueError, match="Either index_path or index_data"):
        query_index.query_video_index(raw_query="person in red")


def test_get_track_timeline_accepts_int_and_string_keys() -> None:
    data = {"track_timeline": {"7": {"frame_ids": [1], "bboxes": [[1, 2, 3, 4]]}}}

    assert get_track_timeline(data, 7)["frame_ids"] == [1]


def test_build_index_payload_matches_query_schema() -> None:
    matching = SimpleNamespace(
        config=SimpleNamespace(
            retrieval=SimpleNamespace(checkpoint_path="weights/checkpoint_best.pth")
        )
    )
    candidates = [
        TrackCandidate(
            track_id=7,
            status="lost",
            chunks=[
                TrackletChunk(
                    track_id=7,
                    chunk_id=0,
                    images=["crop"],
                    frame_ids=[3],
                    bboxes=[[1, 2, 3, 4]],
                )
            ],
            timeline_frame_ids=[3],
            timeline_bboxes=[[1, 2, 3, 4]],
            timeline_timestamps=[0.1],
            timeline_confidence_scores=[0.9],
        )
    ]

    index = build_index_payload(
        video_path=Path("video.mp4"),
        session_id="test-session",
        vision_config="config/vision_pipeline.yaml",
        matching_config="config/matching_engine.yaml",
        matching=matching,  # type: ignore[arg-type]
        device="cpu",
        precision="fp32",
        max_frames=None,
        embeddings=torch.ones(1, 2),
        rows=[
            {
                "track_id": 7,
                "chunk_id": 0,
                "crop_index": 0,
                "frame_id": 3,
                "bbox": [1, 2, 3, 4],
                "timestamp": 0.1,
                "confidence_score": 0.9,
            }
        ],
        candidates=candidates,
        vision_stats={"processed_frames": 1, "read_frames": 1, "skipped_frames": 0},
        num_payload_chunks=1,
    )

    query_index.validate_index(index)
    assert index["session_id"] == "test-session"
    assert index["track_timeline"][7]["frame_ids"] == [3]
    assert index["stats"]["num_embeddings"] == 1


def make_index_data() -> dict[str, Any]:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    return {
        "version": 1,
        "video_path": "video.mp4",
        "session_id": "test",
        "embeddings": embeddings,
        "track_ids": [1, 2, 1],
        "chunk_ids": [0, 0, 1],
        "crop_indices": [0, 0, 0],
        "frame_ids": [10, 20, 11],
        "bboxes": [[1, 2, 3, 4], [4, 5, 6, 7], [2, 3, 4, 5]],
        "timestamps": [0.1, 0.2, 0.3],
        "confidence_scores": [0.9, 0.8, 0.95],
        "track_timeline": {},
        "config": {
            "matching_config": "config/matching_engine.yaml",
            "checkpoint": "weights/checkpoint_best.pth",
        },
        "stats": {"num_embeddings": 3, "embedding_dim": 2},
    }
