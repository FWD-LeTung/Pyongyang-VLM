"""Unit tests for conservative track stitching helpers."""

from __future__ import annotations

from typing import Any

import torch

from src.demo_pipeline.track_stitching import (
    compute_track_appearance_score,
    merge_track_timelines,
    suggest_related_tracks,
)


def test_merge_track_timelines_sorts_frames_and_keeps_higher_confidence() -> None:
    index_data = make_index_data(
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        track_ids=[1, 2],
        timelines={
            1: {
                "frame_ids": [1, 3],
                "bboxes": [[1, 1, 2, 2], [3, 3, 4, 4]],
                "timestamps": [0.1, 0.3],
                "confidence_scores": [0.5, 0.4],
            },
            2: {
                "frame_ids": [2, 3],
                "bboxes": [[2, 2, 3, 3], [9, 9, 10, 10]],
                "timestamps": [0.2, 0.31],
                "confidence_scores": [0.7, 0.9],
            },
        },
    )

    merged = merge_track_timelines(index_data=index_data, track_ids=[1, 2])

    assert merged["frame_ids"] == [1, 2, 3]
    assert merged["bboxes"] == [[1, 1, 2, 2], [2, 2, 3, 3], [9, 9, 10, 10]]
    assert merged["confidence_scores"] == [0.5, 0.7, 0.9]
    assert merged["source_track_ids"] == [1, 2]


def test_compute_track_appearance_score_high_for_same_low_for_different() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    same_score = compute_track_appearance_score(
        embeddings=embeddings,
        rows_a=[0],
        rows_b=[1],
    )
    different_score = compute_track_appearance_score(
        embeddings=embeddings,
        rows_a=[0],
        rows_b=[2],
    )

    assert same_score > 0.99
    assert different_score < 0.1


def test_suggest_related_tracks_accepts_strong_prior_fragment() -> None:
    index_data = make_index_data(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        track_ids=[1, 2, 3],
        timelines={
            1: timeline(0, 5),
            2: timeline(20, 25),
            3: timeline(800, 805),
        },
    )

    related = suggest_related_tracks(
        index_data=index_data,
        target_track_id=2,
        max_gap_frames=100,
    )

    assert [item["track_id"] for item in related] == [1]
    assert related[0]["decision"] == "accepted"
    assert related[0]["direction"] == "before"


def test_suggest_related_tracks_rejects_low_appearance() -> None:
    index_data = make_index_data(
        embeddings=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        track_ids=[1, 2],
        timelines={
            1: timeline(0, 5),
            2: timeline(20, 25),
        },
    )

    related = suggest_related_tracks(
        index_data=index_data,
        target_track_id=2,
        max_gap_frames=100,
    )

    assert related == []


def test_suggest_related_tracks_rejects_high_overlap() -> None:
    index_data = make_index_data(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        track_ids=[1, 2],
        timelines={
            1: timeline(10, 30),
            2: timeline(20, 40),
        },
    )

    related = suggest_related_tracks(
        index_data=index_data,
        target_track_id=2,
        max_overlap_frames=5,
    )

    assert related == []


def test_suggest_related_tracks_rejects_ambiguous_margin() -> None:
    index_data = make_index_data(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.999, 0.04],
            ]
        ),
        track_ids=[1, 2, 3],
        timelines={
            1: timeline(0, 5),
            2: timeline(50, 55),
            3: timeline(20, 25),
        },
    )

    related = suggest_related_tracks(
        index_data=index_data,
        target_track_id=2,
        max_gap_frames=100,
        min_candidate_margin=0.05,
    )

    assert related == []


def test_track_stitching_public_apis_import() -> None:
    from src.demo_pipeline.track_stitching import (  # noqa: PLC0415
        merge_track_timelines as imported_merge,
    )
    from src.demo_pipeline.track_stitching import (  # noqa: PLC0415
        suggest_related_tracks as imported_suggest,
    )

    assert imported_merge is merge_track_timelines
    assert imported_suggest is suggest_related_tracks


def make_index_data(
    *,
    embeddings: torch.Tensor,
    track_ids: list[int],
    timelines: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    frame_ids = [timelines[track_id]["frame_ids"][0] for track_id in track_ids]
    return {
        "version": 1,
        "video_path": "video.mp4",
        "session_id": "test",
        "embeddings": embeddings.float(),
        "track_ids": track_ids,
        "chunk_ids": [0 for _ in track_ids],
        "crop_indices": [0 for _ in track_ids],
        "frame_ids": frame_ids,
        "bboxes": [[0, 0, 1, 1] for _ in track_ids],
        "timestamps": [0.0 for _ in track_ids],
        "confidence_scores": [0.9 for _ in track_ids],
        "track_timeline": timelines,
        "config": {},
        "stats": {},
    }


def timeline(first_frame: int, last_frame: int) -> dict[str, Any]:
    return {
        "frame_ids": [first_frame, last_frame],
        "bboxes": [[0, 0, 1, 1], [1, 1, 2, 2]],
        "timestamps": [float(first_frame), float(last_frame)],
        "confidence_scores": [0.9, 0.95],
        "first_seen": first_frame,
        "last_seen": last_frame,
        "status": "lost",
    }
