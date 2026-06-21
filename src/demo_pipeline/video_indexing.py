"""Build in-memory video embedding indexes for demo retrieval flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import TrackCandidate, TrackletPayloadInput
from src.vision_pipeline.pipeline import VisionPipeline


def resolve_device(raw_device: str | None) -> str:
    """Resolve CLI/UI-style device values and fail clearly for unavailable CUDA."""

    if raw_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw_device.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {raw_device} was requested but CUDA is unavailable."
            )
        return f"cuda:{raw_device}"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {raw_device} was requested but torch.cuda.is_available() is False."
        )
    return raw_device


def build_video_index(
    *,
    video_path: str | Path,
    vision_config: str = "config/vision_pipeline.yaml",
    matching_config: str = "config/matching_engine.yaml",
    checkpoint: str | None = None,
    device: str | None = None,
    precision: str | None = None,
    max_frames: int | None = None,
    session_id: str = "gradio",
) -> dict[str, Any]:
    """Run vision extraction and encode sampled person crops into an index dict."""

    resolved_video_path = Path(video_path)
    resolved_device = resolve_device(device)
    resolved_precision = precision or (
        "fp16" if resolved_device.startswith("cuda") else "fp32"
    )
    if resolved_precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be either 'fp32' or 'fp16'.")

    vision = VisionPipeline.from_config_file(
        vision_config,
        source=str(resolved_video_path),
        mode="video",
    )
    payloads = vision.run(max_frames=max_frames)
    tracklets = [TrackletPayloadInput.model_validate(payload) for payload in payloads]

    matching = MatchingEnginePipeline.from_config_file(
        matching_config,
        checkpoint_path=checkpoint,
        device=resolved_device,
        precision=resolved_precision,
    )
    candidates = matching.candidate_builder.build(tracklets)
    images, rows = collect_sampled_crops(matching, candidates)
    if not images:
        raise RuntimeError("No sampled crops were produced; nothing to index.")

    embeddings = matching.encoder.encode_images(images)
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
    if resolved_precision == "fp16" and resolved_device.startswith("cuda"):
        embeddings = embeddings.half()
    embeddings = embeddings.detach().cpu()

    return build_index_payload(
        video_path=resolved_video_path,
        session_id=session_id,
        vision_config=vision_config,
        matching_config=matching_config,
        matching=matching,
        device=resolved_device,
        precision=resolved_precision,
        max_frames=max_frames,
        embeddings=embeddings,
        rows=rows,
        candidates=candidates,
        vision_stats=dict(getattr(vision, "last_run_stats", {}) or {}),
        num_payload_chunks=len(tracklets),
    )


def collect_sampled_crops(
    matching: MatchingEnginePipeline,
    candidates: list[TrackCandidate],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Sample configured crop evidence from every candidate chunk."""

    images: list[Any] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for chunk in candidate.chunks:
            sampled = matching.chunk_sampler.sample(chunk)
            for sample_i, image in enumerate(sampled.images):
                images.append(image)
                rows.append(
                    {
                        "track_id": int(candidate.track_id),
                        "chunk_id": int(chunk.chunk_id),
                        "crop_index": get_or_none(sampled.sampled_indices, sample_i),
                        "frame_id": get_or_none(sampled.frame_ids, sample_i),
                        "bbox": get_or_default(sampled.bboxes, sample_i, []),
                        "timestamp": get_or_none(sampled.timestamps, sample_i),
                        "confidence_score": get_or_none(
                            sampled.confidence_scores,
                            sample_i,
                        ),
                    }
                )
    return images, rows


def build_index_payload(
    *,
    video_path: Path,
    session_id: str,
    vision_config: str,
    matching_config: str,
    matching: MatchingEnginePipeline,
    device: str,
    precision: str,
    max_frames: int | None,
    embeddings: torch.Tensor,
    rows: list[dict[str, Any]],
    candidates: list[TrackCandidate],
    vision_stats: dict[str, Any],
    num_payload_chunks: int,
) -> dict[str, Any]:
    """Assemble the persisted/in-memory index schema used by query_index."""

    return {
        "version": 1,
        "video_path": str(video_path),
        "session_id": str(session_id),
        "embeddings": embeddings,
        "track_ids": [int(row["track_id"]) for row in rows],
        "chunk_ids": [int(row["chunk_id"]) for row in rows],
        "crop_indices": [row["crop_index"] for row in rows],
        "frame_ids": [row["frame_id"] for row in rows],
        "bboxes": [row["bbox"] for row in rows],
        "timestamps": [row["timestamp"] for row in rows],
        "confidence_scores": [row["confidence_score"] for row in rows],
        "track_timeline": build_track_timeline(candidates),
        "config": {
            "vision_config": str(vision_config),
            "matching_config": str(matching_config),
            "checkpoint": str(matching.config.retrieval.checkpoint_path),
            "device": device,
            "precision": precision,
            "max_frames": max_frames,
        },
        "stats": {
            "processed_frames": vision_stats.get("processed_frames"),
            "read_frames": vision_stats.get("read_frames"),
            "skipped_frames": vision_stats.get("skipped_frames"),
            "num_payload_chunks": num_payload_chunks,
            "num_candidates": len(candidates),
            "num_embeddings": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        },
    }


def build_track_timeline(candidates: list[TrackCandidate]) -> dict[int, dict[str, Any]]:
    """Build renderer-friendly per-track timeline metadata."""

    timeline: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        timeline[int(candidate.track_id)] = {
            "frame_ids": [int(value) for value in candidate.timeline_frame_ids],
            "bboxes": [list(map(int, bbox)) for bbox in candidate.timeline_bboxes],
            "timestamps": [float(value) for value in candidate.timeline_timestamps],
            "confidence_scores": [
                float(value) for value in candidate.timeline_confidence_scores
            ],
            "first_seen": candidate.first_seen,
            "last_seen": candidate.last_seen,
            "status": candidate.status,
        }
    return timeline


def print_summary(index: dict[str, Any], output_path: Path) -> None:
    """Print the same compact export summary used by the CLI wrapper."""

    stats = index["stats"]
    config = index["config"]
    print("\n=== Export Video Embeddings ===")
    print(f"video: {index['video_path']}")
    print(f"output: {output_path}")
    print(f"device: {config['device']}")
    print(f"precision: {config['precision']}")
    print(f"processed_frames: {stats['processed_frames']}")
    print(f"read_frames: {stats['read_frames']}")
    print(f"skipped_frames: {stats['skipped_frames']}")
    print(f"payload_chunks: {stats['num_payload_chunks']}")
    print(f"candidates: {stats['num_candidates']}")
    print(f"embeddings: {stats['num_embeddings']}")
    print(f"embedding_dim: {stats['embedding_dim']}")
    print(f"saved: {output_path.exists()}")


def get_or_none(values: list[Any], index: int) -> Any | None:
    """Return a list value or None when metadata is missing."""

    return values[index] if index < len(values) else None


def get_or_default(values: list[Any], index: int, default: Any) -> Any:
    """Return a list value or a caller-provided default when missing."""

    return values[index] if index < len(values) else default
