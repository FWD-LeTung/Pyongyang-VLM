"""Export sampled Module 2 person-crop embeddings to a temporary .pt index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import TrackCandidate, TrackletPayloadInput
from src.vision_pipeline.pipeline import VisionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum processed frames. Use 0 for the full video.",
    )
    parser.add_argument("--vision-config", default="config/vision_pipeline.yaml")
    parser.add_argument("--matching-config", default="config/matching_engine.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None, choices=["fp32", "fp16"])
    parser.add_argument("--session-id", default="export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    precision = args.precision or ("fp16" if device.startswith("cuda") else "fp32")
    max_frames = None if args.max_frames == 0 else args.max_frames

    if device == "cpu":
        print("WARNING: running image export on CPU; TBPS-CLIP encoding may be slow.")

    print("Running Module 2...")
    vision = VisionPipeline.from_config_file(
        args.vision_config,
        source=args.video,
        mode="video",
    )
    payloads = vision.run(max_frames=max_frames)
    tracklets = [TrackletPayloadInput.model_validate(payload) for payload in payloads]

    print("Loading Matching Engine / TBPS-CLIP image encoder...")
    matching = MatchingEnginePipeline.from_config_file(
        args.matching_config,
        checkpoint_path=args.checkpoint,
        device=device,
        precision=precision,
    )
    candidates = matching.candidate_builder.build(tracklets)
    images, rows = collect_sampled_crops(matching, candidates)
    if not images:
        raise RuntimeError("No sampled crops were produced; nothing to export.")

    print(f"Encoding {len(images)} sampled crop(s)...")
    embeddings = matching.encoder.encode_images(images)
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
    if precision == "fp16" and device.startswith("cuda"):
        embeddings = embeddings.half()
    embeddings = embeddings.detach().cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_index_payload(
        args=args,
        matching=matching,
        device=device,
        precision=precision,
        max_frames=max_frames,
        embeddings=embeddings,
        rows=rows,
        candidates=candidates,
        vision_stats=dict(getattr(vision, "last_run_stats", {}) or {}),
        num_payload_chunks=len(tracklets),
    )
    torch.save(index, output_path)
    print_summary(index, output_path)
    return 0


def resolve_device(raw_device: str | None) -> str:
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


def collect_sampled_crops(
    matching: MatchingEnginePipeline,
    candidates: list[TrackCandidate],
) -> tuple[list[Any], list[dict[str, Any]]]:
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
    args: argparse.Namespace,
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
    return {
        "version": 1,
        "video_path": str(args.video),
        "session_id": str(args.session_id),
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
            "vision_config": str(args.vision_config),
            "matching_config": str(args.matching_config),
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
    return values[index] if index < len(values) else None


def get_or_default(values: list[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default


if __name__ == "__main__":
    raise SystemExit(main())
