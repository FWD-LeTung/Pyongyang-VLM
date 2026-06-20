"""Run Matching Engine Phase 2 on saved payloads or a demo video."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_engine.components.embedding_cache import EmbeddingCache
from src.matching_engine.config import (
    CacheConfig,
    CandidateConfig,
    load_matching_engine_config,
)
from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import (
    MatchingEngineRequest,
    QueryMetadata,
    QueryUnderstandingPayload,
    TrackletPayloadInput,
    VectorSearchPayload,
)
from src.vision_pipeline.pipeline import VisionPipeline


class FakeEncoder:
    """Tiny deterministic backend for smoke testing without a checkpoint."""

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._embedding(text) for text in texts], dim=0)

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/matching_engine.yaml")
    parser.add_argument("--vision-config", default="config/vision_pipeline.yaml")
    parser.add_argument("--backend", choices=["tbps", "fake"], default="tbps")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--query-json", default=None)
    parser.add_argument("--tracklets-json", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--session-id", default="demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        request = _smoke_request()
        pipeline = _fake_pipeline(args.config)
    else:
        request = _build_request(args)
        pipeline = _build_pipeline(args)

    response = pipeline.run(request)
    print(f"status: {response.status}")
    print(f"message: {response.message}")
    print(f"best_track_id: {response.best_track_id}")
    print(f"best_score: {response.best_score:.6f}")
    print("ranking:")
    for result in response.ranking:
        print(
            f"  #{result.rank} track_id={result.track_id} "
            f"score={result.score:.6f} chunks={result.num_chunks} "
            f"samples={result.num_sampled_crops}"
        )
    selected_len = len(response.selected_track.frame_ids) if response.selected_track else 0
    print(f"selected_timeline_length: {selected_len}")
    return 0


def _build_pipeline(args: argparse.Namespace) -> MatchingEnginePipeline:
    if args.backend == "fake":
        return _fake_pipeline(args.config)
    return MatchingEnginePipeline.from_config_file(
        args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
    )


def _fake_pipeline(config_path: str) -> MatchingEnginePipeline:
    config = load_matching_engine_config(config_path)
    config = replace(
        config,
        candidate=CandidateConfig(min_total_images=1, min_total_chunks=1),
        cache=CacheConfig(enabled=True, dtype="fp32", device="cpu"),
    )
    return MatchingEnginePipeline(
        encoder=FakeEncoder(),
        config=config,
        cache=EmbeddingCache(enabled=True, dtype="fp32", device="cpu"),
    )


def _build_request(args: argparse.Namespace) -> MatchingEngineRequest:
    query = _load_query(args)
    if args.tracklets_json:
        tracklets = _load_tracklets(Path(args.tracklets_json))
    elif args.video:
        vision = VisionPipeline.from_config_file(
            args.vision_config,
            source=args.video,
            mode="video",
        )
        tracklets = [
            TrackletPayloadInput.model_validate(payload)
            for payload in vision.run(max_frames=args.max_frames)
        ]
    else:
        raise ValueError("Provide --tracklets-json, --video, or --smoke.")
    return MatchingEngineRequest(
        query=query,
        tracklets=tracklets,
        video_id=args.video_id or args.video,
        session_id=args.session_id,
    )


def _load_query(args: argparse.Namespace) -> QueryUnderstandingPayload:
    if args.query_json:
        return QueryUnderstandingPayload.model_validate(
            json.loads(Path(args.query_json).read_text(encoding="utf-8"))
        )
    if args.query:
        return QueryUnderstandingPayload(
            metadata=QueryMetadata(original_query=args.query, status="success"),
            vector_search_payload=VectorSearchPayload(normalized_text=args.query),
        )
    raise ValueError("Provide --query, --query-json, or --smoke.")


def _load_tracklets(path: Path) -> list[TrackletPayloadInput]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Tracklets JSON must be a list of Module 2 payload chunks.")
    return [TrackletPayloadInput.model_validate(_materialize_images(item)) for item in raw]


def _materialize_images(payload: dict[str, Any]) -> dict[str, Any]:
    converted = dict(payload)
    images = []
    for image in converted.get("images", []):
        image_path = Path(image) if isinstance(image, str) else None
        if image_path is not None and image_path.exists():
            images.append(Image.open(image_path).convert("RGB"))
        else:
            images.append(image)
    converted["images"] = images
    return converted


def _smoke_request() -> MatchingEngineRequest:
    return MatchingEngineRequest(
        query=QueryUnderstandingPayload(
            metadata=QueryMetadata(original_query="red shirt", status="success"),
            vector_search_payload=VectorSearchPayload(normalized_text="red shirt"),
        ),
        tracklets=[
            {
                "track_id": 7,
                "status": "ready",
                "images": ["red-a", "red-b"],
                "metadata": {
                    "frame_ids": [1, 2],
                    "bboxes": [[10, 20, 50, 90], [11, 20, 51, 90]],
                    "timestamps": [0.1, 0.2],
                    "confidence_scores": [0.9, 0.88],
                },
            },
            {
                "track_id": 7,
                "status": "ready",
                "images": ["red-c"],
                "metadata": {
                    "frame_ids": [3],
                    "bboxes": [[12, 20, 52, 90]],
                    "timestamps": [0.3],
                    "confidence_scores": [0.91],
                },
            },
            {
                "track_id": 3,
                "status": "ready",
                "images": ["blue-a", "blue-b"],
                "metadata": {
                    "frame_ids": [1, 2],
                    "bboxes": [[60, 20, 100, 90], [61, 20, 101, 90]],
                    "timestamps": [0.1, 0.2],
                    "confidence_scores": [0.9, 0.9],
                },
            },
        ],
        video_id="smoke-video",
        session_id="smoke-session",
    )


if __name__ == "__main__":
    raise SystemExit(main())
