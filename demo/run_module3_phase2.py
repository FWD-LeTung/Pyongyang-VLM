"""Minimal Module 1 -> Module 2 -> Module 3 demo runner."""

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
from src.matching_engine.schema import (
    MatchingEngineRequest,
    MatchingEngineResponse,
    QueryUnderstandingPayload,
    TrackletPayloadInput,
)
from src.query_understanding.llm_parser import QueryParser
from src.query_understanding.schema import QueryUnderstandingResponse
from src.utils.logger import setup_logger
from src.vision_pipeline.pipeline import VisionPipeline


logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--query", required=True, help="Raw pedestrian query.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--vision-config",
        default="config/vision_pipeline.yaml",
        help="Module 2 config path.",
    )
    parser.add_argument(
        "--matching-config",
        default="config/matching_engine.yaml",
        help="Module 3 config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
    )
    parser.add_argument(
        "--session-id",
        default="demo",
    )
    return parser.parse_args()


def run_module1(raw_query: str) -> QueryUnderstandingPayload:
    """Run Query Understanding and adapt its response for Module 3."""

    logger.info("Running Module 1 Query Understanding.")
    response: QueryUnderstandingResponse = QueryParser().parse(raw_query)
    if response.metadata.status != "success":
        raise RuntimeError(
            "Module 1 failed: "
            f"status={response.metadata.status} "
            f"error_code={response.metadata.error_code}"
        )

    query_payload = QueryUnderstandingPayload.model_validate(
        response.model_dump(mode="json")
    )
    normalized_text = query_payload.vector_search_payload.normalized_text.strip()
    if not normalized_text:
        raise RuntimeError("Module 1 returned an empty normalized_text.")
    return query_payload


def run_module2(
    *,
    video: str,
    vision_config: str,
    max_frames: int,
) -> tuple[list[TrackletPayloadInput], dict[str, Any]]:
    """Run the vision pipeline and return typed tracklet chunks."""

    logger.info("Running Module 2 Vision Pipeline.")
    vision = VisionPipeline.from_config_file(
        vision_config,
        source=video,
        mode="video",
    )
    payloads = vision.run(max_frames=max_frames)
    tracklets = [
        TrackletPayloadInput.model_validate(payload)
        for payload in payloads
    ]
    return tracklets, dict(getattr(vision, "last_run_stats", {}) or {})


def run_module3(
    *,
    query_payload: QueryUnderstandingPayload,
    tracklets: list[TrackletPayloadInput],
    video: str,
    session_id: str,
    matching_config: str,
    checkpoint: str | None,
    device: str,
    precision: str,
) -> MatchingEngineResponse:
    """Run the matching engine over Module 2 tracklet chunks."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but torch.cuda.is_available() is False.")
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use --device cpu for local demo."
        )
    if device == "cpu":
        print(
            "Running TBPS-CLIP on CPU. This may be slow. "
            "Use small --max-frames for local demo."
        )

    logger.info("Running Module 3 Matching Engine.")
    pipeline = MatchingEnginePipeline.from_config_file(
        matching_config,
        checkpoint_path=checkpoint,
        device=device,
        precision=precision,
    )
    request = MatchingEngineRequest(
        query=query_payload,
        tracklets=tracklets,
        video_id=video,
        session_id=session_id,
    )
    return pipeline.run(request)


def print_summary(
    *,
    query_payload: QueryUnderstandingPayload,
    video: str,
    max_frames: int,
    vision_stats: dict[str, Any],
    num_payload_chunks: int,
    response: MatchingEngineResponse,
) -> None:
    """Print a compact human-readable result summary."""

    selected_timeline_length = (
        len(response.selected_track.frame_ids)
        if response.selected_track is not None
        else 0
    )

    print("\n=== Query ===")
    print(f"original_query: {query_payload.metadata.original_query}")
    print(
        "normalized_text: "
        f"{query_payload.vector_search_payload.normalized_text}"
    )
    print(f"language_detected: {query_payload.metadata.language_detected}")
    print(f"generation_source: {query_payload.generation_source}")

    print("\n=== Vision ===")
    print(f"video: {video}")
    print(f"requested_max_frames: {vision_stats.get('requested_max_frames', max_frames)}")
    print(f"processed_frames: {vision_stats.get('processed_frames', 'unknown')}")
    print(f"stop_reason: {vision_stats.get('stop_reason', 'unknown')}")
    print(f"num_payload_chunks: {num_payload_chunks}")

    print("\n=== Matching ===")
    print(f"status: {response.status}")
    print(f"message: {response.message}")
    print(f"best_track_id: {response.best_track_id}")
    print(f"best_score: {response.best_score:.6f}")
    print(f"selected_timeline_length: {selected_timeline_length}")

    if not response.ranking:
        print("\nNo candidate found.")
        return

    if len(response.ranking) >= 2:
        margin = response.ranking[0].score - response.ranking[1].score
        if margin < 0.005:
            print("WARNING: ambiguous match, top1-top2 margin is too small.")

    print("\nTop ranking:")
    for result in response.ranking[:5]:
        print(
            f"#{result.rank} track_id={result.track_id} "
            f"score={result.score:.6f} "
            f"chunks={result.num_chunks} "
            f"samples={result.num_sampled_crops}"
        )


def main() -> int:
    """Run the full demo."""

    args = parse_args()
    try:
        query_payload = run_module1(args.query)
        tracklets, vision_stats = run_module2(
            video=args.video,
            vision_config=args.vision_config,
            max_frames=args.max_frames,
        )
        response = run_module3(
            query_payload=query_payload,
            tracklets=tracklets,
            video=args.video,
            session_id=args.session_id,
            matching_config=args.matching_config,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
        )
        print_summary(
            query_payload=query_payload,
            video=args.video,
            max_frames=args.max_frames,
            vision_stats=vision_stats,
            num_payload_chunks=len(tracklets),
            response=response,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
