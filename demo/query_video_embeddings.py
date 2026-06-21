"""Query a temporary video embedding index and optionally save debug images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_pipeline.query_index import query_video_index
from src.matching_engine.schema import QueryUnderstandingPayload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to exported .pt video index.")
    parser.add_argument("--query", required=True, help="Raw natural-language query.")
    parser.add_argument("--matching-config", default="config/matching_engine.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pooling", default="topk_mean", choices=["topk_mean"])
    parser.add_argument("--score-topk", type=int, default=3)
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Save full-frame and crop images for the top-k ranked tracks.",
    )
    parser.add_argument(
        "--debug-dir",
        default="outputs/debug_check",
        help="Directory to save debug frame/crop images.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Optional video path override for extracting debug images. "
            "Defaults to video_path stored in the exported index."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = query_video_index(
            index_path=args.index,
            raw_query=args.query,
            matching_config=args.matching_config,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
            score_topk=args.score_topk,
        )

        print_summary(
            result=result,
            index_path=Path(args.index),
            top_k=args.top_k,
        )

        if args.save_debug_images:
            data = result["index_data"]
            video_path = args.video or data.get("video_path")
            if not video_path:
                raise RuntimeError(
                    "No video path available for debug image extraction. "
                    "Pass --video explicitly."
                )
            save_debug_images(
                video_path=Path(video_path),
                ranking=result["ranking"],
                top_k=args.top_k,
                output_dir=Path(args.debug_dir),
            )

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def print_summary(
    *,
    result: dict[str, Any],
    index_path: Path,
    top_k: int,
) -> None:
    query_payload: QueryUnderstandingPayload = result["query_payload"]
    data = result["index_data"]
    ranking = result["ranking"]
    stats = data.get("stats", {})

    print("\n=== Query ===")
    print(f"original_query: {query_payload.metadata.original_query}")
    print(
        "normalized_text: "
        f"{query_payload.vector_search_payload.normalized_text}"
    )
    print(f"language_detected: {query_payload.metadata.language_detected}")
    print(f"generation_source: {query_payload.generation_source}")

    print("\n=== Index ===")
    print(f"index: {index_path}")
    print(f"video: {data.get('video_path')}")
    print(f"num_embeddings: {stats.get('num_embeddings', len(data['track_ids']))}")
    print(f"num_tracks: {len(set(data['track_ids']))}")
    print(f"embedding_dim: {stats.get('embedding_dim', data['embeddings'].shape[1])}")

    for warning in result.get("warnings", []):
        print(warning)

    if ranking:
        margin = result["top1_top2_margin"]
        print(f"top1_top2_margin: {margin:.6f}")
        if len(ranking) >= 2 and margin < 0.005:
            print("WARNING: ambiguous match, top1-top2 margin is too small.")

    print("\n=== Ranking ===")
    for item in ranking[:top_k]:
        print(
            f"#{item['rank']} track_id={item['track_id']} "
            f"score={item['score']:.6f} "
            f"best_frame_id={item['best_frame_id']} "
            f"best_bbox={item['best_bbox']} "
            f"evidence={item['evidence']}"
        )


def save_debug_images(
    *,
    video_path: Path,
    ranking: list[dict[str, Any]],
    top_k: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for debug extraction: {video_path}")

    saved_count = 0
    for item in ranking[:top_k]:
        rank = int(item["rank"])
        frame_id = item.get("best_frame_id")
        bbox = item.get("best_bbox")
        track_id = item.get("track_id")

        if frame_id is None or not bbox:
            print(
                f"WARNING: skip rank {rank}, track_id={track_id}, "
                "missing frame_id or bbox."
            )
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(
                f"WARNING: cannot read frame {frame_id} "
                f"for rank {rank}, track_id={track_id}."
            )
            continue

        x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            print(
                f"WARNING: invalid bbox for rank {rank}, "
                f"track_id={track_id}, bbox={bbox}."
            )
            continue

        crop = frame[y1:y2, x1:x2]
        frame_debug = frame.copy()
        cv2.rectangle(frame_debug, (x1, y1), (x2, y2), (0, 255, 255), 3)

        frame_out = output_dir / f"{rank}_frame_{frame_id}.jpg"
        crop_out = output_dir / f"{rank}_track_{track_id}.jpg"

        ok_frame = cv2.imwrite(str(frame_out), frame_debug)
        ok_crop = cv2.imwrite(str(crop_out), crop)

        if not ok_frame or not ok_crop:
            print(
                f"WARNING: failed to write debug images for rank {rank}, "
                f"track_id={track_id}."
            )
            continue

        saved_count += 2

    cap.release()
    print(f"\nSaved {saved_count} debug image(s) to: {output_dir}")


def clamp_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


if __name__ == "__main__":
    raise SystemExit(main())

