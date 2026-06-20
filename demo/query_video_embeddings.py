"""Query a temporary video embedding index exported by export_video_embeddings.py.
This script:

Loads precomputed image embeddings from a .pt index.
Runs Module 1 to normalize the raw user query.
Encodes the normalized text with TBPS-CLIP.
Computes cosine similarity against saved image embeddings.
Aggregates scores by track_id and prints top ranked tracks.
Optionally saves debug frame/crop images for the top-k tracks.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_engine.config import load_matching_engine_config
from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import QueryUnderstandingPayload
from src.query_understanding.llm_parser import QueryParser
from src.query_understanding.schema import QueryUnderstandingResponse


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
        device = resolve_device(args.device)
        data = load_index(Path(args.index))
        warn_if_config_differs(args, data)

        query_payload = run_module1(args.query)
        text_emb = encode_text(
            query_payload.vector_search_payload.normalized_text,
            args=args,
            device=device,
        )

        image_embs = F.normalize(data["embeddings"].float(), dim=1).to(text_emb.device)
        similarities = (text_emb.float() @ image_embs.T).squeeze(0).detach().cpu()

        ranking = rank_tracks(
            similarities=similarities,
            track_ids=[int(value) for value in data["track_ids"]],
            frame_ids=data["frame_ids"],
            bboxes=data["bboxes"],
            score_topk=args.score_topk,
        )

        print_summary(
            query_payload=query_payload,
            data=data,
            index_path=Path(args.index),
            ranking=ranking,
            top_k=args.top_k,
        )

        if args.save_debug_images:
            video_path = args.video or data.get("video_path")
            if not video_path:
                raise RuntimeError(
                    "No video path available for debug image extraction. "
                    "Pass --video explicitly."
                )

            save_debug_images(
                video_path=Path(video_path),
                ranking=ranking,
                top_k=args.top_k,
                output_dir=Path(args.debug_dir),
            )

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def resolve_device(raw_device: str) -> str:
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


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    validate_index(data)
    return data


def validate_index(data: dict[str, Any]) -> None:
    required_fields = [
        "embeddings",
        "track_ids",
        "chunk_ids",
        "crop_indices",
        "frame_ids",
        "bboxes",
        "timestamps",
        "confidence_scores",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Index is missing required field(s): {missing}")
    
    embeddings = data["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Index embeddings must be a 2D torch.Tensor.")

    num_embeddings = int(embeddings.shape[0])
    mismatched = {
        field: len(data[field])
        for field in required_fields[1:]
        if len(data[field]) != num_embeddings
    }
    if mismatched:
        raise ValueError(
            "Index metadata is not aligned with embeddings: "
            f"num_embeddings={num_embeddings} lengths={mismatched}"
        )

    if num_embeddings == 0:
        raise ValueError("Index contains no embeddings.")


def warn_if_config_differs(args: argparse.Namespace, data: dict[str, Any]) -> None:
    exported_matching_config = data.get("config", {}).get("matching_config")
    if exported_matching_config and str(args.matching_config) != str(exported_matching_config):
        print("WARNING: matching-config argument differs from exported index config.")
    
    checkpoint_arg = effective_checkpoint(args)
    exported_checkpoint = data.get("config", {}).get("checkpoint")
    if exported_checkpoint and str(checkpoint_arg) != str(exported_checkpoint):
        print("WARNING: checkpoint argument differs from exported index checkpoint.")


def effective_checkpoint(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        return str(args.checkpoint)
    config = load_matching_engine_config(args.matching_config)
    return str(config.retrieval.checkpoint_path)


def run_module1(raw_query: str) -> QueryUnderstandingPayload:
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
    if not query_payload.vector_search_payload.normalized_text.strip():
        raise RuntimeError("Module 1 returned an empty normalized_text.")

    return query_payload


def encode_text(
    normalized_text: str,
    *,
    args: argparse.Namespace,
    device: str,
) -> torch.Tensor:
    matching = MatchingEnginePipeline.from_config_file(
        args.matching_config,
        checkpoint_path=args.checkpoint,
        device=device,
        precision=args.precision,
    )
    text_emb = matching.encoder.encode_text([normalized_text])
    return F.normalize(text_emb.float(), dim=1)


def rank_tracks(
    *,
    similarities: torch.Tensor,
    track_ids: list[int],
    frame_ids: list[int | None],
    bboxes: list[list[int]],
    score_topk: int,
) -> list[dict[str, Any]]:
    rows_by_track: dict[int, list[int]] = defaultdict(list)
    for row_index, track_id in enumerate(track_ids):
        rows_by_track[track_id].append(row_index)
        
    ranking: list[dict[str, Any]] = []
    for track_id, row_indices in rows_by_track.items():
        track_scores = similarities[row_indices]

        top_count = min(max(1, score_topk), len(row_indices))
        top_values, _top_local_indices = torch.topk(track_scores, k=top_count)

        best_local_index = int(torch.argmax(track_scores).item())
        best_row_index = row_indices[best_local_index]

        ranking.append(
            {
                "track_id": track_id,
                "score": float(top_values.mean().item()),
                "best_frame_id": get_or_none(frame_ids, best_row_index),
                "best_bbox": get_or_default(bboxes, best_row_index, []),
                "evidence": len(row_indices),
            }
        )

    return sorted(ranking, key=lambda item: item["score"], reverse=True)


def print_summary(
    *,
    query_payload: QueryUnderstandingPayload,
    data: dict[str, Any],
    index_path: Path,
    ranking: list[dict[str, Any]],
    top_k: int,
) -> None:
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

    if len(ranking) >= 2:
        margin = ranking[0]["score"] - ranking[1]["score"]
        print(f"top1_top2_margin: {margin:.6f}")
        if margin < 0.005:
            print("WARNING: ambiguous match, top1-top2 margin is too small.")

    print("\n=== Ranking ===")
    for rank, item in enumerate(ranking[:top_k], start=1):
        print(
            f"#{rank} track_id={item['track_id']} "
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
    for rank, item in enumerate(ranking[:top_k], start=1):
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

        x1, y1, x2, y2 = [int(value) for value in bbox]
        h, w = frame.shape[:2]

        # Clamp bbox to frame boundary before cropping.
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            print(
                f"WARNING: invalid bbox for rank {rank}, "
                f"track_id={track_id}, bbox={bbox}."
            )
            continue

        crop = frame[y1:y2, x1:x2]

        frame_out = output_dir / f"{rank}_frame_{frame_id}.jpg"
        crop_out = output_dir / f"{rank}_track_{track_id}.jpg"

        ok_frame = cv2.imwrite(str(frame_out), frame)
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


def get_or_none(values: list[Any], index: int) -> Any | None:
    return values[index] if index < len(values) else None


def get_or_default(values: list[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default


if __name__ == "__main__":
    raise SystemExit(main())