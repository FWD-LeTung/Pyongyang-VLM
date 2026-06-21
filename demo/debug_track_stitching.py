"""Debug conservative stitch candidates for one target track."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_pipeline.query_index import load_index
from src.demo_pipeline.track_stitching import (
    get_track_time_range,
    rank_stitch_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to exported .pt video index.")
    parser.add_argument("--target-track-id", required=True, type=int)
    parser.add_argument("--max-gap-frames", type=int, default=300)
    parser.add_argument("--min-appearance-score", type=float, default=0.78)
    parser.add_argument("--min-candidate-margin", type=float, default=0.05)
    parser.add_argument("--max-overlap-frames", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index_data = load_index(Path(args.index))
        candidates = rank_stitch_candidates(
            index_data=index_data,
            target_track_id=args.target_track_id,
            max_gap_frames=args.max_gap_frames,
            min_appearance_score=args.min_appearance_score,
            min_candidate_margin=args.min_candidate_margin,
            max_overlap_frames=args.max_overlap_frames,
        )
        print_debug_summary(
            index_data=index_data,
            target_track_id=args.target_track_id,
            candidates=candidates,
            top_k=args.top_k,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def print_debug_summary(
    *,
    index_data: dict[str, Any],
    target_track_id: int,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> None:
    target_start, target_end = get_track_time_range(index_data, target_track_id)
    accepted = [item for item in candidates if item["decision"] == "accepted"]
    rejected = [item for item in candidates if item["decision"] == "rejected"]

    print("=== Track Stitching Debug ===")
    print(f"target_track_id: {target_track_id}")
    print(f"target_range: {target_start}-{target_end}")

    print("\nAccepted related tracks:")
    if accepted:
        for item in accepted:
            print(format_candidate_line(item))
    else:
        print("- none")

    print("\nTop rejected candidates:")
    if rejected:
        for item in rejected[: max(0, int(top_k))]:
            print(format_candidate_line(item))
    else:
        print("- none")


def format_candidate_line(item: dict[str, Any]) -> str:
    return (
        "- track_id="
        f"{item['track_id']} "
        f"appearance={float(item['appearance_score']):.6f} "
        f"gap={item['gap_frames']} "
        f"direction={item['direction']} "
        f"reason={item['reason']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
