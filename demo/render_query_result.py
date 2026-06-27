"""Query a video embedding index and render the best track bbox to an MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_pipeline.query_index import query_video_index
from src.demo_pipeline.track_stitching import (
    merge_track_timelines,
    suggest_related_tracks,
)
from src.demo_pipeline.video_renderer import (
    RenderSegment,
    get_track_timeline,
    render_track_video,
)
from src.matching_engine.schema import QueryUnderstandingPayload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to exported .pt video index.")
    parser.add_argument(
        "--video",
        default=None,
        help="Optional video path override. Defaults to video_path in the index.",
    )
    parser.add_argument("--query", required=True, help="Raw natural-language query.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--matching-config", default="config/matching_engine.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--score-topk", type=int, default=3)
    parser.add_argument("--hold-frames", type=int, default=15)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument(
        "--auto-stitch",
        action="store_true",
        help="Conservatively stitch fragmented tracks around the best match.",
    )
    parser.add_argument("--stitch-max-gap-frames", type=int, default=300)
    parser.add_argument("--stitch-min-appearance", type=float, default=0.78)
    parser.add_argument("--stitch-min-margin", type=float, default=0.05)
    parser.add_argument("--stitch-max-overlap", type=int, default=12)
    parser.add_argument("--stitch-max-tracks", type=int, default=3)
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Render even when score or top1-top2 margin is below threshold.",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim output to the person segment (first_seen..last_seen + padding). "
            "Default: on. Use --no-trim to render the full video."
        ),
    )
    parser.add_argument(
        "--trim-pad-frames",
        type=int,
        default=30,
        help="Padding frames on each side of the person segment when --trim is on.",
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

        best_track_id = result["best_track_id"]
        if best_track_id is None:
            raise RuntimeError("No ranked track was produced by query_video_index().")

        best_score = float(result["best_score"])
        margin = float(result["top1_top2_margin"])
        low_confidence = best_score < args.min_score or margin < args.min_margin
        status = "low_confidence" if low_confidence else "accepted"
        stitching_summary = {
            "auto_stitch": bool(args.auto_stitch),
            "render_track_ids": [int(best_track_id)],
            "stitched_tracks": [],
            "stitch_candidates": [],
        }

        data = result["index_data"]
        raw_video_path = args.video or data.get("video_path")
        if not raw_video_path:
            raise RuntimeError("No video path was provided and index has no video_path.")
        video_path = Path(raw_video_path)
        output_path = Path(args.output)

        if low_confidence:
            print(
                "WARNING: low confidence match "
                f"(score={best_score:.6f}, margin={margin:.6f})."
            )
            if not args.force_render:
                print_summary(
                    result=result,
                    best_track_id=int(best_track_id),
                    status=status,
                    video_path=video_path,
                    output_path=output_path,
                    hold_frames=args.hold_frames,
                    saved=False,
                    stitching_summary=stitching_summary,
                    segment=None,
                )
                print("Match confidence is low. Use --force-render to render anyway.")
                return 0

        if args.auto_stitch:
            related_tracks = suggest_related_tracks(
                index_data=data,
                target_track_id=int(best_track_id),
                query_ranking=result["ranking"],
                max_gap_frames=args.stitch_max_gap_frames,
                min_appearance_score=args.stitch_min_appearance,
                min_candidate_margin=args.stitch_min_margin,
                max_overlap_frames=args.stitch_max_overlap,
                max_related_tracks=args.stitch_max_tracks,
            )
            related_track_ids = [int(item["track_id"]) for item in related_tracks]
            render_track_ids = related_track_ids + [int(best_track_id)]
            timeline = (
                merge_track_timelines(index_data=data, track_ids=render_track_ids)
                if related_track_ids
                else get_track_timeline(data, int(best_track_id))
            )
            stitching_summary = {
                "auto_stitch": True,
                "render_track_ids": render_track_ids,
                "stitched_tracks": related_track_ids,
                "stitch_candidates": related_tracks,
            }
        else:
            timeline = get_track_timeline(data, int(best_track_id))
        segment = render_track_video(
            video_path=video_path,
            output_path=output_path,
            track_id=int(best_track_id),
            timeline=timeline,
            score=best_score,
            hold_frames=args.hold_frames,
            trim_segment=args.trim,
            trim_pad_frames=args.trim_pad_frames,
        )
        print_summary(
            result=result,
            best_track_id=int(best_track_id),
            status=status,
            video_path=video_path,
            output_path=output_path,
            hold_frames=args.hold_frames,
            saved=output_path.exists(),
            stitching_summary=stitching_summary,
            segment=segment,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def print_summary(
    *,
    result: dict[str, Any],
    best_track_id: int,
    status: str,
    video_path: Path,
    output_path: Path,
    hold_frames: int,
    saved: bool,
    stitching_summary: dict[str, Any],
    segment: RenderSegment | None = None,
) -> None:
    query_payload: QueryUnderstandingPayload = result["query_payload"]

    print("\n=== Query ===")
    print(f"original_query: {query_payload.metadata.original_query}")
    print(
        "normalized_text: "
        f"{query_payload.vector_search_payload.normalized_text}"
    )

    print("\n=== Best Match ===")
    print(f"best_track_id: {best_track_id}")
    print(f"best_score: {float(result['best_score']):.6f}")
    print(f"top1_top2_margin: {float(result['top1_top2_margin']):.6f}")
    print(f"status: {status}")

    print("\n=== Render ===")
    print(f"video: {video_path}")
    print(f"output: {output_path}")
    print(f"hold_frames: {hold_frames}")
    print(f"saved: {saved}")
    if segment is not None and segment.start_frame is not None:
        print(
            "trim: on "
            f"(start_frame={segment.start_frame}, "
            f"end_frame={segment.end_frame}, "
            f"segment_length={segment.segment_length}, "
            f"frames_written={segment.frames_written})"
        )
    else:
        print("trim: off (full video)")

    print("\n=== Stitching ===")
    print(f"auto_stitch: {stitching_summary['auto_stitch']}")
    print(f"render_track_ids: {stitching_summary['render_track_ids']}")
    stitched_tracks = stitching_summary["stitched_tracks"]
    print(f"stitched_tracks: {stitched_tracks if stitched_tracks else 'none'}")
    stitch_candidates = stitching_summary["stitch_candidates"]
    if stitch_candidates:
        print("stitch_candidates:")
        for candidate in stitch_candidates:
            print(
                "- track_id="
                f"{candidate['track_id']} "
                f"appearance={float(candidate['appearance_score']):.6f} "
                f"gap={candidate['gap_frames']} "
                f"direction={candidate['direction']} "
                f"reason={candidate['reason']}"
            )
    else:
        print("stitch_candidates: none")


if __name__ == "__main__":
    raise SystemExit(main())
