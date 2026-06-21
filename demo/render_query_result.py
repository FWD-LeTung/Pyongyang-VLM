"""Query a video embedding index and render the best track bbox to an MP4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
        "--force-render",
        action="store_true",
        help="Render even when score or top1-top2 margin is below threshold.",
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

        data = result["index_data"]
        raw_video_path = args.video or data.get("video_path")
        if not raw_video_path:
            raise RuntimeError("No video path was provided and index has no video_path.")
        video_path = Path(raw_video_path)

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
                    output_path=Path(args.output),
                    hold_frames=args.hold_frames,
                    saved=False,
                )
                print("Match confidence is low. Use --force-render to render anyway.")
                return 0

        timeline = get_track_timeline(data, int(best_track_id))
        output_path = Path(args.output)
        render_track_video(
            video_path=video_path,
            output_path=output_path,
            track_id=int(best_track_id),
            timeline=timeline,
            score=best_score,
            hold_frames=args.hold_frames,
        )
        print_summary(
            result=result,
            best_track_id=int(best_track_id),
            status=status,
            video_path=video_path,
            output_path=output_path,
            hold_frames=args.hold_frames,
            saved=output_path.exists(),
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def get_track_timeline(data: dict[str, Any], track_id: int) -> dict[str, Any]:
    track_timeline = data.get("track_timeline")
    if not isinstance(track_timeline, dict):
        raise RuntimeError("Index does not contain a valid track_timeline dictionary.")

    timeline = track_timeline.get(track_id)
    if timeline is None:
        timeline = track_timeline.get(str(track_id))
    if timeline is None:
        raise RuntimeError(f"Cannot find timeline for track_id={track_id}")
    if not isinstance(timeline, dict):
        raise RuntimeError(f"Timeline for track_id={track_id} is not a dictionary.")
    return timeline


def render_track_video(
    *,
    video_path: Path,
    output_path: Path,
    track_id: int,
    timeline: dict[str, Any],
    score: float,
    hold_frames: int = 15,
) -> None:
    """Render one track timeline onto a copy of the input video."""

    frame_ids = timeline.get("frame_ids") or []
    bboxes = timeline.get("bboxes") or []
    if not frame_ids or not bboxes:
        raise RuntimeError("Timeline must contain non-empty frame_ids and bboxes.")
    if len(frame_ids) != len(bboxes):
        raise RuntimeError(
            "Timeline frame_ids and bboxes lengths differ: "
            f"{len(frame_ids)} != {len(bboxes)}"
        )

    bbox_by_frame = {
        int(frame_id): [int(value) for value in bbox]
        for frame_id, bbox in zip(frame_ids, bboxes, strict=False)
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions from: {video_path}")

    process = start_h264_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
    )

    frame_index = 0
    frames_written = 0
    last_bbox: list[int] | None = None
    last_bbox_frame = -max(0, hold_frames) - 1
    render_error: Exception | None = None
    finish_error: Exception | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            bbox = bbox_by_frame.get(frame_index)
            if bbox is not None:
                last_bbox = bbox
                last_bbox_frame = frame_index
            elif last_bbox is not None and frame_index - last_bbox_frame <= hold_frames:
                bbox = last_bbox

            if bbox is not None:
                draw_bbox(
                    frame=frame,
                    bbox=bbox,
                    track_id=track_id,
                    score=score,
                )

            write_frame(process, frame)
            frame_index += 1
            frames_written += 1
    except Exception as exc:
        render_error = exc
    finally:
        cap.release()
        try:
            finish_h264_writer(process)
        except Exception as exc:
            finish_error = exc

    if render_error is not None:
        raise render_error
    if finish_error is not None:
        raise finish_error
    if frames_written == 0:
        raise RuntimeError(f"No frames were read from video: {video_path}")


def start_h264_writer(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen[bytes]:
    """Start ffmpeg and feed it raw BGR frames for browser-friendly H.264 MP4."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write H.264 MP4 output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def write_frame(process: subprocess.Popen[bytes], frame: Any) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is not available.")
    try:
        process.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        stderr = read_process_stderr(process)
        raise RuntimeError(f"ffmpeg stopped while writing frames: {stderr}") from exc


def finish_h264_writer(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        process.stdin.close()
    stderr = read_process_stderr(process)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}: {stderr}"
        )


def read_process_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode("utf-8", errors="replace").strip()


def draw_bbox(
    *,
    frame: Any,
    bbox: list[int],
    track_id: int,
    score: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return

    color = (0, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"track {track_id} score {score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = max(text_h + baseline + 6, y1)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 6),
        (min(width, x1 + text_w + 10), label_y + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y - baseline),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def clamp_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def print_summary(
    *,
    result: dict[str, Any],
    best_track_id: int,
    status: str,
    video_path: Path,
    output_path: Path,
    hold_frames: int,
    saved: bool,
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


if __name__ == "__main__":
    raise SystemExit(main())
