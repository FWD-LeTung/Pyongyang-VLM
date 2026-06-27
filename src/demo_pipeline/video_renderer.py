"""Render selected person tracks onto browser-friendly H.264 MP4 videos."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

import cv2


def get_track_timeline(data: dict[str, Any], track_id: int) -> dict[str, Any]:
    """Fetch timeline metadata for a track ID from int-keyed or str-keyed indexes."""

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


class RenderSegment(NamedTuple):
    """Result of a render: output path plus optional trimmed-segment metadata.

    ``start_frame``/``end_frame``/``segment_length`` are ``None`` when the full
    video was rendered (``trim_segment=False``).
    """

    output_path: Path
    start_frame: int | None
    end_frame: int | None
    segment_length: int | None
    frames_written: int


def compute_segment_bounds(
    *,
    frame_ids: list[int],
    total_frame_count: int | None,
    pad_frames: int = 0,
) -> tuple[int, int]:
    """Return clamped inclusive (start_frame, end_frame) for a trimmed render.

    start = max(0, min(frame_ids) - pad_frames)
    end   = min(total_frame_count - 1, max(frame_ids) + pad_frames)

    When ``total_frame_count`` is ``None`` or non-positive (unknown), ``end``
    is left unclamped and the read-loop EOF guard terminates the render.
    """

    if not frame_ids:
        raise ValueError("frame_ids must contain at least one frame id.")

    pad = max(0, int(pad_frames))
    start = max(0, int(min(frame_ids)) - pad)
    end = int(max(frame_ids)) + pad
    if total_frame_count is not None and int(total_frame_count) > 0:
        end = min(end, int(total_frame_count) - 1)
    if end < start:
        end = start
    return start, end


def render_track_video(
    *,
    video_path: str | Path,
    output_path: str | Path,
    track_id: int,
    timeline: dict[str, Any],
    score: float,
    hold_frames: int = 15,
    trim_segment: bool = False,
    trim_pad_frames: int = 30,
) -> RenderSegment:
    """Render one track timeline onto a copy of the input video.

    When ``trim_segment`` is True, only the segment spanning the track's first
    to last seen frame (plus ``trim_pad_frames`` padding on each side, clamped to
    the video bounds) is rendered, instead of the entire input video. Bounds are
    derived from ``min/max(timeline["frame_ids"])`` so this works for both single
    and stitched (merged) timelines. The bounding box is still drawn.
    """

    resolved_video_path = Path(video_path)
    resolved_output_path = Path(output_path)
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

    cap = cv2.VideoCapture(str(resolved_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {resolved_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions from: {resolved_video_path}")

    start_frame = 0
    end_frame: int | None = None
    if trim_segment:
        raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frame_count = int(raw_total) if raw_total and int(raw_total) > 0 else None
        start_frame, end_frame = compute_segment_bounds(
            frame_ids=[int(value) for value in frame_ids],
            total_frame_count=total_frame_count,
            pad_frames=trim_pad_frames,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    process = start_h264_writer(
        output_path=resolved_output_path,
        width=width,
        height=height,
        fps=fps,
    )

    frame_index = start_frame
    frames_written = 0
    last_bbox: list[int] | None = None
    last_bbox_frame = -max(0, hold_frames) - 1
    render_error: Exception | None = None
    finish_error: Exception | None = None
    try:
        while True:
            if trim_segment and end_frame is not None and frame_index > end_frame:
                break
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
        raise RuntimeError(f"No frames were read from video: {resolved_video_path}")

    segment_length: int | None = None
    if trim_segment and end_frame is not None:
        segment_length = end_frame - start_frame + 1
    return RenderSegment(
        output_path=resolved_output_path,
        start_frame=start_frame if trim_segment else None,
        end_frame=end_frame if trim_segment else None,
        segment_length=segment_length,
        frames_written=frames_written,
    )


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
    """Write one raw BGR frame to the ffmpeg process."""

    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is not available.")
    try:
        process.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        stderr = read_process_stderr(process)
        raise RuntimeError(f"ffmpeg stopped while writing frames: {stderr}") from exc


def finish_h264_writer(process: subprocess.Popen[bytes]) -> None:
    """Close stdin and surface ffmpeg failures with stderr context."""

    if process.stdin is not None:
        process.stdin.close()
    stderr = read_process_stderr(process)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")


def read_process_stderr(process: subprocess.Popen[bytes]) -> str:
    """Return captured process stderr as text."""

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
    """Draw one highlighted track bounding box and label onto a frame."""

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
    """Clamp bbox coordinates to frame bounds."""

    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )
