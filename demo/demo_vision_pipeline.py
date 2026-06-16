from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision_pipeline.pipeline import VisionPipeline  # noqa: E402
from src.vision_pipeline.schema import (  # noqa: E402
    Detection,
    FrameProcessingResult,
    PersonData,
    TrackedObject,
    TrackletPayload,
    load_vision_pipeline_config,
)


@dataclass
class DebugRunResult:
    """Demo outputs collected from one Module 2 run."""

    frames: list[dict[str, object]]
    payloads: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Module 2 demo."""

    parser = argparse.ArgumentParser(
        description="Run Module 2 vision pipeline and export tracklet payload JSON.",
    )
    parser.add_argument(
        "--config",
        default="config/vision_pipeline.yaml",
        help="Path to vision pipeline YAML config.",
    )
    parser.add_argument(
        "--source",
        help="Video path or RTSP URL. Overrides config source.",
    )
    parser.add_argument(
        "--mode",
        choices=["video", "stream"],
        help="Input mode. Overrides config source.mode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional raw-frame cap. Use 0 for no cap.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Source timeline seconds to export. Use 0 for full source or max-frames cap.",
    )
    parser.add_argument(
        "--output-json",
        default="data/output/vision_pipeline_payloads.json",
        help="Metadata-only JSON output file.",
    )
    parser.add_argument(
        "--output-video",
        default="data/output/output_tracking.mp4",
        help="Annotated tracking video output file.",
    )
    parser.add_argument(
        "--debug-crop-dir",
        default="data/output/debug_crops",
        help="Directory for 224x224 debug crop images.",
    )
    parser.add_argument(
        "--no-debug-crops",
        action="store_false",
        dest="save_debug_crops",
        help="Disable debug crop image export.",
    )
    parser.set_defaults(save_debug_crops=True)
    return parser.parse_args()


def main() -> int:
    """Run the pipeline and export demo-friendly JSON."""

    args = parse_args()
    config = load_vision_pipeline_config(PROJECT_ROOT / args.config)
    if args.source is not None or args.mode is not None:
        config = replace(
            config,
            reader=replace(
                config.reader,
                source=args.source or config.reader.source,
                mode=args.mode or config.reader.mode,
            ),
        )

    max_frames = None if args.max_frames <= 0 else args.max_frames
    duration_sec = None if args.duration_sec <= 0 else args.duration_sec
    pipeline = VisionPipeline(config)
    payload_queue: Queue[TrackletPayload] = Queue()

    output_json = PROJECT_ROOT / args.output_json
    output_video = PROJECT_ROOT / args.output_video
    debug_crop_dir = PROJECT_ROOT / args.debug_crop_dir
    debug_result = run_debug_pipeline(
        pipeline,
        payload_queue=payload_queue,
        max_frames=max_frames,
        duration_sec=duration_sec,
        output_video=output_video,
        debug_crop_dir=debug_crop_dir,
        save_debug_crops=args.save_debug_crops,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "mode": config.reader.mode,
        "source": config.reader.source,
        "requested_duration_sec": duration_sec,
        "processing_fps": config.reader.processing_fps,
        "processed_frame_count": len(debug_result.frames),
        "tracklet_payload_count": len(debug_result.payloads),
        "frames": debug_result.frames,
        "tracklet_payloads": debug_result.payloads,
    }
    output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote Module 2 debug JSON to {_relative(output_json)}")
    print(f"Wrote tracking video to {_relative(output_video)}")
    if args.save_debug_crops:
        print(f"Saved debug crops under {_relative(debug_crop_dir)}")
    return 0


def run_debug_pipeline(
    pipeline: VisionPipeline,
    *,
    payload_queue: Queue[TrackletPayload],
    max_frames: int | None,
    duration_sec: float | None,
    output_video: Path,
    debug_crop_dir: Path,
    save_debug_crops: bool,
) -> DebugRunResult:
    """Run Module 2 and collect queue payloads for debug outputs."""

    writer: TrackingVideoWriter | None = None
    frame_records: list[dict[str, object]] = []
    payload_records: list[dict[str, object]] = []
    image_counter = 0
    frame_index = 0
    start_timestamp: float | None = None

    output_video.parent.mkdir(parents=True, exist_ok=True)
    if save_debug_crops:
        prepare_debug_crop_dir(debug_crop_dir)

    try:
        with pipeline.reader:
            for packet in pipeline.reader.frames(max_frames=max_frames):
                if not pipeline.should_process(packet.timestamp):
                    continue
                if start_timestamp is None:
                    start_timestamp = packet.timestamp
                elapsed_sec = packet.timestamp - start_timestamp
                if duration_sec is not None and elapsed_sec > duration_sec:
                    break

                result = pipeline.process_frame_debug(packet)
                frame_records.append(
                    frame_record(frame_index, packet.timestamp, elapsed_sec, result)
                )
                frame_index += 1

                for payload in result.payloads:
                    payload_queue.put(payload)
                image_counter = consume_payload_queue(
                    payload_queue,
                    records=payload_records,
                    debug_crop_dir=debug_crop_dir,
                    save_debug_crops=save_debug_crops,
                    image_counter=image_counter,
                )

                annotated = draw_tracks(packet.frame.copy(), result.tracked_objects)
                if writer is None:
                    writer = TrackingVideoWriter(
                        output_video,
                        frame_shape=annotated.shape,
                        fps=output_fps(pipeline),
                    )
                writer.write(annotated)

        for payload in pipeline.buffer_manager.flush_all(status="lost"):
            payload_queue.put(payload)
        consume_payload_queue(
            payload_queue,
            records=payload_records,
            debug_crop_dir=debug_crop_dir,
            save_debug_crops=save_debug_crops,
            image_counter=image_counter,
        )
    finally:
        if writer is not None:
            writer.release()

    return DebugRunResult(frames=frame_records, payloads=payload_records)


def consume_payload_queue(
    payload_queue: Queue[TrackletPayload],
    *,
    records: list[dict[str, object]],
    debug_crop_dir: Path,
    save_debug_crops: bool,
    image_counter: int,
) -> int:
    """Drain TrackletPayload queue into JSON records and optional debug crops."""

    while not payload_queue.empty():
        payload = payload_queue.get()
        track_id = int(payload["track_id"])
        metadata = payload.get("metadata", {})
        timestamps = [float(ts) for ts in metadata.get("timestamps", [])]
        time_start = timestamps[0] if timestamps else None
        time_end = timestamps[-1] if timestamps else None
        image_paths: list[str] = []

        if save_debug_crops:
            for image_index, image in enumerate(payload.get("images", [])):
                image_path = (
                    debug_crop_dir
                    / f"track_{track_id}_img_{image_counter:06d}_{image_index:02d}.jpg"
                )
                cv2.imwrite(str(image_path), image)
                image_paths.append(_relative(image_path))
                image_counter += 1

        records.append(
            {
                "payload_index": len(records),
                "track_id": track_id,
                "status": str(payload["status"]),
                "time_start": time_start,
                "time_end": time_end,
                "duration_sec": (
                    time_end - time_start
                    if time_start is not None and time_end is not None
                    else None
                ),
                "first_seen": metadata.get("first_seen"),
                "last_seen": metadata.get("last_seen"),
                "image_count": len(payload.get("images", [])),
                "image_shape": image_shape(payload),
                "debug_crop_paths": image_paths,
                "bboxes": metadata.get("bboxes", []),
                "confidences": metadata.get("confidence_scores", []),
                "timestamps": timestamps,
            }
        )
    return image_counter


def frame_record(
    frame_index: int,
    timestamp: float,
    elapsed_sec: float,
    result: FrameProcessingResult,
) -> dict[str, object]:
    """Serialize all per-frame Module 2 stage outputs without raw images."""

    return {
        "frame_index": frame_index,
        "timestamp": float(timestamp),
        "elapsed_sec": float(elapsed_sec),
        "detections": [detection_record(item) for item in result.detections],
        "tracked_objects": [track_record(item) for item in result.tracked_objects],
        "crops": [crop_record(item) for item in result.people],
        "emitted_payload_count": len(result.payloads),
        "emitted_payload_track_ids": [
            int(payload["track_id"]) for payload in result.payloads
        ],
    }


def detection_record(detection: Detection) -> dict[str, object]:
    """Serialize one detector output."""

    return {
        "bbox": [
            float(detection.x1),
            float(detection.y1),
            float(detection.x2),
            float(detection.y2),
        ],
        "confidence": float(detection.conf),
    }


def track_record(tracked_object: TrackedObject) -> dict[str, object]:
    """Serialize one tracker output."""

    return {
        "track_id": int(tracked_object.track_id),
        "bbox": [float(value) for value in tracked_object.bbox],
        "confidence": float(tracked_object.conf),
    }


def crop_record(person: PersonData) -> dict[str, object]:
    """Serialize one cropper output without raw pixels."""

    return {
        "track_id": int(person.track_id),
        "bbox": [int(value) for value in person.bbox],
        "confidence": float(person.conf),
        "timestamp": float(person.timestamp),
        "image_shape": list(person.image_crop.shape),
    }


def image_shape(payload: TrackletPayload) -> list[int] | None:
    """Return the shape of payload images without serializing image data."""

    images = payload.get("images", [])
    if not images:
        return None
    return list(images[0].shape)


def draw_tracks(frame, tracked_objects: list[TrackedObject]):
    """Draw bbox and track_id on one original frame."""

    for tracked_object in tracked_objects:
        x1, y1, x2, y2 = [int(round(value)) for value in tracked_object.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_origin = (x1, max(20, y1 - 8))
        cv2.putText(
            frame,
            f"ID {tracked_object.track_id}",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


class TrackingVideoWriter:
    """Write OpenCV frames and transcode to broadly playable H.264 MP4."""

    def __init__(
        self,
        output_video: Path,
        *,
        frame_shape: tuple[int, ...],
        fps: float,
    ) -> None:
        self.output_video = output_video
        self.raw_video = output_video.with_name(f"{output_video.stem}.raw.mp4")
        self.frame_count = 0
        self.last_frame = None

        height, width = frame_shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.raw_video}")

    def write(self, frame) -> None:
        self.writer.write(frame)
        self.frame_count += 1
        self.last_frame = frame

    def release(self) -> None:
        if self.frame_count == 1 and self.last_frame is not None:
            self.writer.write(self.last_frame)
        self.writer.release()
        self._transcode_to_h264()

    def _transcode_to_h264(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.raw_video.replace(self.output_video)
            return

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(self.raw_video),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.output_video),
                ],
                check=True,
            )
            self.raw_video.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            self.raw_video.replace(self.output_video)


def prepare_debug_crop_dir(debug_crop_dir: Path) -> None:
    """Create debug crop directory and remove stale crop images from prior runs."""

    debug_crop_dir.mkdir(parents=True, exist_ok=True)
    for image_path in debug_crop_dir.glob("*.jpg"):
        image_path.unlink()


def output_fps(pipeline: VisionPipeline) -> float:
    """Pick a stable FPS for the debug tracking video."""

    if pipeline.config.reader.processing_fps > 0:
        return pipeline.config.reader.processing_fps
    return float(pipeline.config.tracker.frame_rate)


def _relative(path: Path) -> str:
    """Return a compact project-relative path when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
