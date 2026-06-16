"""Typed data contracts and configuration for Module 2 vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import yaml


ImageArray: TypeAlias = np.ndarray
PipelineMode = Literal["video", "stream"]
TrackletStatus = Literal["ready", "lost"]
TrackletPayload: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class FramePacket:
    """Raw frame and the timestamp attached at read time."""

    frame: ImageArray
    timestamp: float


@dataclass(frozen=True)
class Detection:
    """Person detection from YOLO in ``[x1, y1, x2, y2, conf]`` format."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def as_list(self) -> list[float]:
        """Return the compact list format used by downstream code."""

        return [self.x1, self.y1, self.x2, self.y2, self.conf]


@dataclass(frozen=True)
class TrackedObject:
    """Tracked detection in ``[x1, y1, x2, y2, track_id, conf]`` format."""

    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    conf: float

    @property
    def bbox(self) -> list[float]:
        """Return the bbox without tracking metadata."""

        return [self.x1, self.y1, self.x2, self.y2]

    def as_list(self) -> list[float]:
        """Return the compact list format expected by crop/buffer code."""

        return [self.x1, self.y1, self.x2, self.y2, float(self.track_id), self.conf]


@dataclass(frozen=True)
class PersonData:
    """One cropped person sample in RAM, ready for buffering."""

    track_id: int
    image_crop: ImageArray
    bbox: list[int]
    conf: float
    timestamp: float


@dataclass(frozen=True)
class FrameProcessingResult:
    """Full per-frame Module 2 result for debug/demo inspection."""

    detections: list[Detection]
    tracked_objects: list[TrackedObject]
    people: list[PersonData]
    payloads: list[TrackletPayload]


@dataclass(frozen=True)
class ReaderConfig:
    """Frame reader settings for either video files or RTSP streams."""

    source: str = "data/test_videos/cctv_full.mp4"
    mode: PipelineMode = "video"
    queue_size: int = 2
    reconnect_interval_sec: float = 2.0
    read_timeout_sec: float = 1.0
    max_empty_reads: int = 100
    processing_fps: float = 0.0


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO person detector settings."""

    weights_path: str = "weights/yolov8n.pt"
    confidence_threshold: float = 0.35
    classes: tuple[int, ...] = (0,)
    device: str | None = None
    image_size: int = 640


@dataclass(frozen=True)
class TrackerConfig:
    """ByteTrack settings mirrored from Ultralytics tracker args."""

    frame_rate: int = 30
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


@dataclass(frozen=True)
class CropperConfig:
    """Crop validation and resize settings."""

    output_size: tuple[int, int] = (224, 224)
    min_width: int = 40
    min_height: int = 60


@dataclass(frozen=True)
class BufferConfig:
    """Tracklet buffering settings before dispatching to Module 3."""

    batch_size: int = 10
    lost_timeout_sec: float = 1.0


@dataclass(frozen=True)
class VisionPipelineConfig:
    """Root configuration object for Module 2."""

    reader: ReaderConfig = field(default_factory=ReaderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    cropper: CropperConfig = field(default_factory=CropperConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any] | None) -> "VisionPipelineConfig":
        """Build config from YAML-compatible nested dictionaries."""

        data = raw_config or {}
        source_data = _as_dict(data.get("source"))
        reader_data = _as_dict(data.get("reader"))
        detector_data = _as_dict(data.get("detector"))
        tracker_data = _as_dict(data.get("tracker"))
        cropper_data = _as_dict(data.get("cropper"))
        buffer_data = _as_dict(data.get("buffer"))

        mode = str(source_data.get("mode", reader_data.get("mode", ReaderConfig.mode)))
        if mode not in {"video", "stream"}:
            raise ValueError("vision_pipeline source.mode must be 'video' or 'stream'.")
        source_key = "stream_uri" if mode == "stream" else "video_uri"
        source = str(
            source_data.get(
                "uri",
                source_data.get(
                    source_key,
                    source_data.get(
                        "video_path",
                        reader_data.get("source", ReaderConfig.source),
                    ),
                ),
            )
        )

        output_size = _int_pair(
            cropper_data.get("output_size", CropperConfig.output_size)
        )
        classes = tuple(int(value) for value in detector_data.get("classes", (0,)))

        tracker_frame_rate = int(
            tracker_data.get("frame_rate", TrackerConfig.frame_rate)
        )

        return cls(
            reader=ReaderConfig(
                source=source,
                mode=mode,  # type: ignore[arg-type]
                queue_size=int(reader_data.get("queue_size", ReaderConfig.queue_size)),
                reconnect_interval_sec=float(
                    reader_data.get(
                        "reconnect_interval_sec",
                        ReaderConfig.reconnect_interval_sec,
                    )
                ),
                read_timeout_sec=float(
                    reader_data.get("read_timeout_sec", ReaderConfig.read_timeout_sec)
                ),
                max_empty_reads=int(
                    reader_data.get("max_empty_reads", ReaderConfig.max_empty_reads)
                ),
                processing_fps=float(
                    reader_data.get("processing_fps", ReaderConfig.processing_fps)
                ),
            ),
            detector=DetectorConfig(
                weights_path=str(
                    detector_data.get("weights_path", DetectorConfig.weights_path)
                ),
                confidence_threshold=float(
                    detector_data.get(
                        "confidence_threshold",
                        DetectorConfig.confidence_threshold,
                    )
                ),
                classes=classes,
                device=detector_data.get("device", DetectorConfig.device),
                image_size=int(
                    detector_data.get("image_size", DetectorConfig.image_size)
                ),
            ),
            tracker=TrackerConfig(
                frame_rate=tracker_frame_rate,
                track_high_thresh=float(
                    tracker_data.get(
                        "track_high_thresh",
                        TrackerConfig.track_high_thresh,
                    )
                ),
                track_low_thresh=float(
                    tracker_data.get("track_low_thresh", TrackerConfig.track_low_thresh)
                ),
                new_track_thresh=float(
                    tracker_data.get(
                        "new_track_thresh",
                        TrackerConfig.new_track_thresh,
                    )
                ),
                track_buffer=int(
                    tracker_data.get("track_buffer", TrackerConfig.track_buffer)
                ),
                match_thresh=float(
                    tracker_data.get("match_thresh", TrackerConfig.match_thresh)
                ),
                fuse_score=bool(tracker_data.get("fuse_score", TrackerConfig.fuse_score)),
            ),
            cropper=CropperConfig(
                output_size=output_size,
                min_width=int(cropper_data.get("min_width", CropperConfig.min_width)),
                min_height=int(cropper_data.get("min_height", CropperConfig.min_height)),
            ),
            buffer=BufferConfig(
                batch_size=int(buffer_data.get("batch_size", BufferConfig.batch_size)),
                lost_timeout_sec=float(
                    buffer_data.get(
                        "lost_timeout_sec",
                        BufferConfig.lost_timeout_sec,
                    )
                ),
            ),
        )


def load_vision_pipeline_config(config_path: str | Path) -> VisionPipelineConfig:
    """Load Module 2 config from YAML."""

    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError("vision_pipeline config must be a YAML mapping.")
    return VisionPipelineConfig.from_dict(config_data)


def _as_dict(value: object) -> dict[str, Any]:
    """Return a mapping or an empty dict for missing config sections."""

    return value if isinstance(value, dict) else {}


def _int_pair(value: object) -> tuple[int, int]:
    """Normalize a two-item size setting into ``(width, height)``."""

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("cropper.output_size must contain exactly two integers.")
