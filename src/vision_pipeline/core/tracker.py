"""ByteTrack wrapper for person tracking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src.utils.logger import setup_logger
from src.utils.timer import time_it
from src.vision_pipeline.schema import Detection, TrackedObject, TrackerConfig


logger = setup_logger(__name__)


class ByteTrackPersonTracker:
    """Convert YOLO detections into stable ``track_id`` assignments."""

    def __init__(
        self,
        config: TrackerConfig,
        *,
        tracker: Any | None = None,
    ) -> None:
        self.config = config
        self.tracker = tracker if tracker is not None else self._load_tracker(config)

    @time_it
    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> list[TrackedObject]:
        """Update ByteTrack state and return active tracked objects."""

        del timestamp
        results = _DetectionResults(detections)
        tracked = self.tracker.update(results, frame)
        tracked_array = np.asarray(tracked, dtype=np.float32)
        if tracked_array.size == 0:
            logger.info("Tracked 0 active person(s).")
            return []
        if tracked_array.ndim == 1:
            tracked_array = tracked_array.reshape(1, -1)

        objects = [
            TrackedObject(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                track_id=int(row[4]),
                conf=float(row[5]),
            )
            for row in tracked_array
        ]
        logger.info("Tracked %s active person(s).", len(objects))
        return objects

    @staticmethod
    def _load_tracker(config: TrackerConfig) -> Any:
        """Create Ultralytics ByteTrack with explicit Module 2 config."""

        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ModuleNotFoundError as exc:
            if exc.name == "lap":
                raise RuntimeError(
                    "Ultralytics ByteTrack requires `lap`; install it with `uv add lap`."
                ) from exc
            raise

        args = SimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            fuse_score=config.fuse_score,
        )
        return BYTETracker(args, frame_rate=config.frame_rate)


class _DetectionResults:
    """Small adapter matching the fields Ultralytics BYTETracker consumes."""

    def __init__(self, detections: list[Detection] | np.ndarray) -> None:
        if isinstance(detections, np.ndarray):
            data = detections.astype(np.float32, copy=False)
        else:
            data = np.asarray([detection.as_list() for detection in detections], dtype=np.float32)

        if data.size == 0:
            data = np.empty((0, 5), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 5:
            raise ValueError("ByteTrack detections must have shape Nx5.")

        self.data = data
        self.xyxy = data[:, :4]
        self.conf = data[:, 4]
        self.cls = np.zeros(len(data), dtype=np.float32)
        self.xywh = self._xyxy_to_xywh(self.xyxy)

    def __len__(self) -> int:
        """Return number of detections."""

        return len(self.data)

    def __getitem__(self, index: object) -> "_DetectionResults":
        """Support ByteTrack boolean masks and slices."""

        return _DetectionResults(self.data[index])

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        """Convert top-left/bottom-right boxes to center-width-height boxes."""

        if len(xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        centers_x = xyxy[:, 0] + widths / 2.0
        centers_y = xyxy[:, 1] + heights / 2.0
        return np.column_stack((centers_x, centers_y, widths, heights)).astype(np.float32)
