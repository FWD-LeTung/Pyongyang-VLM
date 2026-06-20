"""Tracklet buffering and dispatch payload creation."""

from __future__ import annotations

from typing import Any

from src.utils.logger import setup_logger
from src.utils.timer import time_it
from src.vision_pipeline.schema import BufferConfig, PersonData, TrackletPayload, TrackletStatus


logger = setup_logger(__name__)


class TrackletBufferManager:
    """Group per-frame crops by track ID before dispatching to Module 3."""

    def __init__(self, config: BufferConfig) -> None:
        self.config = config
        self.tracklets_buffer: dict[int, dict[str, Any]] = {}

    @time_it
    def update(
        self,
        people: list[PersonData],
        timestamp: float | None = None,
    ) -> list[TrackletPayload]:
        """Append current frame crops and emit ready/lost tracklet payloads."""

        payloads: list[TrackletPayload] = []
        current_track_ids: set[int] = set()
        current_timestamp = self._current_timestamp(people, timestamp)

        for person in people:
            current_track_ids.add(person.track_id)
            buffer = self._ensure_buffer(person)
            if not buffer["images"]:
                buffer["first_seen"] = float(person.timestamp)
            buffer["images"].append(person.image_crop)
            buffer["frame_ids"].append(int(person.frame_id))
            buffer["bboxes"].append(person.bbox)
            buffer["confidence_scores"].append(float(person.conf))
            buffer["timestamps"].append(float(person.timestamp))
            buffer["last_seen"] = float(person.timestamp)

            if len(buffer["images"]) >= self.config.batch_size:
                payloads.append(self._build_payload(person.track_id, "ready"))
                self._clear_samples(person.track_id)

        if current_timestamp is None:
            return payloads

        for track_id in list(self.tracklets_buffer):
            if track_id in current_track_ids:
                continue

            buffer = self.tracklets_buffer[track_id]
            time_since_seen = current_timestamp - float(buffer["last_seen"])
            if time_since_seen > self.config.lost_timeout_sec:
                if buffer["images"]:
                    payloads.append(self._build_payload(track_id, "lost"))
                del self.tracklets_buffer[track_id]

        if payloads:
            logger.info("Emitted %s tracklet payload(s).", len(payloads))
        return payloads

    def flush_all(self, status: TrackletStatus = "lost") -> list[TrackletPayload]:
        """Flush every non-empty buffer and remove all IDs from RAM."""

        payloads: list[TrackletPayload] = []
        for track_id in list(self.tracklets_buffer):
            if self.tracklets_buffer[track_id]["images"]:
                payloads.append(self._build_payload(track_id, status))
            del self.tracklets_buffer[track_id]
        return payloads

    def _ensure_buffer(self, person: PersonData) -> dict[str, Any]:
        """Create the in-RAM basket for a track ID if it does not exist."""

        if person.track_id not in self.tracklets_buffer:
            self.tracklets_buffer[person.track_id] = {
                "images": [],
                "frame_ids": [],
                "bboxes": [],
                "confidence_scores": [],
                "timestamps": [],
                "first_seen": float(person.timestamp),
                "last_seen": float(person.timestamp),
            }
        return self.tracklets_buffer[person.track_id]

    def _build_payload(self, track_id: int, status: TrackletStatus) -> TrackletPayload:
        """Copy buffered references into the standard Tracklet Payload dict."""

        buffer = self.tracklets_buffer[track_id]
        return {
            "track_id": int(track_id),
            "status": status,
            "images": list(buffer["images"]),
            "metadata": {
                "frame_ids": [int(frame_id) for frame_id in buffer["frame_ids"]],
                "bboxes": [list(map(int, bbox)) for bbox in buffer["bboxes"]],
                "confidence_scores": [
                    float(score) for score in buffer["confidence_scores"]
                ],
                "timestamps": [float(ts) for ts in buffer["timestamps"]],
                "timeline_frame_ids": [
                    int(frame_id) for frame_id in buffer["frame_ids"]
                ],
                "timeline_bboxes": [
                    list(map(int, bbox)) for bbox in buffer["bboxes"]
                ],
                "timeline_timestamps": [
                    float(ts) for ts in buffer["timestamps"]
                ],
                "timeline_confidence_scores": [
                    float(score) for score in buffer["confidence_scores"]
                ],
                "time_start": float(buffer["timestamps"][0]),
                "time_end": float(buffer["timestamps"][-1]),
                "first_seen": float(buffer["first_seen"]),
                "last_seen": float(buffer["last_seen"]),
            },
        }

    def _clear_samples(self, track_id: int) -> None:
        """Empty a track basket after dispatch while keeping its ID alive."""

        buffer = self.tracklets_buffer[track_id]
        buffer["images"].clear()
        buffer["frame_ids"].clear()
        buffer["bboxes"].clear()
        buffer["confidence_scores"].clear()
        buffer["timestamps"].clear()

    @staticmethod
    def _current_timestamp(
        people: list[PersonData],
        timestamp: float | None,
    ) -> float | None:
        """Resolve the frame timestamp used for timestamp-based GC."""

        if timestamp is not None:
            return float(timestamp)
        if people:
            return max(float(person.timestamp) for person in people)
        return None
