"""End-to-end Module 2 detection, tracking, cropping, and dispatch pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from queue import Queue
from typing import Any

from src.utils.logger import setup_logger
from src.utils.timer import time_it
from src.vision_pipeline.components.buffer_manager import TrackletBufferManager
from src.vision_pipeline.components.image_cropper import PersonCropper
from src.vision_pipeline.components.video_reader import FrameSource
from src.vision_pipeline.core.detector import YOLOPersonDetector
from src.vision_pipeline.core.tracker import ByteTrackPersonTracker
from src.vision_pipeline.schema import (
    FramePacket,
    FrameProcessingResult,
    TrackedObject,
    TrackletPayload,
    VisionPipelineConfig,
    load_vision_pipeline_config,
)


logger = setup_logger(__name__)


class VisionPipeline:
    """Run the four processing stages of Module 2 and emit tracklet payloads."""

    def __init__(
        self,
        config: VisionPipelineConfig,
        *,
        reader: Any | None = None,
        detector: Any | None = None,
        tracker: Any | None = None,
        cropper: Any | None = None,
        buffer_manager: TrackletBufferManager | None = None,
    ) -> None:
        self.config = config
        self.reader = reader or FrameSource(config.reader)
        self.detector = detector or YOLOPersonDetector(config.detector)
        self.tracker = tracker or ByteTrackPersonTracker(config.tracker)
        self.cropper = cropper or PersonCropper(config.cropper)
        self.buffer_manager = buffer_manager or TrackletBufferManager(config.buffer)
        self._last_processed_timestamp: float | None = None
        self.last_run_stats: dict[str, int | str | None] = {}

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "config/vision_pipeline.yaml",
        *,
        source: str | None = None,
        mode: str | None = None,
    ) -> "VisionPipeline":
        """Create a pipeline from YAML with optional source overrides."""

        config = load_vision_pipeline_config(config_path)
        if source is not None or mode is not None:
            reader = replace(
                config.reader,
                source=source or config.reader.source,
                mode=mode or config.reader.mode,
            )
            config = replace(config, reader=reader)
        return cls(config)

    def run(
        self,
        *,
        max_frames: int | None = None,
        output_queue: Queue[TrackletPayload] | None = None,
        flush_on_end: bool = True,
    ) -> list[TrackletPayload]:
        """Process frames and optionally push emitted payloads to a queue."""

        emitted_payloads: list[TrackletPayload] = []
        processed_frames = 0
        read_frames = 0
        skipped_frames = 0
        stop_reason = "end_of_video" if self.config.reader.mode == "video" else "stream_stopped"

        with self.reader:
            for packet in self.reader.frames(max_frames=None):
                read_frames += 1
                if not self.should_process(packet.timestamp):
                    skipped_frames += 1
                    continue

                payloads = self.process_frame(packet)
                self._dispatch(payloads, output_queue)
                emitted_payloads.extend(payloads)
                processed_frames += 1
                if max_frames is not None and processed_frames >= max_frames:
                    stop_reason = "max_frames_reached"
                    break

        if flush_on_end:
            payloads = self.buffer_manager.flush_all(status="lost")
            self._dispatch(payloads, output_queue)
            emitted_payloads.extend(payloads)

        self.last_run_stats = {
            "requested_max_frames": max_frames,
            "processed_frames": processed_frames,
            "read_frames": read_frames,
            "skipped_frames": skipped_frames,
            "payloads": len(emitted_payloads),
            "stop_reason": stop_reason,
        }
        logger.info(
            "Vision pipeline stopped requested_max_frames=%s processed_frames=%s "
            "read_frames=%s skipped_frames=%s payloads=%s stop_reason=%s",
            max_frames,
            processed_frames,
            read_frames,
            skipped_frames,
            len(emitted_payloads),
            stop_reason,
        )
        return emitted_payloads

    @time_it
    def process_frame(self, packet: FramePacket) -> list[TrackletPayload]:
        """Run detection, tracking, cropping, and buffering for one frame."""

        return self.process_frame_debug(packet).payloads

    def process_frame_with_tracks(
        self,
        packet: FramePacket,
    ) -> tuple[list[TrackedObject], list[TrackletPayload]]:
        """Process one frame and return tracked objects for visualization."""

        result = self.process_frame_debug(packet)
        return result.tracked_objects, result.payloads

    def process_frame_debug(self, packet: FramePacket) -> FrameProcessingResult:
        """Process one frame and return every Module 2 stage output."""

        detections = self.detector.detect(packet.frame, packet.timestamp)
        tracked_objects = self.tracker.update(detections, packet.frame, packet.timestamp)
        people = self.cropper.crop(
            packet.frame,
            tracked_objects,
            packet.timestamp,
            packet.frame_id,
        )
        payloads = self.buffer_manager.update(people, packet.timestamp)
        return FrameProcessingResult(
            detections=detections,
            tracked_objects=tracked_objects,
            people=people,
            payloads=payloads,
        )

    def should_process(self, timestamp: float) -> bool:
        """Apply optional FPS sampling before heavy AI stages."""

        fps = self.config.reader.processing_fps
        if fps <= 0:
            return True
        if self._last_processed_timestamp is None:
            self._last_processed_timestamp = timestamp
            return True

        min_interval = 1.0 / fps
        if timestamp - self._last_processed_timestamp >= min_interval:
            self._last_processed_timestamp = timestamp
            return True
        return False

    @staticmethod
    def _dispatch(
        payloads: list[TrackletPayload],
        output_queue: Queue[TrackletPayload] | None,
    ) -> None:
        """Push payloads to the Module 3 queue when one is provided."""

        if output_queue is None:
            return
        for payload in payloads:
            output_queue.put(payload)
