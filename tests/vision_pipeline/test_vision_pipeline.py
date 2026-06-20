"""Tests for Module 2 detection/tracking pipeline components."""

from __future__ import annotations

from queue import Queue

import numpy as np
import pytest

from src.vision_pipeline.components.buffer_manager import TrackletBufferManager
from src.vision_pipeline.components.image_cropper import PersonCropper
from src.vision_pipeline.components.video_reader import FrameSource
from src.vision_pipeline.core.tracker import ByteTrackPersonTracker
from src.vision_pipeline.pipeline import VisionPipeline
from src.vision_pipeline.schema import (
    BufferConfig,
    CropperConfig,
    Detection,
    FramePacket,
    PersonData,
    ReaderConfig,
    TrackedObject,
    TrackerConfig,
    VisionPipelineConfig,
)


def test_video_reader_uses_video_timestamp() -> None:
    """Video mode should use CAP_PROP_POS_MSEC timestamps."""

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FakeVideoCapture([frame], [1.5])
    reader = FrameSource(
        ReaderConfig(source="fake.mp4", mode="video"),
        capture_factory=lambda _source: capture,
    )

    with reader:
        packet = reader.read()

    assert packet is not None
    assert packet.timestamp == 1.5
    assert packet.frame_id == 0
    assert np.array_equal(packet.frame, frame)
    assert capture.released is True


def test_stream_queue_drops_old_frames_when_full() -> None:
    """Stream mode should keep the newest frame when AI lags behind."""

    reader = FrameSource(ReaderConfig(source="rtsp://fake", mode="stream", queue_size=1))
    old_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    new_frame = np.ones((2, 2, 3), dtype=np.uint8)

    reader._put_latest(FramePacket(old_frame, 1.0, 7))
    reader._put_latest(FramePacket(new_frame, 2.0, 8))
    packet = reader.frame_queue.get_nowait()

    assert packet.timestamp == 2.0
    assert packet.frame_id == 8
    assert np.array_equal(packet.frame, new_frame)
    assert reader.dropped_frames == 1


def test_cropper_clamps_bbox_and_filters_small_crops() -> None:
    """Cropper should clamp boxes, filter tiny people, and letterbox crops."""

    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    cropper = PersonCropper(CropperConfig(output_size=(224, 224), min_width=20, min_height=20))
    tracked_objects = [
        TrackedObject(-10, 10, 50, 80, track_id=5, conf=0.9),
        TrackedObject(1, 1, 10, 12, track_id=6, conf=0.8),
    ]

    people = cropper.crop(frame, tracked_objects, timestamp=123.0, frame_id=42)

    assert len(people) == 1
    assert people[0].track_id == 5
    assert people[0].frame_id == 42
    assert people[0].bbox == [0, 10, 50, 80]
    assert people[0].image_crop.shape == (224, 224, 3)
    assert np.all(people[0].image_crop[:, :32] == 0)
    assert np.all(people[0].image_crop[:, 32:192] == 255)
    assert np.all(people[0].image_crop[:, 192:] == 0)


def test_resize_with_padding_handles_upscale_and_downscale() -> None:
    """Letterbox resize should preserve aspect ratio in both directions."""

    cropper = PersonCropper(CropperConfig(output_size=(224, 224)))
    small = np.full((40, 20, 3), 255, dtype=np.uint8)
    large = np.full((400, 200, 3), 255, dtype=np.uint8)

    upscaled = cropper._resize_with_padding(small)
    downscaled = cropper._resize_with_padding(large)

    assert upscaled.shape == (224, 224, 3)
    assert downscaled.shape == (224, 224, 3)
    assert np.all(upscaled[:, :56] == 0)
    assert np.all(upscaled[:, 56:168] == 255)
    assert np.all(upscaled[:, 168:] == 0)
    assert np.all(downscaled[:, :56] == 0)
    assert np.all(downscaled[:, 56:168] == 255)
    assert np.all(downscaled[:, 168:] == 0)


def test_buffer_emits_ready_payload_and_keeps_track_id() -> None:
    """A full basket should emit ready and clear samples while keeping the ID."""

    manager = TrackletBufferManager(BufferConfig(batch_size=2, lost_timeout_sec=30.0))
    first = make_person(track_id=105, timestamp=1.0)
    second = make_person(track_id=105, timestamp=2.0)

    assert manager.update([first], timestamp=1.0) == []
    payloads = manager.update([second], timestamp=2.0)

    assert len(payloads) == 1
    assert payloads[0]["track_id"] == 105
    assert payloads[0]["status"] == "ready"
    assert len(payloads[0]["images"]) == 2
    assert payloads[0]["metadata"]["frame_ids"] == [0, 0]
    assert payloads[0]["metadata"]["timeline_frame_ids"] == [0, 0]
    assert manager.tracklets_buffer[105]["images"] == []
    assert manager.tracklets_buffer[105]["frame_ids"] == []
    assert manager.tracklets_buffer[105]["first_seen"] == 1.0
    assert manager.tracklets_buffer[105]["last_seen"] == 2.0
    assert 105 in manager.tracklets_buffer


def test_buffer_flushes_lost_track_after_timeout() -> None:
    """A missing track should flush remaining crops and then be removed."""

    manager = TrackletBufferManager(BufferConfig(batch_size=10, lost_timeout_sec=2.0))
    manager.update([make_person(track_id=108, timestamp=1.0)], timestamp=1.0)

    assert manager.update([], timestamp=3.0) == []
    payloads = manager.update([], timestamp=3.01)

    assert len(payloads) == 1
    assert payloads[0]["track_id"] == 108
    assert payloads[0]["status"] == "lost"
    assert 108 not in manager.tracklets_buffer


def test_tracker_adapter_returns_compact_tracked_objects() -> None:
    """The ByteTrack wrapper should adapt detections without loading YOLO."""

    fake_tracker = FakeTracker()
    tracker = ByteTrackPersonTracker(TrackerConfig(), tracker=fake_tracker)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    objects = tracker.update([Detection(10, 20, 50, 90, 0.91)], frame, timestamp=1.0)

    assert fake_tracker.last_results is not None
    assert fake_tracker.last_results.xyxy.tolist() == [[10.0, 20.0, 50.0, 90.0]]
    assert fake_tracker.last_results.conf.tolist() == pytest.approx([0.91])
    assert objects[0].bbox == [10.0, 20.0, 50.0, 90.0]
    assert objects[0].track_id == 7
    assert objects[0].conf == pytest.approx(0.91)


def test_pipeline_emits_payloads_and_pushes_output_queue() -> None:
    """Pipeline should compose mocked stages and dispatch payloads to a queue."""

    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    config = VisionPipelineConfig(
        reader=ReaderConfig(source="fake.mp4", mode="video", processing_fps=0.0),
        cropper=CropperConfig(output_size=(224, 224), min_width=20, min_height=20),
        buffer=BufferConfig(batch_size=2, lost_timeout_sec=30.0),
    )
    output_queue: Queue = Queue()
    pipeline = VisionPipeline(
        config,
        reader=FakeReader([FramePacket(frame, 1.0, 0), FramePacket(frame, 2.0, 1)]),
        detector=FakeDetector(),
        tracker=FakePipelineTracker(),
        cropper=PersonCropper(config.cropper),
        buffer_manager=TrackletBufferManager(config.buffer),
    )

    payloads = pipeline.run(
        max_frames=2,
        output_queue=output_queue,
        flush_on_end=False,
    )

    assert len(payloads) == 1
    assert payloads[0]["status"] == "ready"
    assert payloads[0]["metadata"]["frame_ids"] == [0, 1]
    assert output_queue.get_nowait()["track_id"] == 1


def test_pipeline_max_frames_counts_processed_frames_after_fps_sampling() -> None:
    """The demo max frame limit should apply after FPS sampling."""

    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    config = VisionPipelineConfig(
        reader=ReaderConfig(source="fake.mp4", mode="video", processing_fps=1.0),
        cropper=CropperConfig(output_size=(224, 224), min_width=20, min_height=20),
        buffer=BufferConfig(batch_size=10, lost_timeout_sec=30.0),
    )
    pipeline = VisionPipeline(
        config,
        reader=FakeReader(
            [
                FramePacket(frame, 0.0, 0),
                FramePacket(frame, 0.2, 1),
                FramePacket(frame, 1.1, 2),
                FramePacket(frame, 2.2, 3),
            ]
        ),
        detector=FakeDetector(),
        tracker=FakePipelineTracker(),
        cropper=PersonCropper(config.cropper),
        buffer_manager=TrackletBufferManager(config.buffer),
    )

    pipeline.run(max_frames=2, flush_on_end=False)

    assert pipeline.last_run_stats["requested_max_frames"] == 2
    assert pipeline.last_run_stats["processed_frames"] == 2
    assert pipeline.last_run_stats["read_frames"] == 3
    assert pipeline.last_run_stats["skipped_frames"] == 1
    assert pipeline.last_run_stats["stop_reason"] == "max_frames_reached"


def make_person(track_id: int, timestamp: float) -> PersonData:
    """Build a small in-RAM crop sample for buffer tests."""

    return PersonData(
        track_id=track_id,
        image_crop=np.zeros((224, 224, 3), dtype=np.uint8),
        bbox=[10, 20, 80, 180],
        conf=0.88,
        timestamp=timestamp,
        frame_id=0,
    )


class FakeVideoCapture:
    """Minimal OpenCV-like video capture for timestamp tests."""

    def __init__(self, frames: list[np.ndarray], timestamps: list[float]) -> None:
        self.frames = frames
        self.timestamps = timestamps
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, _prop_id: int) -> float:
        return self.timestamps[self.index - 1] * 1000.0

    def release(self) -> None:
        self.released = True


class FakeTracker:
    """Fake Ultralytics tracker returning one tracked object."""

    def __init__(self) -> None:
        self.last_results = None

    def update(self, results: object, frame: np.ndarray) -> np.ndarray:
        del frame
        self.last_results = results
        return np.array([[10, 20, 50, 90, 7, 0.91, 0, 0]], dtype=np.float32)


class FakeReader:
    """Context-managed frame source for pipeline integration tests."""

    def __init__(self, packets: list[FramePacket]) -> None:
        self.packets = packets

    def __enter__(self) -> "FakeReader":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def frames(self, *, max_frames: int | None = None) -> list[FramePacket]:
        return self.packets[:max_frames]


class FakeDetector:
    """Detector test double."""

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        del frame, timestamp
        return [Detection(10, 20, 80, 90, 0.9)]


class FakePipelineTracker:
    """Tracker test double that keeps a stable track ID."""

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        timestamp: float,
    ) -> list[TrackedObject]:
        del detections, frame, timestamp
        return [TrackedObject(10, 20, 80, 90, track_id=1, conf=0.9)]
