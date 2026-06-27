"""Tests for the renderer's trimmed-segment output (trim to the person segment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.demo_pipeline import video_renderer
from src.demo_pipeline.video_renderer import (
    compute_segment_bounds,
    render_track_video,
)

BBOX_YELLOW = (0, 255, 255)


class FakeCap:
    """OpenCV-like capture supporting seek + frame count for renderer tests.

    Each emitted frame is filled with a pixel value equal to its source frame
    index so tests can map written frames back to their source position.
    """

    def __init__(
        self,
        num_frames: int,
        width: int = 8,
        height: int = 8,
        fps: float = 30.0,
    ) -> None:
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.pos = 0
        self.released = False
        self.requested_seek: list[int] = []

    def isOpened(self) -> bool:
        return True

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.requested_seek.append(int(value))
            self.pos = int(value)
            return True
        return False

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.num_frames)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.pos >= self.num_frames:
            return False, None
        frame = np.full((self.height, self.width, 3), self.pos, dtype=np.uint8)
        self.pos += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []


def has_yellow(frame: np.ndarray) -> bool:
    return bool(np.any(np.all(frame == np.array(BBOX_YELLOW), axis=-1)))


def install_fake_writer(monkeypatch: pytest.MonkeyPatch) -> FakeWriter:
    """Stub the ffmpeg-backed writer helpers and return the captured writer."""

    writer = FakeWriter()
    monkeypatch.setattr(video_renderer, "start_h264_writer", lambda **_: writer)
    monkeypatch.setattr(
        video_renderer, "write_frame", lambda process, frame: process.frames.append(frame)
    )
    monkeypatch.setattr(video_renderer, "finish_h264_writer", lambda process: None)
    return writer


def install_fake_cap(
    monkeypatch: pytest.MonkeyPatch,
    cap: FakeCap,
) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: cap)


# --------------------------------------------------------------------------- #
# compute_segment_bounds (pure logic, no cv2/ffmpeg)
# --------------------------------------------------------------------------- #


def test_segment_bounds_basic_padding() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[50, 51, 52], total_frame_count=100, pad_frames=10
    )
    assert (start, end) == (40, 62)


def test_segment_bounds_clamps_start_to_zero() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[5, 6], total_frame_count=100, pad_frames=30
    )
    assert start == 0
    assert end == 36


def test_segment_bounds_clamps_end_to_total() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[95, 96], total_frame_count=100, pad_frames=30
    )
    assert start == 65
    assert end == 99


def test_segment_bounds_unknown_total_leaves_end_unclamped() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[10, 12], total_frame_count=None, pad_frames=5
    )
    assert (start, end) == (5, 17)


def test_segment_bounds_single_frame() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[42], total_frame_count=100, pad_frames=3
    )
    assert (start, end) == (39, 45)


def test_segment_bounds_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_segment_bounds(frame_ids=[], total_frame_count=100, pad_frames=5)


def test_segment_bounds_uses_min_and_max_not_order() -> None:
    start, end = compute_segment_bounds(
        frame_ids=[60, 40, 50], total_frame_count=100, pad_frames=0
    )
    assert (start, end) == (40, 60)


# --------------------------------------------------------------------------- #
# render_track_video (mocked capture + writer)
# --------------------------------------------------------------------------- #


def _timeline(frame_ids: list[int]) -> dict[str, Any]:
    return {
        "frame_ids": frame_ids,
        "bboxes": [[1, 1, 2, 2] for _ in frame_ids],
    }


def test_render_trim_writes_only_in_range_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = install_fake_writer(monkeypatch)
    cap = FakeCap(num_frames=100)
    install_fake_cap(monkeypatch, cap)

    segment = render_track_video(
        video_path="x.mp4",
        output_path="out.mp4",
        track_id=1,
        timeline=_timeline([50, 51, 52]),
        score=0.5,
        trim_segment=True,
        trim_pad_frames=10,
    )

    assert cap.requested_seek == [40]
    assert segment.start_frame == 40
    assert segment.end_frame == 62
    assert segment.segment_length == 23
    assert segment.frames_written == 23
    assert len(writer.frames) == 23
    assert Path(segment.output_path).name == "out.mp4"


def test_render_trim_clamps_end_to_video(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_writer(monkeypatch)
    cap = FakeCap(num_frames=100)
    install_fake_cap(monkeypatch, cap)

    segment = render_track_video(
        video_path="x.mp4",
        output_path="out.mp4",
        track_id=1,
        timeline=_timeline([95]),
        score=0.5,
        trim_segment=True,
        trim_pad_frames=30,
    )

    assert segment.end_frame == 99
    assert segment.frames_written == 100 - segment.start_frame


def test_render_no_trim_is_full_video(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = install_fake_writer(monkeypatch)
    cap = FakeCap(num_frames=100)
    install_fake_cap(monkeypatch, cap)

    segment = render_track_video(
        video_path="x.mp4",
        output_path="out.mp4",
        track_id=1,
        timeline=_timeline([50, 51, 52]),
        score=0.5,
        trim_segment=False,
    )

    assert cap.requested_seek == []
    assert segment.start_frame is None
    assert segment.end_frame is None
    assert segment.segment_length is None
    assert segment.frames_written == 100
    assert len(writer.frames) == 100


def test_render_default_kwargs_render_full_video(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_writer(monkeypatch)
    cap = FakeCap(num_frames=100)
    install_fake_cap(monkeypatch, cap)

    segment = render_track_video(
        video_path="x.mp4",
        output_path="out.mp4",
        track_id=1,
        timeline=_timeline([50, 51, 52]),
        score=0.5,
    )

    assert cap.requested_seek == []
    assert segment.start_frame is None
    assert segment.frames_written == 100


def test_render_trim_with_gap_draws_box_only_on_timeline_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = install_fake_writer(monkeypatch)
    cap = FakeCap(num_frames=100)
    install_fake_cap(monkeypatch, cap)

    # Stitched-like timeline: only frames 40 and 60 carry a box (gap 41..59).
    # hold_frames=0 so no box is carried into the gap frames.
    segment = render_track_video(
        video_path="x.mp4",
        output_path="out.mp4",
        track_id=1,
        timeline=_timeline([40, 60]),
        score=0.5,
        hold_frames=0,
        trim_segment=True,
        trim_pad_frames=0,
    )

    assert segment.start_frame == 40
    assert segment.end_frame == 60
    assert segment.segment_length == 21
    assert len(writer.frames) == 21
    # Only the two timeline frames (source 40 and 60) should carry a yellow box.
    boxed = [has_yellow(frame) for frame in writer.frames]
    assert sum(boxed) == 2
    assert boxed[0] is True  # source frame 40
    assert boxed[-1] is True  # source frame 60
