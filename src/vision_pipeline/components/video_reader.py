"""Frame readers for video files and RTSP streams."""

from __future__ import annotations

import threading
import time
import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np

from src.utils.logger import setup_logger
from src.vision_pipeline.schema import FramePacket, PipelineMode, ReaderConfig


logger = setup_logger(__name__)


CaptureFactory = Callable[[str], Any]
TimeSource = Callable[[], float]


class FrameSource:
    """Read frames from a video file or keep the latest RTSP frame in RAM."""

    def __init__(
        self,
        config: ReaderConfig,
        *,
        capture_factory: CaptureFactory | None = None,
        time_source: TimeSource | None = None,
    ) -> None:
        self.config = config
        self.source = config.source
        self.mode: PipelineMode = config.mode
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.time_source = time_source or time.time
        self.frame_queue: Queue[FramePacket] = Queue(maxsize=max(1, config.queue_size))
        self.dropped_frames = 0

        self._capture: Any | None = None
        self._stream_capture: Any | None = None
        self._video_read_attempts = 0
        self._stream_frame_id = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._drop_lock = threading.Lock()

    def __enter__(self) -> "FrameSource":
        """Open the underlying source for context-managed usage."""

        if self.mode == "video":
            self.open()
        else:
            self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release resources when leaving a context manager."""

        self.close()

    def open(self) -> None:
        """Open a sequential video source."""

        if self.mode != "video":
            self.start()
            return
        if self._capture is not None:
            return

        capture = self.capture_factory(self.source)
        if not self._is_opened(capture):
            self._release_capture(capture)
            capture = self._open_ffmpeg_fallback()
            if capture is None:
                raise RuntimeError(f"Cannot open video source: {self.source}")
        self._capture = capture
        logger.info("Opened video source: %s", self.source)

    def start(self) -> None:
        """Start the RTSP reader thread."""

        if self.mode != "stream":
            self.open()
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._stream_loop,
            name="vision-pipeline-stream-reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started stream reader thread for: %s", self.source)

    def read(self, timeout: float | None = None) -> FramePacket | None:
        """Read the next video frame or the latest queued stream frame."""

        if self.mode == "video":
            return self._read_video_frame()

        self.start()
        try:
            return self.frame_queue.get(
                timeout=self.config.read_timeout_sec if timeout is None else timeout
            )
        except Empty:
            return None

    def frames(
        self,
        *,
        max_frames: int | None = None,
        max_empty_reads: int | None = None,
    ) -> Iterator[FramePacket]:
        """Yield frames until the source ends, max_frames is reached, or stopped."""

        emitted = 0
        empty_reads = 0
        empty_limit = (
            self.config.max_empty_reads if max_empty_reads is None else max_empty_reads
        )

        if self.mode == "video":
            self.open()
        else:
            self.start()

        while not self._stop_event.is_set():
            if max_frames is not None and emitted >= max_frames:
                break

            packet = self.read()
            if packet is None:
                if self.mode == "video":
                    break
                empty_reads += 1
                if max_frames is not None and empty_reads >= empty_limit:
                    logger.warning(
                        "Stopping stream read after %s empty reads.",
                        empty_reads,
                    )
                    break
                continue

            empty_reads = 0
            emitted += 1
            yield packet

    def close(self) -> None:
        """Stop threads and release OpenCV captures."""

        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, self.config.reconnect_interval_sec))
        self._thread = None

        self._release_capture(self._capture)
        self._release_capture(self._stream_capture)
        self._capture = None
        self._stream_capture = None

    def _read_video_frame(self) -> FramePacket | None:
        """Read one frame from a file and use OpenCV's video timestamp."""

        if self._capture is None:
            self.open()
        if self._capture is None:
            return None

        ok, frame = self._capture.read()
        if not ok:
            if self._video_read_attempts > 0:
                return None
            fallback = self._open_ffmpeg_fallback()
            if fallback is None:
                return None
            self._release_capture(self._capture)
            self._capture = fallback
            ok, frame = self._capture.read()
            if not ok:
                return None

        frame_id = self._video_read_attempts
        self._video_read_attempts += 1
        timestamp = 0.0
        if hasattr(self._capture, "get"):
            timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        return FramePacket(frame=frame, timestamp=timestamp, frame_id=frame_id)

    def _open_ffmpeg_fallback(self) -> "_FfmpegVideoCapture | None":
        """Open a video file through ffmpeg when OpenCV cannot decode it."""

        if self.mode != "video":
            return None
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            return None
        try:
            capture = _FfmpegVideoCapture(self.source)
        except Exception as exc:
            logger.warning("ffmpeg fallback failed for %s: %s", self.source, exc)
            return None
        logger.info("Using ffmpeg fallback decoder for video source: %s", self.source)
        return capture

    def _stream_loop(self) -> None:
        """Continuously reconnect and publish the newest RTSP frame."""

        while not self._stop_event.is_set():
            capture = self.capture_factory(self.source)
            self._stream_capture = capture

            if not self._is_opened(capture):
                logger.warning("Cannot connect to RTSP source. Retrying: %s", self.source)
                self._release_capture(capture)
                self._wait_before_reconnect()
                continue

            logger.info("Connected to RTSP source: %s", self.source)
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    logger.warning("RTSP read failed. Reconnecting: %s", self.source)
                    break
                packet = FramePacket(
                    frame=frame,
                    timestamp=self.time_source(),
                    frame_id=self._stream_frame_id,
                )
                self._stream_frame_id += 1
                self._put_latest(packet)

            self._release_capture(capture)
            self._stream_capture = None
            self._wait_before_reconnect()

    def _put_latest(self, packet: FramePacket) -> None:
        """Keep only the newest frame when the realtime queue is full."""

        while not self._stop_event.is_set():
            try:
                self.frame_queue.put_nowait(packet)
                return
            except Full:
                try:
                    self.frame_queue.get_nowait()
                    with self._drop_lock:
                        self.dropped_frames += 1
                except Empty:
                    continue

    def _wait_before_reconnect(self) -> None:
        """Sleep between reconnect attempts while still allowing fast shutdown."""

        self._stop_event.wait(max(0.0, self.config.reconnect_interval_sec))

    @staticmethod
    def _is_opened(capture: Any) -> bool:
        """Return whether a capture object is usable."""

        return bool(capture is not None and (not hasattr(capture, "isOpened") or capture.isOpened()))

    @staticmethod
    def _release_capture(capture: Any | None) -> None:
        """Release OpenCV-like capture objects."""

        if capture is not None and hasattr(capture, "release"):
            capture.release()


class _FfmpegVideoCapture:
    """Small rawvideo ffmpeg reader compatible with the methods used here."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.width, self.height, self.fps = self._probe(source)
        self.frame_size = self.width * self.height * 3
        self.frame_index = 0
        self.last_timestamp = 0.0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                source,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self.process.poll() is None and self.process.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.process.stdout is None:
            return False, None
        raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None

        self.last_timestamp = self.frame_index / self.fps
        self.frame_index += 1
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame.copy()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return self.last_timestamp * 1000.0
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    @staticmethod
    def _probe(source: str) -> tuple[int, int, float]:
        output = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                source,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(output.stdout)["streams"][0]
        return (
            int(stream["width"]),
            int(stream["height"]),
            _parse_fps(str(stream["r_frame_rate"])),
        )


def _parse_fps(value: str) -> float:
    """Parse ffprobe frame-rate fractions such as ``30000/1001``."""

    numerator, separator, denominator = value.partition("/")
    if not separator:
        return max(float(numerator), 1.0)
    fps = float(numerator) / float(denominator)
    return max(fps, 1.0)
