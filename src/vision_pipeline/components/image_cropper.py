"""Person crop and resize utilities."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import setup_logger
from src.utils.timer import time_it
from src.vision_pipeline.schema import CropperConfig, PersonData, TrackedObject


logger = setup_logger(__name__)


class PersonCropper:
    """Clamp bboxes, validate crop quality, and resize to Module 3 input size."""

    def __init__(self, config: CropperConfig) -> None:
        self.config = config
        self.output_size = config.output_size

    @time_it
    def crop(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        timestamp: float,
    ) -> list[PersonData]:
        """Crop each tracked person into RAM-only ``PersonData`` objects."""

        people: list[PersonData] = []
        for tracked_object in tracked_objects:
            clipped_bbox = self._clip_bbox(tracked_object.bbox, frame.shape)
            if clipped_bbox is None:
                continue

            x1, y1, x2, y2 = clipped_bbox
            width = x2 - x1
            height = y2 - y1
            if width < self.config.min_width or height < self.config.min_height:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = self._resize_with_padding(crop)
            people.append(
                PersonData(
                    track_id=tracked_object.track_id,
                    image_crop=resized,
                    bbox=[x1, y1, x2, y2],
                    conf=tracked_object.conf,
                    timestamp=timestamp,
                )
            )

        logger.info("Created %s valid crop(s).", len(people))
        return people

    def _resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """Resize while preserving aspect ratio, then center on a black canvas."""

        target_w, target_h = self.output_size
        original_h, original_w = img.shape[:2]
        scale = min(target_w / original_w, target_h / original_h)
        resized_w = max(1, int(round(original_w * scale)))
        resized_h = max(1, int(round(original_h * scale)))

        resized = cv2.resize(
            img,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - resized_w) // 2
        y_offset = (target_h - resized_h) // 2
        canvas[
            y_offset : y_offset + resized_h,
            x_offset : x_offset + resized_w,
        ] = resized
        return canvas

    @staticmethod
    def _clip_bbox(
        bbox: list[float],
        frame_shape: tuple[int, ...],
    ) -> list[int] | None:
        """Clip a bbox to image boundaries and reject invalid geometry."""

        height, width = frame_shape[:2]
        x1 = max(0, min(width, int(round(bbox[0]))))
        y1 = max(0, min(height, int(round(bbox[1]))))
        x2 = max(0, min(width, int(round(bbox[2]))))
        y2 = max(0, min(height, int(round(bbox[3]))))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]
