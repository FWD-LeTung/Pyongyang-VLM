"""YOLO person detector for Module 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import setup_logger
from src.utils.timer import time_it
from src.vision_pipeline.schema import Detection, DetectorConfig


logger = setup_logger(__name__)


class YOLOPersonDetector:
    """Detect pedestrians with YOLO and return compact RAM-only detections."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self.model = model if model is not None else self._load_model(config.weights_path)
        self.classes = set(config.classes)

    @time_it
    def detect(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Run YOLO and return ``[x1, y1, x2, y2, conf]`` person detections."""

        del timestamp
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            classes=list(self.classes),
            device=self.config.device,
            imgsz=self.config.image_size,
            verbose=False,
        )
        detections: list[Detection] = []

        for result in _as_iterable(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
            confs = _to_numpy(getattr(boxes, "conf", np.empty((0,))))
            classes = _to_numpy(getattr(boxes, "cls", np.zeros(len(confs))))

            for bbox, conf, cls_id in zip(xyxy, confs, classes):
                confidence = float(conf)
                if confidence < self.config.confidence_threshold:
                    continue
                if int(cls_id) not in self.classes:
                    continue
                detections.append(
                    Detection(
                        x1=float(bbox[0]),
                        y1=float(bbox[1]),
                        x2=float(bbox[2]),
                        y2=float(bbox[3]),
                        conf=confidence,
                    )
                )

        logger.info("Detected %s person(s).", len(detections))
        return detections

    @staticmethod
    def _load_model(weights_path: str) -> Any:
        """Load YOLO lazily so tests can inject a model without Ultralytics."""

        from ultralytics import YOLO

        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        return YOLO(str(weights))


def _as_iterable(value: Any) -> list[Any]:
    """Normalize Ultralytics result variants to a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/Ultralytics tensors and Python lists to numpy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
