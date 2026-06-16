"""Module 2: detection, tracking, cropping, and tracklet dispatch."""

from src.vision_pipeline.pipeline import VisionPipeline
from src.vision_pipeline.schema import (
    Detection,
    FramePacket,
    PersonData,
    TrackedObject,
    TrackletPayload,
    VisionPipelineConfig,
    load_vision_pipeline_config,
)

__all__ = [
    "Detection",
    "FramePacket",
    "PersonData",
    "TrackedObject",
    "TrackletPayload",
    "VisionPipeline",
    "VisionPipelineConfig",
    "load_vision_pipeline_config",
]
