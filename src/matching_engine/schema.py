"""Pydantic contracts for Matching Engine runtime inputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryMetadata(BaseModel):
    """Metadata emitted by Module 1."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    original_query: str = ""
    language_detected: str = "unknown"
    status: str = "success"
    error_code: str | None = None


class VectorSearchPayload(BaseModel):
    """Normalized text query emitted by Module 1."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    normalized_text: str = ""


class HybridFilterPayload(BaseModel):
    """Structured filters emitted by Module 1."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    gender: str = "unknown"
    upper_color: str = "unknown"
    upper_type: str = "unknown"
    lower_color: str = "unknown"
    lower_type: str = "unknown"
    footwear: str = "unknown"
    accessory: str = "unknown"


class QueryUnderstandingPayload(BaseModel):
    """Module 1 output consumed by Matching Engine."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    metadata: QueryMetadata
    vector_search_payload: VectorSearchPayload = Field(
        default_factory=VectorSearchPayload
    )
    hybrid_filter_payload: HybridFilterPayload = Field(
        default_factory=HybridFilterPayload
    )
    generation_source: str = "none"


class TrackletMetadata(BaseModel):
    """Tracklet metadata emitted by Module 2."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    bboxes: list[list[int]] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None


class TrackletPayloadInput(BaseModel):
    """Module 2 Tracklet Payload consumed by Matching Engine."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    status: Literal["ready", "lost"]
    images: list[Any] = Field(default_factory=list)
    metadata: TrackletMetadata = Field(default_factory=TrackletMetadata)


class MatchingEngineRequest(BaseModel):
    """Runtime request joining Module 1 query and Module 2 tracklet."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    query: QueryUnderstandingPayload
    tracklet: TrackletPayloadInput
