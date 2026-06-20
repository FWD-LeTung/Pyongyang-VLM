"""Pydantic contracts for Matching Engine runtime inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    frame_ids: list[int] = Field(default_factory=list)
    bboxes: list[list[int]] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    timeline_frame_ids: list[int] = Field(default_factory=list)
    timeline_bboxes: list[list[int]] = Field(default_factory=list)
    timeline_timestamps: list[float] = Field(default_factory=list)
    timeline_confidence_scores: list[float] = Field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None


class TrackletPayloadInput(BaseModel):
    """Module 2 Tracklet Payload consumed by Matching Engine."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    status: Literal["ready", "lost"]
    images: list[Any] = Field(default_factory=list)
    metadata: TrackletMetadata = Field(default_factory=TrackletMetadata)


class TrackletChunk(BaseModel):
    """One temporal evidence chunk for a single track ID."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    chunk_id: int
    status: str = ""
    images: list[Any] = Field(default_factory=list)
    frame_ids: list[int] = Field(default_factory=list)
    bboxes: list[list[int]] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None


class TrackCandidate(BaseModel):
    """A person candidate grouped by track ID with temporal chunks intact."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    status: str = ""
    chunks: list[TrackletChunk] = Field(default_factory=list)
    timeline_frame_ids: list[int] = Field(default_factory=list)
    timeline_bboxes: list[list[int]] = Field(default_factory=list)
    timeline_timestamps: list[float] = Field(default_factory=list)
    timeline_confidence_scores: list[float] = Field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None


class ChunkMatchResult(BaseModel):
    """Debug score for one temporal chunk."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    chunk_id: int
    score: float
    num_samples: int
    best_crop_index: int | None = None
    first_seen: float | None = None
    last_seen: float | None = None


class TrackletMatchResult(BaseModel):
    """Ranked matching result for one track ID candidate."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    rank: int
    track_id: int
    score: float
    num_chunks: int
    num_sampled_crops: int
    best_chunk_id: int | None = None
    best_crop_index: int | None = None
    status: str = ""
    first_seen: float | None = None
    last_seen: float | None = None
    top_chunks: list[ChunkMatchResult] = Field(default_factory=list)


class SelectedTrackTimeline(BaseModel):
    """Timeline metadata needed by a renderer for the selected track."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    track_id: int
    frame_ids: list[int] = Field(default_factory=list)
    bboxes: list[list[int]] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)


class MatchingEngineResponse(BaseModel):
    """Production response from Module 3 Phase 2."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    best_track_id: int | None = None
    best_score: float = 0.0
    ranking: list[TrackletMatchResult] = Field(default_factory=list)
    selected_track: SelectedTrackTimeline | None = None
    status: Literal["success", "no_candidate", "invalid_query", "error"] = "success"
    message: str = ""


class MatchingEngineRequest(BaseModel):
    """Runtime request joining Module 1 query and Module 2 tracklet chunks."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    query: QueryUnderstandingPayload
    tracklets: list[TrackletPayloadInput] = Field(default_factory=list)
    video_id: str | None = None
    session_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_tracklet(cls, values: Any) -> Any:
        """Accept the old single-tracklet shape while preferring tracklets."""

        if not isinstance(values, dict):
            return values
        if values.get("tracklets"):
            return values
        legacy_tracklet = values.get("tracklet")
        if legacy_tracklet is not None:
            upgraded = dict(values)
            upgraded["tracklets"] = [legacy_tracklet]
            return upgraded
        return values
