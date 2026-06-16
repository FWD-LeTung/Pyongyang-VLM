"""Pydantic schemas for the query-understanding module.

The response object is intentionally shaped for the retrieval pipeline: one
payload feeds dense text search, while the other feeds structured/hybrid
filtering.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Status = Literal["success", "error", "rejected"]
Gender = Literal["male", "female", "unknown"]
GenerationSource = Literal["vertex_gemini", "local_fallback", "none"]


class Metadata(BaseModel):
    """Metadata about validation, language routing, and parser status."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    original_query: str
    language_detected: str
    status: Status
    error_code: str | None = None


class VectorSearchPayload(BaseModel):
    """Dense retrieval text normalized to CUHK-PEDES-style English."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    normalized_text: str = ""

    @field_validator("normalized_text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        """Coerce missing text to an empty string for rejected/error outputs."""

        if value is None:
            return ""
        return str(value).strip()


class HybridFilterPayload(BaseModel):
    """Structured attributes used by hybrid filtering."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    gender: Gender = "unknown"
    upper_color: str = "unknown"
    upper_type: str = "unknown"
    lower_color: str = "unknown"
    lower_type: str = "unknown"
    footwear: str = "unknown"
    accessory: str = "unknown"

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: object) -> Gender:
        """Normalize common model variants into the allowed gender enum."""

        if value is None:
            return "unknown"
        normalized = str(value).strip().lower()
        if normalized in {"male", "man", "boy", "gentleman"}:
            return "male"
        if normalized in {"female", "woman", "girl", "lady"}:
            return "female"
        return "unknown"

    @field_validator(
        "upper_color",
        "upper_type",
        "lower_color",
        "lower_type",
        "footwear",
        "accessory",
        mode="before",
    )
    @classmethod
    def default_unknown(cls, value: object) -> str:
        """Use the retrieval-safe sentinel for omitted or empty attributes."""

        if value is None:
            return "unknown"
        normalized = str(value).strip().lower()
        return normalized or "unknown"


class QueryUnderstandingResponse(BaseModel):
    """Root response model for Module 2 query understanding."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    metadata: Metadata
    vector_search_payload: VectorSearchPayload = Field(
        default_factory=VectorSearchPayload
    )
    hybrid_filter_payload: HybridFilterPayload = Field(
        default_factory=HybridFilterPayload
    )
    generation_source: GenerationSource = "none"

    @model_validator(mode="after")
    def enforce_failure_payloads(self) -> "QueryUnderstandingResponse":
        """Ensure rejected and error responses never leak stale retrieval text."""

        if self.metadata.status in {"error", "rejected"}:
            object.__setattr__(self.vector_search_payload, "normalized_text", "")
            if self.metadata.status == "rejected":
                object.__setattr__(self, "hybrid_filter_payload", HybridFilterPayload())
        if self.metadata.status == "success":
            object.__setattr__(self.metadata, "error_code", None)
        return self
