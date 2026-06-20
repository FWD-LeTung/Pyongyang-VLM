"""Configuration objects for Matching Engine production inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RetrievalConfig:
    """TBPS-CLIP retrieval backend settings."""

    backend: str = "tbps_clip"
    tbps_root: str = "src/matching_engine/TBPS-CLIP"
    config_path: str = "src/matching_engine/TBPS-CLIP/config/config.yaml"
    checkpoint_path: str = "weights/checkpoint_best.pth"
    device: str = "cuda"
    precision: str = "fp16"


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate-level quality gates that do not use semantic hard filters."""

    min_total_images: int = 3
    min_total_chunks: int = 1


@dataclass(frozen=True)
class ChunkConfig:
    """Temporal chunk sampling settings."""

    samples_per_chunk: int = 3
    max_samples_per_chunk: int = 5
    min_images_per_chunk: int = 1
    sampling_strategy: str = "quality_even"


@dataclass(frozen=True)
class CacheConfig:
    """Image embedding cache settings."""

    enabled: bool = True
    dtype: str = "fp16"
    device: str = "cuda"
    storage: str = "memory"


@dataclass(frozen=True)
class TrackScoringConfig:
    """Hierarchical retrieval scoring settings."""

    crop_topk: int = 2
    top_chunks: int = 5
    aggregation: str = "top_chunks_mean"


@dataclass(frozen=True)
class ConfidenceConfig:
    """Score diagnostics for ambiguous matches."""

    ambiguous_margin_threshold: float = 0.005


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime batching settings."""

    encode_batch_size: int = 32


@dataclass(frozen=True)
class MatchingEngineConfig:
    """Root config for Module 3 Phase 2 inference."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    track_scoring: TrackScoringConfig = field(default_factory=TrackScoringConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any] | None) -> "MatchingEngineConfig":
        """Build config from a YAML-compatible mapping."""

        data = raw_config or {}
        retrieval = _as_dict(data.get("retrieval"))
        candidate = _as_dict(data.get("candidate"))
        chunk = _as_dict(data.get("chunk"))
        cache = _as_dict(data.get("cache"))
        scoring = _as_dict(data.get("track_scoring"))
        confidence = _as_dict(data.get("confidence"))
        runtime = _as_dict(data.get("runtime"))

        return cls(
            retrieval=RetrievalConfig(
                backend=str(retrieval.get("backend", RetrievalConfig.backend)),
                tbps_root=str(retrieval.get("tbps_root", RetrievalConfig.tbps_root)),
                config_path=str(
                    retrieval.get("config_path", RetrievalConfig.config_path)
                ),
                checkpoint_path=str(
                    retrieval.get(
                        "checkpoint_path",
                        RetrievalConfig.checkpoint_path,
                    )
                ),
                device=str(retrieval.get("device", RetrievalConfig.device)),
                precision=str(retrieval.get("precision", RetrievalConfig.precision)),
            ),
            candidate=CandidateConfig(
                min_total_images=int(
                    candidate.get(
                        "min_total_images",
                        CandidateConfig.min_total_images,
                    )
                ),
                min_total_chunks=int(
                    candidate.get(
                        "min_total_chunks",
                        CandidateConfig.min_total_chunks,
                    )
                ),
            ),
            chunk=ChunkConfig(
                samples_per_chunk=int(
                    chunk.get("samples_per_chunk", ChunkConfig.samples_per_chunk)
                ),
                max_samples_per_chunk=int(
                    chunk.get(
                        "max_samples_per_chunk",
                        ChunkConfig.max_samples_per_chunk,
                    )
                ),
                min_images_per_chunk=int(
                    chunk.get("min_images_per_chunk", ChunkConfig.min_images_per_chunk)
                ),
                sampling_strategy=str(
                    chunk.get("sampling_strategy", ChunkConfig.sampling_strategy)
                ),
            ),
            cache=CacheConfig(
                enabled=bool(cache.get("enabled", CacheConfig.enabled)),
                dtype=str(cache.get("dtype", CacheConfig.dtype)),
                device=str(cache.get("device", CacheConfig.device)),
                storage=str(cache.get("storage", CacheConfig.storage)),
            ),
            track_scoring=TrackScoringConfig(
                crop_topk=int(scoring.get("crop_topk", TrackScoringConfig.crop_topk)),
                top_chunks=int(
                    scoring.get("top_chunks", TrackScoringConfig.top_chunks)
                ),
                aggregation=str(
                    scoring.get("aggregation", TrackScoringConfig.aggregation)
                ),
            ),
            confidence=ConfidenceConfig(
                ambiguous_margin_threshold=float(
                    confidence.get(
                        "ambiguous_margin_threshold",
                        ConfidenceConfig.ambiguous_margin_threshold,
                    )
                )
            ),
            runtime=RuntimeConfig(
                encode_batch_size=int(
                    runtime.get(
                        "encode_batch_size",
                        RuntimeConfig.encode_batch_size,
                    )
                )
            ),
        )


def load_matching_engine_config(config_path: str | Path) -> MatchingEngineConfig:
    """Load Module 3 inference config from YAML."""

    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError("matching_engine config must be a YAML mapping.")
    return MatchingEngineConfig.from_dict(config_data)


def _as_dict(value: object) -> dict[str, Any]:
    """Return a mapping or an empty dict for missing config sections."""

    return value if isinstance(value, dict) else {}
