"""Sampling within temporal chunks for matching evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.matching_engine.schema import TrackletChunk


@dataclass(frozen=True)
class SampledChunk:
    """Sampled images and aligned metadata from one temporal chunk."""

    track_id: int
    chunk_id: int
    status: str
    images: list[Any] = field(default_factory=list)
    sampled_indices: list[int] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    bboxes: list[list[int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None


class ChunkSampler:
    """Sample representative crops inside a single temporal chunk."""

    def __init__(
        self,
        *,
        samples_per_chunk: int = 3,
        max_samples_per_chunk: int = 5,
        min_images_per_chunk: int = 1,
        sampling_strategy: str = "quality_even",
    ) -> None:
        self.samples_per_chunk = max(1, int(samples_per_chunk))
        self.max_samples_per_chunk = max(1, int(max_samples_per_chunk))
        self.min_images_per_chunk = max(0, int(min_images_per_chunk))
        self.sampling_strategy = sampling_strategy

    def sample(self, chunk: TrackletChunk) -> SampledChunk:
        """Return sampled crops without mutating ``chunk``."""

        num_images = len(chunk.images)
        if num_images < self.min_images_per_chunk:
            return self._empty(chunk)

        target = min(self.samples_per_chunk, self.max_samples_per_chunk)
        if num_images <= target:
            indices = list(range(num_images))
        elif self.sampling_strategy == "quality_even":
            indices = self._quality_even_indices(chunk, target)
        else:
            indices = self._even_indices(num_images, target)

        return SampledChunk(
            track_id=chunk.track_id,
            chunk_id=chunk.chunk_id,
            status=chunk.status,
            images=[chunk.images[index] for index in indices],
            sampled_indices=list(indices),
            frame_ids=_sample_if_aligned(chunk.frame_ids, indices, num_images),
            bboxes=_sample_if_aligned(chunk.bboxes, indices, num_images),
            timestamps=_sample_if_aligned(chunk.timestamps, indices, num_images),
            confidence_scores=_sample_if_aligned(
                chunk.confidence_scores,
                indices,
                num_images,
            ),
            first_seen=chunk.first_seen,
            last_seen=chunk.last_seen,
        )

    def _empty(self, chunk: TrackletChunk) -> SampledChunk:
        return SampledChunk(
            track_id=chunk.track_id,
            chunk_id=chunk.chunk_id,
            status=chunk.status,
            first_seen=chunk.first_seen,
            last_seen=chunk.last_seen,
        )

    def _quality_even_indices(self, chunk: TrackletChunk, count: int) -> list[int]:
        """Balance crop confidence against temporal coverage."""

        num_images = len(chunk.images)
        if len(chunk.confidence_scores) != num_images:
            return self._even_indices(num_images, count)

        if count == 1:
            best_index = max(
                range(num_images),
                key=lambda index: float(chunk.confidence_scores[index]),
            )
            return [best_index]

        anchors = self._even_indices(num_images, count)
        max_distance = max(1, num_images - 1)
        min_conf = min(float(score) for score in chunk.confidence_scores)
        max_conf = max(float(score) for score in chunk.confidence_scores)
        conf_span = max(max_conf - min_conf, 1e-6)

        scored: list[tuple[float, int]] = []
        for index, confidence in enumerate(chunk.confidence_scores):
            nearest_anchor_distance = min(abs(index - anchor) for anchor in anchors)
            even_score = 1.0 - (nearest_anchor_distance / max_distance)
            confidence_score = (float(confidence) - min_conf) / conf_span
            score = (0.65 * confidence_score) + (0.35 * even_score)
            scored.append((score, index))

        selected = [index for _score, index in sorted(scored, reverse=True)[:count]]
        return sorted(selected)

    @staticmethod
    def _even_indices(num_images: int, count: int) -> list[int]:
        if count <= 0 or num_images <= 0:
            return []
        if count == 1:
            return [num_images // 2]
        return sorted(
            {
                round(position * (num_images - 1) / (count - 1))
                for position in range(count)
            }
        )


def _sample_if_aligned(values: list[Any], indices: list[int], expected_len: int) -> list[Any]:
    """Sample metadata only when it is fully aligned with images."""

    if len(values) != expected_len:
        return []
    return [values[index] for index in indices]
