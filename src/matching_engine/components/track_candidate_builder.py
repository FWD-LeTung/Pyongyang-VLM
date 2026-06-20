"""Build track-level candidates while preserving temporal chunks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from src.matching_engine.schema import (
    TrackCandidate,
    TrackletChunk,
    TrackletPayloadInput,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class TrackCandidateBuilder:
    """Group Module 2 payload chunks by track ID without flattening crops."""

    def build(self, tracklets: Sequence[TrackletPayloadInput]) -> list[TrackCandidate]:
        """Convert raw Module 2 chunks into grouped track candidates."""

        grouped: dict[int, list[TrackletChunk]] = defaultdict(list)
        timelines: dict[int, _TimelineAccumulator] = defaultdict(_TimelineAccumulator)

        for payload in tracklets:
            chunk = self._payload_to_chunk(payload, len(grouped[payload.track_id]))
            grouped[payload.track_id].append(chunk)
            timelines[payload.track_id].add_payload(payload, chunk)

        candidates: list[TrackCandidate] = []
        for track_id, chunks in grouped.items():
            sorted_chunks = sorted(
                chunks,
                key=lambda chunk: (
                    float("inf") if chunk.first_seen is None else chunk.first_seen,
                    chunk.chunk_id,
                ),
            )
            renumbered = [
                chunk.model_copy(update={"chunk_id": index})
                for index, chunk in enumerate(sorted_chunks)
            ]
            first_seen_values = [
                chunk.first_seen for chunk in renumbered if chunk.first_seen is not None
            ]
            last_seen_values = [
                chunk.last_seen for chunk in renumbered if chunk.last_seen is not None
            ]
            timeline = timelines[track_id]
            candidates.append(
                TrackCandidate(
                    track_id=track_id,
                    status=renumbered[-1].status if renumbered else "",
                    chunks=renumbered,
                    timeline_frame_ids=timeline.frame_ids,
                    timeline_bboxes=timeline.bboxes,
                    timeline_timestamps=timeline.timestamps,
                    timeline_confidence_scores=timeline.confidence_scores,
                    first_seen=min(first_seen_values) if first_seen_values else None,
                    last_seen=max(last_seen_values) if last_seen_values else None,
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.track_id)

    def _payload_to_chunk(
        self,
        payload: TrackletPayloadInput,
        chunk_id: int,
    ) -> TrackletChunk:
        metadata = payload.metadata
        images = list(payload.images)
        num_images = len(images)

        frame_ids = list(metadata.frame_ids)
        if not frame_ids:
            logger.warning(
                "Track %s chunk %s has no frame_ids; renderer should prefer future "
                "timeline metadata when available.",
                payload.track_id,
                chunk_id,
            )

        self._warn_if_misaligned(
            track_id=payload.track_id,
            chunk_id=chunk_id,
            field_name="frame_ids",
            values=frame_ids,
            num_images=num_images,
        )
        self._warn_if_misaligned(
            track_id=payload.track_id,
            chunk_id=chunk_id,
            field_name="bboxes",
            values=metadata.bboxes,
            num_images=num_images,
        )
        self._warn_if_misaligned(
            track_id=payload.track_id,
            chunk_id=chunk_id,
            field_name="timestamps",
            values=metadata.timestamps,
            num_images=num_images,
        )
        self._warn_if_misaligned(
            track_id=payload.track_id,
            chunk_id=chunk_id,
            field_name="confidence_scores",
            values=metadata.confidence_scores,
            num_images=num_images,
        )

        first_seen = metadata.first_seen
        if first_seen is None and metadata.timestamps:
            first_seen = min(float(ts) for ts in metadata.timestamps)
        last_seen = metadata.last_seen
        if last_seen is None and metadata.timestamps:
            last_seen = max(float(ts) for ts in metadata.timestamps)

        return TrackletChunk(
            track_id=payload.track_id,
            chunk_id=chunk_id,
            status=payload.status,
            images=images,
            frame_ids=[int(frame_id) for frame_id in frame_ids],
            bboxes=[list(map(int, bbox)) for bbox in metadata.bboxes],
            timestamps=[float(ts) for ts in metadata.timestamps],
            confidence_scores=[float(score) for score in metadata.confidence_scores],
            first_seen=first_seen,
            last_seen=last_seen,
        )

    @staticmethod
    def _warn_if_misaligned(
        *,
        track_id: int,
        chunk_id: int,
        field_name: str,
        values: Sequence[object],
        num_images: int,
    ) -> None:
        if values and len(values) != num_images:
            logger.warning(
                "Track %s chunk %s metadata %s length %s does not match images "
                "length %s.",
                track_id,
                chunk_id,
                field_name,
                len(values),
                num_images,
            )


class _TimelineAccumulator:
    """Collect the best available renderer timeline for a track."""

    def __init__(self) -> None:
        self.frame_ids: list[int] = []
        self.bboxes: list[list[int]] = []
        self.timestamps: list[float] = []
        self.confidence_scores: list[float] = []

    def add_payload(
        self,
        payload: TrackletPayloadInput,
        chunk: TrackletChunk,
    ) -> None:
        metadata = payload.metadata
        frame_ids = metadata.timeline_frame_ids or chunk.frame_ids
        bboxes = metadata.timeline_bboxes or chunk.bboxes
        timestamps = metadata.timeline_timestamps or chunk.timestamps
        confidence_scores = (
            metadata.timeline_confidence_scores or chunk.confidence_scores
        )

        self.frame_ids.extend(int(frame_id) for frame_id in frame_ids)
        self.bboxes.extend(list(map(int, bbox)) for bbox in bboxes)
        self.timestamps.extend(float(ts) for ts in timestamps)
        self.confidence_scores.extend(float(score) for score in confidence_scores)
