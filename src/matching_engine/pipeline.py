"""Production inference pipeline for Matching Engine Phase 2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.matching_engine.components.chunk_sampler import ChunkSampler, SampledChunk
from src.matching_engine.components.embedding_cache import (
    CachedChunkEmbeddings,
    EmbeddingCache,
    cache_scope,
)
from src.matching_engine.components.hierarchical_scorer import HierarchicalScorer
from src.matching_engine.components.track_candidate_builder import TrackCandidateBuilder
from src.matching_engine.config import MatchingEngineConfig, load_matching_engine_config
from src.matching_engine.core.retrieval_encoder import RetrievalEncoder
from src.matching_engine.schema import (
    ChunkMatchResult,
    MatchingEngineRequest,
    MatchingEngineResponse,
    SelectedTrackTimeline,
    TrackCandidate,
    TrackletChunk,
    TrackletMatchResult,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class MatchingEnginePipeline:
    """Query-to-track_id matching with temporal chunk retrieval evidence."""

    def __init__(
        self,
        *,
        encoder: RetrievalEncoder,
        config: MatchingEngineConfig | None = None,
        candidate_builder: TrackCandidateBuilder | None = None,
        chunk_sampler: ChunkSampler | None = None,
        scorer: HierarchicalScorer | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.config = config or MatchingEngineConfig()
        self.encoder = encoder
        self.candidate_builder = candidate_builder or TrackCandidateBuilder()
        self.chunk_sampler = chunk_sampler or ChunkSampler(
            samples_per_chunk=self.config.chunk.samples_per_chunk,
            max_samples_per_chunk=self.config.chunk.max_samples_per_chunk,
            min_images_per_chunk=self.config.chunk.min_images_per_chunk,
            sampling_strategy=self.config.chunk.sampling_strategy,
        )
        self.scorer = scorer or HierarchicalScorer()
        self.cache = cache or EmbeddingCache(
            enabled=self.config.cache.enabled,
            dtype=self.config.cache.dtype,
            device=self.config.cache.device,
            storage=self.config.cache.storage,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "config/matching_engine.yaml",
        *,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        precision: str | None = None,
    ) -> "MatchingEnginePipeline":
        """Create a TBPS-CLIP backed pipeline from YAML config."""

        config = load_matching_engine_config(config_path)
        retrieval = config.retrieval
        if checkpoint_path is not None or device is not None or precision is not None:
            retrieval = replace(
                retrieval,
                checkpoint_path=str(checkpoint_path or retrieval.checkpoint_path),
                device=str(device or retrieval.device),
                precision=str(precision or retrieval.precision),
            )
            config = replace(config, retrieval=retrieval)
        if retrieval.backend != "tbps_clip":
            raise ValueError(f"Unsupported retrieval backend: {retrieval.backend}")

        from src.matching_engine.core.tbps_clip_encoder import TBPSCLIPEncoder

        encoder = TBPSCLIPEncoder(
            checkpoint_path=retrieval.checkpoint_path,
            config_path=retrieval.config_path,
            tbps_root=retrieval.tbps_root,
            device=retrieval.device,
            precision=retrieval.precision,
            batch_size=config.runtime.encode_batch_size,
        )
        return cls(encoder=encoder, config=config)

    def run(self, request: MatchingEngineRequest) -> MatchingEngineResponse:
        """Run production matching and return a ranked track-level response."""

        total_start = perf_counter()
        if request.query.metadata.status != "success":
            return MatchingEngineResponse(
                status="invalid_query",
                message="Query understanding payload status is not success.",
            )

        normalized_text = request.query.vector_search_payload.normalized_text.strip()
        if not normalized_text:
            return MatchingEngineResponse(
                status="invalid_query",
                message="Query normalized_text is empty.",
            )
        if not request.tracklets:
            return MatchingEngineResponse(
                status="no_candidate",
                message="No Module 2 tracklet chunks were provided.",
            )

        logger.info("Matching input payload chunks: %s", len(request.tracklets))
        build_start = perf_counter()
        candidates = self.candidate_builder.build(request.tracklets)
        valid_candidates = [
            candidate for candidate in candidates if self._candidate_is_valid(candidate)
        ]
        total_chunks = sum(len(candidate.chunks) for candidate in valid_candidates)
        logger.info(
            "Built %s track_id candidate(s), %s valid, %s chunk(s).",
            len(candidates),
            len(valid_candidates),
            total_chunks,
        )
        if not valid_candidates:
            return MatchingEngineResponse(
                status="no_candidate",
                message="No valid candidates with image evidence were found.",
            )

        cache_start = perf_counter()
        chunk_items: dict[tuple[int, int], CachedChunkEmbeddings] = {}
        sampled_crops = 0
        cached_chunks = 0
        encoded_chunks = 0
        for candidate in valid_candidates:
            for chunk in candidate.chunks:
                item, from_cache = self._get_or_encode_chunk(
                    chunk,
                    video_id=request.video_id,
                    session_id=request.session_id,
                )
                if item is None:
                    continue
                chunk_items[(candidate.track_id, chunk.chunk_id)] = item
                sampled_crops += len(item.sampled_indices)
                cached_chunks += int(from_cache)
                encoded_chunks += int(not from_cache)

        if not chunk_items:
            return MatchingEngineResponse(
                status="no_candidate",
                message="No chunk produced sampleable image evidence.",
            )

        query_start = perf_counter()
        query_emb = self.encoder.encode_text([normalized_text])
        query_time = perf_counter() - query_start

        score_start = perf_counter()
        ranking = self._score_candidates(valid_candidates, chunk_items, query_emb)
        score_time = perf_counter() - score_start
        if not ranking:
            return MatchingEngineResponse(
                status="no_candidate",
                message="No candidate produced a valid score.",
            )

        selected_candidate = self._candidate_by_track_id(
            valid_candidates,
            ranking[0].track_id,
        )
        selected_track = self._selected_timeline(selected_candidate)
        cache_time = query_start - cache_start
        build_time = cache_start - build_start
        total_time = perf_counter() - total_start
        logger.info(
            "Matching done: chunks_in=%s candidates=%s valid=%s sampled_crops=%s "
            "cache_hits=%s encoded_chunks=%s build=%.4fs cache=%.4fs "
            "query=%.4fs score=%.4fs total=%.4fs best_track_id=%s best_score=%.6f",
            len(request.tracklets),
            len(candidates),
            len(valid_candidates),
            sampled_crops,
            cached_chunks,
            encoded_chunks,
            build_time,
            cache_time,
            query_time,
            score_time,
            total_time,
            ranking[0].track_id,
            ranking[0].score,
        )
        return MatchingEngineResponse(
            best_track_id=ranking[0].track_id,
            best_score=ranking[0].score,
            ranking=ranking,
            selected_track=selected_track,
            status="success",
            message="Matched query to track_id using temporal chunk evidence.",
        )

    def _get_or_encode_chunk(
        self,
        chunk: TrackletChunk,
        *,
        video_id: str | None,
        session_id: str | None,
    ) -> tuple[CachedChunkEmbeddings | None, bool]:
        cached = self.cache.get(
            video_id=video_id,
            session_id=session_id,
            track_id=chunk.track_id,
            chunk_id=chunk.chunk_id,
        )
        if cached is not None:
            return cached, True

        sampled = self.chunk_sampler.sample(chunk)
        if not sampled.images:
            return None, False

        image_embeddings = self.encoder.encode_images(sampled.images)
        item = self._cache_item_from_sampled(
            sampled,
            embeddings=image_embeddings,
            video_id=video_id,
            session_id=session_id,
        )
        stored = self.cache.set(item)
        return stored, False

    def _score_candidates(
        self,
        candidates: list[TrackCandidate],
        chunk_items: dict[tuple[int, int], CachedChunkEmbeddings],
        query_emb: torch.Tensor,
    ) -> list[TrackletMatchResult]:
        results: list[TrackletMatchResult] = []
        for candidate in candidates:
            chunk_scores: list[ChunkMatchResult] = []
            num_sampled_crops = 0
            for chunk in candidate.chunks:
                item = chunk_items.get((candidate.track_id, chunk.chunk_id))
                if item is None:
                    continue
                score, best_sampled_index = self.scorer.score_chunk(
                    query_emb,
                    item.embeddings,
                    crop_topk=self.config.track_scoring.crop_topk,
                )
                best_crop_index = None
                if best_sampled_index is not None and best_sampled_index < len(
                    item.sampled_indices
                ):
                    best_crop_index = item.sampled_indices[best_sampled_index]
                chunk_scores.append(
                    ChunkMatchResult(
                        track_id=candidate.track_id,
                        chunk_id=chunk.chunk_id,
                        score=score,
                        num_samples=len(item.sampled_indices),
                        best_crop_index=best_crop_index,
                        first_seen=chunk.first_seen,
                        last_seen=chunk.last_seen,
                    )
                )
                num_sampled_crops += len(item.sampled_indices)

            if not chunk_scores:
                continue
            sorted_chunks = sorted(
                chunk_scores,
                key=lambda chunk_score: chunk_score.score,
                reverse=True,
            )
            track_score = self.scorer.score_track(
                sorted_chunks,
                top_chunks=self.config.track_scoring.top_chunks,
                aggregation=self.config.track_scoring.aggregation,
            )
            best_chunk = sorted_chunks[0]
            results.append(
                TrackletMatchResult(
                    rank=0,
                    track_id=candidate.track_id,
                    score=track_score,
                    num_chunks=len(candidate.chunks),
                    num_sampled_crops=num_sampled_crops,
                    best_chunk_id=best_chunk.chunk_id,
                    best_crop_index=best_chunk.best_crop_index,
                    status=candidate.status,
                    first_seen=candidate.first_seen,
                    last_seen=candidate.last_seen,
                    top_chunks=sorted_chunks[: self.config.track_scoring.top_chunks],
                )
            )

        ranked = sorted(results, key=lambda result: result.score, reverse=True)
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(ranked, start=1)
        ]

    def _candidate_is_valid(self, candidate: TrackCandidate) -> bool:
        total_images = sum(len(chunk.images) for chunk in candidate.chunks)
        total_chunks_with_images = sum(1 for chunk in candidate.chunks if chunk.images)
        if total_images < self.config.candidate.min_total_images:
            return False
        return total_chunks_with_images >= self.config.candidate.min_total_chunks

    @staticmethod
    def _candidate_by_track_id(
        candidates: list[TrackCandidate],
        track_id: int,
    ) -> TrackCandidate:
        for candidate in candidates:
            if candidate.track_id == track_id:
                return candidate
        raise KeyError(f"Track candidate not found: {track_id}")

    @staticmethod
    def _selected_timeline(candidate: TrackCandidate) -> SelectedTrackTimeline:
        return SelectedTrackTimeline(
            track_id=candidate.track_id,
            frame_ids=list(candidate.timeline_frame_ids),
            bboxes=[list(map(int, bbox)) for bbox in candidate.timeline_bboxes],
            timestamps=[float(ts) for ts in candidate.timeline_timestamps],
            confidence_scores=[
                float(score) for score in candidate.timeline_confidence_scores
            ],
        )

    @staticmethod
    def _cache_item_from_sampled(
        sampled: SampledChunk,
        *,
        embeddings: torch.Tensor,
        video_id: str | None,
        session_id: str | None,
    ) -> CachedChunkEmbeddings:
        return CachedChunkEmbeddings(
            scope=cache_scope(video_id=video_id, session_id=session_id),
            track_id=sampled.track_id,
            chunk_id=sampled.chunk_id,
            embeddings=embeddings,
            sampled_indices=list(sampled.sampled_indices),
            frame_ids=list(sampled.frame_ids),
            bboxes=[list(map(int, bbox)) for bbox in sampled.bboxes],
            timestamps=[float(ts) for ts in sampled.timestamps],
            confidence_scores=[float(score) for score in sampled.confidence_scores],
            first_seen=sampled.first_seen,
            last_seen=sampled.last_seen,
            status=sampled.status,
        )


def build_pipeline_from_config(
    config_path: str | Path = "config/matching_engine.yaml",
    **overrides: Any,
) -> MatchingEnginePipeline:
    """Convenience factory for scripts and notebooks."""

    return MatchingEnginePipeline.from_config_file(config_path, **overrides)
