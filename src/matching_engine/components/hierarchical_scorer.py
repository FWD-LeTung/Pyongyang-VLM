"""Hierarchical query-to-track scoring for temporal chunk evidence."""

from __future__ import annotations

import torch

from src.matching_engine.schema import ChunkMatchResult


class HierarchicalScorer:
    """Score chunks first, then aggregate top chunks into a track score."""

    def score_chunk(
        self,
        query_emb: torch.Tensor,
        chunk_image_embs: torch.Tensor,
        *,
        crop_topk: int = 2,
    ) -> tuple[float, int | None]:
        """Return ``mean(top-k crop similarities)`` and best sampled index."""

        if chunk_image_embs.numel() == 0 or chunk_image_embs.shape[0] == 0:
            return 0.0, None

        query = query_emb
        if query.ndim == 2:
            query = query[0]
        query = query.to(
            device=chunk_image_embs.device,
            dtype=chunk_image_embs.dtype,
        )
        similarities = query.reshape(1, -1) @ chunk_image_embs.T
        similarities = similarities.reshape(-1).float()
        if similarities.numel() == 0:
            return 0.0, None

        k = max(1, min(int(crop_topk), similarities.numel()))
        top_values, top_indices = torch.topk(similarities, k=k)
        best_sampled_index = int(top_indices[0].item())
        return float(top_values.mean().item()), best_sampled_index

    def score_track(
        self,
        chunk_scores: list[ChunkMatchResult],
        *,
        top_chunks: int = 5,
        aggregation: str = "top_chunks_mean",
    ) -> float:
        """Aggregate chunk scores into one track-level score."""

        valid_scores = [score.score for score in chunk_scores if score.num_samples > 0]
        if not valid_scores:
            return 0.0
        if aggregation != "top_chunks_mean":
            raise ValueError(f"Unsupported track aggregation: {aggregation}")

        count = max(1, min(int(top_chunks), len(valid_scores)))
        top_values = sorted(valid_scores, reverse=True)[:count]
        return float(sum(top_values) / len(top_values))
