"""Shared query logic for temporary video embedding indexes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.matching_engine.config import load_matching_engine_config
from src.matching_engine.pipeline import MatchingEnginePipeline
from src.matching_engine.schema import QueryUnderstandingPayload
from src.query_understanding.llm_parser import QueryParser
from src.query_understanding.schema import QueryUnderstandingResponse


def query_video_index(
    *,
    index_path: str | Path,
    raw_query: str,
    matching_config: str = "config/matching_engine.yaml",
    checkpoint: str | None = None,
    device: str = "cpu",
    precision: str = "fp32",
    score_topk: int = 3,
) -> dict[str, Any]:
    """Run Module 1 + text retrieval against a saved video embedding index."""

    resolved_device = resolve_device(device)
    data = load_index(Path(index_path))
    warnings = warn_if_config_differs(
        matching_config=matching_config,
        checkpoint=checkpoint,
        data=data,
    )

    query_payload = run_module1(raw_query)
    normalized_text = query_payload.vector_search_payload.normalized_text
    text_emb = encode_text(
        normalized_text,
        matching_config=matching_config,
        checkpoint=checkpoint,
        device=resolved_device,
        precision=precision,
    )

    image_embs = F.normalize(data["embeddings"].float(), dim=1).to(text_emb.device)
    similarities = (text_emb.float() @ image_embs.T).squeeze(0).detach().cpu()
    ranking = rank_tracks(
        similarities=similarities,
        track_ids=[int(value) for value in data["track_ids"]],
        frame_ids=data["frame_ids"],
        bboxes=data["bboxes"],
        score_topk=score_topk,
    )

    best_track_id = ranking[0]["track_id"] if ranking else None
    best_score = ranking[0]["score"] if ranking else 0.0
    top1_top2_margin = (
        ranking[0]["score"] - ranking[1]["score"]
        if len(ranking) >= 2
        else best_score
    )

    return {
        "query_payload": query_payload,
        "index_data": data,
        "ranking": ranking,
        "best_track_id": best_track_id,
        "best_score": best_score,
        "top1_top2_margin": float(top1_top2_margin),
        "warnings": warnings,
    }


def resolve_device(raw_device: str) -> str:
    """Resolve CLI-style device strings and fail fast on unavailable CUDA."""

    if raw_device.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {raw_device} was requested but CUDA is unavailable."
            )
        return f"cuda:{raw_device}"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {raw_device} was requested but torch.cuda.is_available() is False."
        )
    return raw_device


def load_index(path: Path) -> dict[str, Any]:
    """Load and validate a temporary torch-saved video index."""

    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    validate_index(data)
    return data


def validate_index(data: dict[str, Any]) -> None:
    """Check the minimal shape/alignment contract needed for query ranking."""

    if not isinstance(data, dict):
        raise ValueError("Index must be a dictionary saved by torch.save().")

    required_fields = [
        "embeddings",
        "track_ids",
        "chunk_ids",
        "crop_indices",
        "frame_ids",
        "bboxes",
        "timestamps",
        "confidence_scores",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Index is missing required field(s): {missing}")

    embeddings = data["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Index embeddings must be a 2D torch.Tensor.")

    num_embeddings = int(embeddings.shape[0])
    mismatched = {
        field: len(data[field])
        for field in required_fields[1:]
        if len(data[field]) != num_embeddings
    }
    if mismatched:
        raise ValueError(
            "Index metadata is not aligned with embeddings: "
            f"num_embeddings={num_embeddings} lengths={mismatched}"
        )
    if num_embeddings == 0:
        raise ValueError("Index contains no embeddings.")


def warn_if_config_differs(
    *,
    matching_config: str,
    checkpoint: str | None,
    data: dict[str, Any],
) -> list[str]:
    """Return warning messages for query/export config mismatches."""

    warnings: list[str] = []
    exported_matching_config = data.get("config", {}).get("matching_config")
    if exported_matching_config and str(matching_config) != str(
        exported_matching_config
    ):
        warnings.append(
            "WARNING: matching-config argument differs from exported index config."
        )

    checkpoint_arg = effective_checkpoint(
        matching_config=matching_config,
        checkpoint=checkpoint,
    )
    exported_checkpoint = data.get("config", {}).get("checkpoint")
    if exported_checkpoint and str(checkpoint_arg) != str(exported_checkpoint):
        warnings.append(
            "WARNING: checkpoint argument differs from exported index checkpoint."
        )
    return warnings


def effective_checkpoint(
    *,
    matching_config: str,
    checkpoint: str | None,
) -> str:
    """Return the checkpoint path that will actually be used for text encoding."""

    if checkpoint is not None:
        return str(checkpoint)
    config = load_matching_engine_config(matching_config)
    return str(config.retrieval.checkpoint_path)


def run_module1(raw_query: str) -> QueryUnderstandingPayload:
    """Run Query Understanding and adapt its response for Matching Engine."""

    response: QueryUnderstandingResponse = QueryParser().parse(raw_query)
    if response.metadata.status != "success":
        raise RuntimeError(
            "Module 1 failed: "
            f"status={response.metadata.status} "
            f"error_code={response.metadata.error_code}"
        )
    query_payload = QueryUnderstandingPayload.model_validate(
        response.model_dump(mode="json")
    )
    if not query_payload.vector_search_payload.normalized_text.strip():
        raise RuntimeError("Module 1 returned an empty normalized_text.")
    return query_payload


def encode_text(
    normalized_text: str,
    *,
    matching_config: str,
    checkpoint: str | None,
    device: str,
    precision: str,
) -> torch.Tensor:
    """Encode normalized query text with the configured TBPS-CLIP encoder."""

    matching = MatchingEnginePipeline.from_config_file(
        matching_config,
        checkpoint_path=checkpoint,
        device=device,
        precision=precision,
    )
    text_emb = matching.encoder.encode_text([normalized_text])
    return F.normalize(text_emb.float(), dim=1)


def rank_tracks(
    *,
    similarities: torch.Tensor,
    track_ids: list[int],
    frame_ids: list[int | None],
    bboxes: list[list[int]],
    score_topk: int,
) -> list[dict[str, Any]]:
    """Aggregate image-level similarities into track-level top-k mean scores."""

    rows_by_track: dict[int, list[int]] = defaultdict(list)
    for row_index, track_id in enumerate(track_ids):
        rows_by_track[track_id].append(row_index)

    ranking: list[dict[str, Any]] = []
    for track_id, row_indices in rows_by_track.items():
        track_scores = similarities[row_indices]
        top_count = min(max(1, score_topk), len(row_indices))
        top_values = torch.topk(track_scores, k=top_count).values
        best_local_index = int(torch.argmax(track_scores).item())
        best_row_index = row_indices[best_local_index]
        ranking.append(
            {
                "rank": 0,
                "track_id": track_id,
                "score": float(top_values.mean().item()),
                "best_frame_id": get_or_none(frame_ids, best_row_index),
                "best_bbox": get_or_default(bboxes, best_row_index, []),
                "evidence": len(row_indices),
            }
        )

    ranked = sorted(ranking, key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def get_or_none(values: list[Any], index: int) -> Any | None:
    """Return a list value or None when metadata is missing."""

    return values[index] if index < len(values) else None


def get_or_default(values: list[Any], index: int, default: Any) -> Any:
    """Return a list value or a caller-provided default when missing."""

    return values[index] if index < len(values) else default
