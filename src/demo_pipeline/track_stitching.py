"""Conservative query-conditioned track stitching helpers."""

from __future__ import annotations

from collections import defaultdict
from math import inf
from typing import Any

import torch
import torch.nn.functional as F


QUERY_SCORE_TOLERANCE = 0.08


def get_track_time_range(
    index_data: dict[str, Any],
    track_id: int,
) -> tuple[int | None, int | None]:
    """Return the first and last frame for a track, accepting int or str keys."""

    frames: list[int] = []
    timeline = get_timeline_or_none(index_data, track_id)
    if timeline is not None:
        frames.extend(coerce_frame_ids(timeline.get("frame_ids") or []))

    if not frames:
        rows_by_track = build_rows_by_track(index_data)
        frame_ids = index_data.get("frame_ids") or []
        for row_index in rows_by_track.get(int(track_id), []):
            if row_index >= len(frame_ids):
                continue
            raw_frame_id = frame_ids[row_index]
            if raw_frame_id is None:
                continue
            frames.append(int(raw_frame_id))

    if not frames:
        return None, None
    return min(frames), max(frames)


def build_rows_by_track(index_data: dict[str, Any]) -> dict[int, list[int]]:
    """Group embedding row indices by integer track ID."""

    rows_by_track: dict[int, list[int]] = defaultdict(list)
    for row_index, raw_track_id in enumerate(index_data.get("track_ids") or []):
        track_id = coerce_int_or_none(raw_track_id)
        if track_id is None:
            continue
        rows_by_track[track_id].append(row_index)
    return dict(rows_by_track)


def compute_track_appearance_score(
    *,
    embeddings: torch.Tensor,
    rows_a: list[int],
    rows_b: list[int],
    topk: int = 5,
) -> float:
    """Score two tracks by mean top-k normalized crop embedding similarity."""

    if not rows_a or not rows_b:
        return 0.0
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D torch.Tensor.")

    valid_rows_a = valid_embedding_rows(rows_a, int(embeddings.shape[0]))
    valid_rows_b = valid_embedding_rows(rows_b, int(embeddings.shape[0]))
    if not valid_rows_a or not valid_rows_b:
        return 0.0

    normalized = F.normalize(embeddings.float(), dim=1)
    sims = normalized[valid_rows_a] @ normalized[valid_rows_b].T
    if sims.numel() == 0:
        return 0.0

    top_count = min(max(1, int(topk)), int(sims.numel()))
    top_values = torch.topk(sims.flatten(), k=top_count).values
    return float(top_values.mean().item())


def compute_temporal_gap(
    *,
    range_a: tuple[int | None, int | None],
    range_b: tuple[int | None, int | None],
) -> dict[str, Any]:
    """Describe how track A is positioned in time relative to track B."""

    first_a, last_a = range_a
    first_b, last_b = range_b
    if first_a is None or last_a is None or first_b is None or last_b is None:
        return {
            "direction": "unknown",
            "gap_frames": None,
            "overlap_frames": 0,
        }

    if first_a > last_a:
        first_a, last_a = last_a, first_a
    if first_b > last_b:
        first_b, last_b = last_b, first_b

    if last_a < first_b:
        return {
            "direction": "before",
            "gap_frames": int(first_b - last_a),
            "overlap_frames": 0,
        }
    if first_a > last_b:
        return {
            "direction": "after",
            "gap_frames": int(first_a - last_b),
            "overlap_frames": 0,
        }

    overlap_start = max(first_a, first_b)
    overlap_end = min(last_a, last_b)
    return {
        "direction": "overlap",
        "gap_frames": 0,
        "overlap_frames": int(max(0, overlap_end - overlap_start + 1)),
    }


def find_best_appearance_neighbor(
    *,
    index_data: dict[str, Any],
    source_track_id: int,
    candidate_track_ids: list[int],
    topk_similarity: int = 5,
) -> tuple[int | None, float]:
    """Return the highest-scoring appearance neighbor for one source track."""

    rows_by_track = build_rows_by_track(index_data)
    embeddings = index_data.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("index_data['embeddings'] must be a torch.Tensor.")

    ranked = rank_appearance_neighbors(
        embeddings=embeddings,
        rows_by_track=rows_by_track,
        source_track_id=int(source_track_id),
        candidate_track_ids=candidate_track_ids,
        topk_similarity=topk_similarity,
    )
    if not ranked:
        return None, 0.0
    return ranked[0]


def suggest_related_tracks(
    *,
    index_data: dict[str, Any],
    target_track_id: int,
    query_ranking: list[dict[str, Any]] | None = None,
    max_gap_frames: int = 300,
    min_appearance_score: float = 0.78,
    min_candidate_margin: float = 0.05,
    max_overlap_frames: int = 12,
    topk_similarity: int = 5,
    max_related_tracks: int = 3,
) -> list[dict[str, Any]]:
    """Return only conservative accepted stitch candidates for a target track."""

    ranked = rank_stitch_candidates(
        index_data=index_data,
        target_track_id=target_track_id,
        query_ranking=query_ranking,
        max_gap_frames=max_gap_frames,
        min_appearance_score=min_appearance_score,
        min_candidate_margin=min_candidate_margin,
        max_overlap_frames=max_overlap_frames,
        topk_similarity=topk_similarity,
    )
    accepted = [item for item in ranked if item["decision"] == "accepted"]
    accepted = sorted(
        accepted,
        key=lambda item: (
            track_sort_frame(index_data, int(item["track_id"])),
            -float(item["appearance_score"]),
            int(item["track_id"]),
        ),
    )
    return accepted[: max(0, int(max_related_tracks))]


def rank_stitch_candidates(
    *,
    index_data: dict[str, Any],
    target_track_id: int,
    query_ranking: list[dict[str, Any]] | None = None,
    max_gap_frames: int = 300,
    min_appearance_score: float = 0.78,
    min_candidate_margin: float = 0.05,
    max_overlap_frames: int = 12,
    topk_similarity: int = 5,
) -> list[dict[str, Any]]:
    """Rank accepted and rejected stitch candidates with conservative reasons."""

    embeddings = index_data.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("index_data['embeddings'] must be a torch.Tensor.")

    rows_by_track = build_rows_by_track(index_data)
    target_track_id = int(target_track_id)
    target_rows = rows_by_track.get(target_track_id, [])
    if not target_rows:
        return []

    track_ids = sorted(rows_by_track)
    query_scores = build_query_score_lookup(query_ranking)
    best_query_score = best_query_ranking_score(query_ranking)
    candidate_infos: dict[int, dict[str, Any]] = {}

    for candidate_track_id in track_ids:
        if candidate_track_id == target_track_id:
            continue
        temporal = compute_temporal_gap(
            range_a=get_track_time_range(index_data, candidate_track_id),
            range_b=get_track_time_range(index_data, target_track_id),
        )
        appearance_score = compute_track_appearance_score(
            embeddings=embeddings,
            rows_a=target_rows,
            rows_b=rows_by_track.get(candidate_track_id, []),
            topk=topk_similarity,
        )
        candidate_infos[candidate_track_id] = {
            "track_id": candidate_track_id,
            "appearance_score": appearance_score,
            "gap_frames": temporal["gap_frames"],
            "direction": temporal["direction"],
            "overlap_frames": temporal["overlap_frames"],
            "query_score": query_scores.get(candidate_track_id),
            "decision": "rejected",
            "reason": "",
            "_temporal_ok": is_temporally_compatible(
                temporal=temporal,
                max_gap_frames=max_gap_frames,
                max_overlap_frames=max_overlap_frames,
            ),
        }

    target_neighbor_stats = best_target_neighbors_by_direction(
        candidate_infos=candidate_infos,
    )

    for candidate_track_id, info in candidate_infos.items():
        reason = first_rejection_reason(
            info=info,
            best_query_score=best_query_score,
            max_gap_frames=max_gap_frames,
            min_appearance_score=min_appearance_score,
            max_overlap_frames=max_overlap_frames,
        )
        if reason is not None:
            info["reason"] = reason
            continue

        direction = str(info["direction"])
        target_stats = target_neighbor_stats.get(direction, {})
        if target_stats.get("track_id") != candidate_track_id:
            info["reason"] = "not target best appearance neighbor"
            continue
        if float(target_stats.get("margin", 0.0)) < min_candidate_margin:
            info["reason"] = "target margin below threshold"
            continue

        reverse_candidate_ids = compatible_track_ids_for_source(
            index_data=index_data,
            source_track_id=candidate_track_id,
            track_ids=track_ids,
            max_gap_frames=max_gap_frames,
            max_overlap_frames=max_overlap_frames,
        )
        reverse_ranked = rank_appearance_neighbors(
            embeddings=embeddings,
            rows_by_track=rows_by_track,
            source_track_id=candidate_track_id,
            candidate_track_ids=reverse_candidate_ids,
            topk_similarity=topk_similarity,
        )
        reverse_best_track_id = reverse_ranked[0][0] if reverse_ranked else None
        reverse_margin = appearance_margin(reverse_ranked)
        if reverse_best_track_id != target_track_id:
            info["reason"] = "not mutual best"
            continue
        if reverse_margin < min_candidate_margin:
            info["reason"] = "reverse margin below threshold"
            continue

        info["decision"] = "accepted"
        info["reason"] = "appearance, temporal gap, and mutual best passed"

    results = [public_candidate_item(info) for info in candidate_infos.values()]
    return sorted(
        results,
        key=lambda item: (
            item["decision"] != "accepted",
            -float(item["appearance_score"]),
            int(item["track_id"]),
        ),
    )


def merge_track_timelines(
    *,
    index_data: dict[str, Any],
    track_ids: list[int],
) -> dict[str, Any]:
    """Merge track timelines into one renderer-friendly stitched timeline."""

    if not track_ids:
        raise ValueError("track_ids must contain at least one track ID.")

    source_track_ids = unique_ints(track_ids)
    frame_entries: dict[int, dict[str, Any]] = {}
    for track_id in source_track_ids:
        timeline = get_timeline_or_none(index_data, track_id)
        if timeline is None:
            raise ValueError(f"Cannot find timeline for track_id={track_id}.")

        frame_ids = timeline.get("frame_ids") or []
        bboxes = timeline.get("bboxes") or []
        timestamps = timeline.get("timestamps") or []
        confidence_scores = timeline.get("confidence_scores") or []

        for item_index, raw_frame_id in enumerate(frame_ids):
            if raw_frame_id is None or item_index >= len(bboxes):
                continue
            frame_id = int(raw_frame_id)
            entry = {
                "frame_id": frame_id,
                "bbox": [int(value) for value in bboxes[item_index]],
                "timestamp": get_list_value(timestamps, item_index),
                "confidence_score": coerce_float_or_none(
                    get_list_value(confidence_scores, item_index)
                ),
            }
            existing = frame_entries.get(frame_id)
            if existing is None or should_replace_duplicate_frame(existing, entry):
                frame_entries[frame_id] = entry

    if not frame_entries:
        raise ValueError("No timeline frames were found for the requested track IDs.")

    ordered_entries = [frame_entries[frame_id] for frame_id in sorted(frame_entries)]
    return {
        "frame_ids": [entry["frame_id"] for entry in ordered_entries],
        "bboxes": [entry["bbox"] for entry in ordered_entries],
        "timestamps": [entry["timestamp"] for entry in ordered_entries],
        "confidence_scores": [
            entry["confidence_score"] for entry in ordered_entries
        ],
        "status": "stitched",
        "source_track_ids": source_track_ids,
    }


def get_timeline_or_none(
    index_data: dict[str, Any],
    track_id: int,
) -> dict[str, Any] | None:
    track_timeline = index_data.get("track_timeline")
    if not isinstance(track_timeline, dict):
        return None

    timeline = track_timeline.get(track_id)
    if timeline is None:
        timeline = track_timeline.get(str(track_id))
    if timeline is None:
        for raw_key, value in track_timeline.items():
            if coerce_int_or_none(raw_key) == int(track_id):
                timeline = value
                break
    return timeline if isinstance(timeline, dict) else None


def coerce_frame_ids(raw_frame_ids: Any) -> list[int]:
    frames: list[int] = []
    for raw_frame_id in raw_frame_ids:
        if raw_frame_id is None:
            continue
        frames.append(int(raw_frame_id))
    return frames


def valid_embedding_rows(rows: list[int], num_embeddings: int) -> list[int]:
    valid_rows: list[int] = []
    for row in rows:
        row_index = int(row)
        if 0 <= row_index < num_embeddings:
            valid_rows.append(row_index)
    return valid_rows


def rank_appearance_neighbors(
    *,
    embeddings: torch.Tensor,
    rows_by_track: dict[int, list[int]],
    source_track_id: int,
    candidate_track_ids: list[int],
    topk_similarity: int,
) -> list[tuple[int, float]]:
    source_rows = rows_by_track.get(int(source_track_id), [])
    ranked: list[tuple[int, float]] = []
    for raw_candidate_track_id in candidate_track_ids:
        candidate_track_id = int(raw_candidate_track_id)
        if candidate_track_id == int(source_track_id):
            continue
        candidate_rows = rows_by_track.get(candidate_track_id, [])
        score = compute_track_appearance_score(
            embeddings=embeddings,
            rows_a=source_rows,
            rows_b=candidate_rows,
            topk=topk_similarity,
        )
        ranked.append((candidate_track_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def compatible_track_ids_for_source(
    *,
    index_data: dict[str, Any],
    source_track_id: int,
    track_ids: list[int],
    max_gap_frames: int,
    max_overlap_frames: int,
) -> list[int]:
    source_range = get_track_time_range(index_data, source_track_id)
    compatible: list[int] = []
    for track_id in track_ids:
        if int(track_id) == int(source_track_id):
            continue
        temporal = compute_temporal_gap(
            range_a=source_range,
            range_b=get_track_time_range(index_data, int(track_id)),
        )
        if is_temporally_compatible(
            temporal=temporal,
            max_gap_frames=max_gap_frames,
            max_overlap_frames=max_overlap_frames,
        ):
            compatible.append(int(track_id))
    return compatible


def best_target_neighbors_by_direction(
    *,
    candidate_infos: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_direction: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for candidate_track_id, info in candidate_infos.items():
        if not info.get("_temporal_ok"):
            continue
        direction = str(info["direction"])
        by_direction[direction].append(
            (int(candidate_track_id), float(info["appearance_score"]))
        )

    best_by_direction: dict[str, dict[str, Any]] = {}
    for direction, candidates in by_direction.items():
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        best_track_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -inf
        best_by_direction[direction] = {
            "track_id": best_track_id,
            "score": best_score,
            "margin": best_score - second_score,
        }
    return best_by_direction


def appearance_margin(ranked: list[tuple[int, float]]) -> float:
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return inf
    return float(ranked[0][1] - ranked[1][1])


def first_rejection_reason(
    *,
    info: dict[str, Any],
    best_query_score: float | None,
    max_gap_frames: int,
    min_appearance_score: float,
    max_overlap_frames: int,
) -> str | None:
    direction = info["direction"]
    gap_frames = info["gap_frames"]
    overlap_frames = int(info["overlap_frames"])
    if direction == "unknown":
        return "missing temporal info"
    if gap_frames is not None and int(gap_frames) > max_gap_frames:
        return "gap too large"
    if overlap_frames > max_overlap_frames:
        return "overlap too high"
    if float(info["appearance_score"]) < min_appearance_score:
        return "appearance below threshold"
    query_score = info.get("query_score")
    if (
        best_query_score is not None
        and query_score is not None
        and float(query_score) < best_query_score - QUERY_SCORE_TOLERANCE
    ):
        return "query score too low"
    return None


def is_temporally_compatible(
    *,
    temporal: dict[str, Any],
    max_gap_frames: int,
    max_overlap_frames: int,
) -> bool:
    if temporal["direction"] == "unknown":
        return False
    gap_frames = temporal["gap_frames"]
    if gap_frames is not None and int(gap_frames) > max_gap_frames:
        return False
    return int(temporal["overlap_frames"]) <= max_overlap_frames


def build_query_score_lookup(
    query_ranking: list[dict[str, Any]] | None,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for item in query_ranking or []:
        if not isinstance(item, dict):
            continue
        track_id = coerce_int_or_none(item.get("track_id"))
        score = coerce_float_or_none(item.get("score"))
        if track_id is None or score is None:
            continue
        scores[track_id] = score
    return scores


def best_query_ranking_score(
    query_ranking: list[dict[str, Any]] | None,
) -> float | None:
    scores = build_query_score_lookup(query_ranking)
    if not scores:
        return None
    return max(scores.values())


def public_candidate_item(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": int(info["track_id"]),
        "appearance_score": float(info["appearance_score"]),
        "gap_frames": info["gap_frames"],
        "direction": str(info["direction"]),
        "overlap_frames": int(info["overlap_frames"]),
        "query_score": info["query_score"],
        "decision": str(info["decision"]),
        "reason": str(info["reason"]),
    }


def should_replace_duplicate_frame(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    existing_conf = existing.get("confidence_score")
    candidate_conf = candidate.get("confidence_score")
    if existing_conf is None or candidate_conf is None:
        return False
    return float(candidate_conf) > float(existing_conf)


def track_sort_frame(index_data: dict[str, Any], track_id: int) -> float:
    first_frame, _ = get_track_time_range(index_data, int(track_id))
    return float(first_frame) if first_frame is not None else inf


def unique_ints(values: list[int]) -> list[int]:
    unique_values: list[int] = []
    seen: set[int] = set()
    for value in values:
        int_value = int(value)
        if int_value in seen:
            continue
        seen.add(int_value)
        unique_values.append(int_value)
    return unique_values


def get_list_value(values: Any, index: int) -> Any | None:
    return values[index] if isinstance(values, list) and index < len(values) else None


def coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
