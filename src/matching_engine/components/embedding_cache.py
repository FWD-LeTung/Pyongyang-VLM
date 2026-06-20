"""In-memory image embedding cache for temporal tracklet chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """Stable key for one temporal chunk in a video/session scope."""

    scope: str
    track_id: int
    chunk_id: int


@dataclass(frozen=True)
class CachedChunkEmbeddings:
    """Cached embeddings plus sampled metadata for one chunk."""

    scope: str
    track_id: int
    chunk_id: int
    embeddings: torch.Tensor
    sampled_indices: list[int] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    bboxes: list[list[int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None
    status: str = ""


class EmbeddingCache:
    """Simple memory cache keyed by ``video/session + track_id + chunk_id``."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        dtype: str = "fp16",
        device: str = "cuda",
        storage: str = "memory",
        allow_cpu_fallback: bool = False,
    ) -> None:
        if storage != "memory":
            raise NotImplementedError("Only in-memory embedding cache is implemented.")
        self.enabled = enabled
        self.dtype = dtype
        self.device = self._resolve_device(
            device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.storage = storage
        self._items: dict[CacheKey, CachedChunkEmbeddings] = {}

    def get(
        self,
        *,
        video_id: str | None,
        session_id: str | None,
        track_id: int,
        chunk_id: int,
    ) -> CachedChunkEmbeddings | None:
        """Return cached embeddings for one chunk, if present."""

        if not self.enabled:
            return None
        return self._items.get(
            CacheKey(
                scope=cache_scope(video_id=video_id, session_id=session_id),
                track_id=int(track_id),
                chunk_id=int(chunk_id),
            )
        )

    def set(self, item: CachedChunkEmbeddings) -> CachedChunkEmbeddings:
        """Store embeddings according to cache dtype/device settings."""

        prepared = CachedChunkEmbeddings(
            scope=item.scope,
            track_id=item.track_id,
            chunk_id=item.chunk_id,
            embeddings=self._prepare_embeddings(item.embeddings),
            sampled_indices=list(item.sampled_indices),
            frame_ids=list(item.frame_ids),
            bboxes=[list(map(int, bbox)) for bbox in item.bboxes],
            timestamps=[float(ts) for ts in item.timestamps],
            confidence_scores=[float(score) for score in item.confidence_scores],
            first_seen=item.first_seen,
            last_seen=item.last_seen,
            status=item.status,
        )
        if self.enabled:
            self._items[
                CacheKey(
                    scope=prepared.scope,
                    track_id=prepared.track_id,
                    chunk_id=prepared.chunk_id,
                )
            ] = prepared
        return prepared

    def clear(self) -> None:
        """Remove all cached chunks."""

        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def _prepare_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Detach and place embeddings on the configured cache device/dtype."""

        target_dtype = torch.float16 if self.dtype == "fp16" else torch.float32
        if self.device.type != "cuda":
            target_dtype = torch.float32
        return embeddings.detach().to(device=self.device, dtype=target_dtype).contiguous()

    @staticmethod
    def _resolve_device(device: str, *, allow_cpu_fallback: bool = False) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            message = (
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use --device cpu or pass --allow-cpu-fallback."
            )
            if not allow_cpu_fallback:
                raise RuntimeError(message)
            logger.warning("%s Falling back to CPU cache.", message)
            return torch.device("cpu")
        return torch.device(device)


def cache_scope(*, video_id: str | None, session_id: str | None) -> str:
    """Build the namespace used for image embedding reuse."""

    if video_id:
        return f"video:{video_id}"
    if session_id:
        return f"session:{session_id}"
    return "default"
