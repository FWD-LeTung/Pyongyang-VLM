"""Protocol for image-text retrieval backends used by Module 3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class RetrievalEncoder(Protocol):
    """Minimal image-text encoder interface for production matching."""

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings with shape ``[N, D]``."""

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        """Return L2-normalized image embeddings with shape ``[N, D]``."""
