"""Shared training utilities for Matching Engine Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.matching_engine.core.clip_model import base_clip_model
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a file path or current directory."""

    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            return path
    return Path.cwd().resolve()


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve absolute or project-relative paths consistently."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = project_root(root) if root is not None else project_root()
    return (base / candidate).resolve()


def load_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = resolve_path(path, root)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping.")
    return config


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch and return mean InfoNCE loss."""

    model.train()
    losses: list[float] = []
    progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
    for batch in progress:
        inputs = to_device_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = clip_infonce_loss(model, inputs)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    mean_loss = sum(losses) / max(len(losses), 1)
    logger.info("Epoch %s train_loss=%.6f", epoch, mean_loss)
    return mean_loss


def clip_infonce_loss(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute symmetric CLIP InfoNCE loss."""

    outputs = model(**inputs)
    image_features = F.normalize(outputs.image_embeds, p=2, dim=1)
    text_features = F.normalize(outputs.text_embeds, p=2, dim=1)
    logit_scale = base_clip_model(model).logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    ) / 2.0


def to_device_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move model inputs to device and remove non-model fields."""

    return {
        key: value.to(device)
        for key, value in batch.items()
        if key != "label" and isinstance(value, torch.Tensor)
    }


def pick_device() -> torch.device:
    """Return the best local torch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_metrics(path: str | Path, record: dict[str, Any]) -> None:
    """Append one checkpoint metric record as JSONL."""

    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
