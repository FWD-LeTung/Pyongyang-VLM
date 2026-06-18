"""Shared training utilities for Matching Engine Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.matching_engine.core.clip_model import base_clip_model
from src.utils.logger import setup_logger


logger = setup_logger(__name__)
EXTRA_STATE_FILE = "matching_extra.pt"


class IdentityClassificationHead(nn.Module):
    """Shared identity classifier for image and text embeddings."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        """Initialize a linear identity classifier."""

        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return identity logits for embedding features."""

        return self.classifier(features)


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
    accelerator: Any,
    epoch: int,
    loss_config: dict[str, Any] | None = None,
    id_head: IdentityClassificationHead | None = None,
) -> float:
    """Train one epoch and return mean configured loss."""

    model.train()
    losses: list[float] = []
    progress = tqdm(
        train_loader,
        desc=f"epoch {epoch}",
        disable=not accelerator.is_main_process,
        leave=False,
    )
    for batch in progress:
        inputs = to_device_batch(batch, accelerator.device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_training_loss(model, inputs, loss_config, id_head, accelerator)
        accelerator.backward(loss)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    mean_loss = sum(losses) / max(len(losses), 1)
    if accelerator.is_main_process:
        logger.info("Epoch %s train_loss=%.6f", epoch, mean_loss)
    return mean_loss


def clip_infonce_loss(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    accelerator: Any = None,
) -> torch.Tensor:
    """Compute symmetric CLIP InfoNCE loss."""

    outputs = model(**model_inputs(inputs))
    return clip_infonce_from_features(
        outputs.image_embeds,
        outputs.text_embeds,
        clip_logit_scale(model, accelerator),
    )


def multi_positive_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute label-aware symmetric multi-positive contrastive loss."""

    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits_i2t = logit_scale * image_features @ text_features.t()
    logits_t2i = logits_i2t.t()
    positive_mask = labels.view(-1, 1).eq(labels.view(1, -1)).float()
    image_mask = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    text_mask = positive_mask.t() / positive_mask.t().sum(dim=1, keepdim=True).clamp_min(1.0)
    loss_i2t = -(image_mask * F.log_softmax(logits_i2t, dim=1)).sum(dim=1).mean()
    loss_t2i = -(text_mask * F.log_softmax(logits_t2i, dim=1)).sum(dim=1).mean()
    return (loss_i2t + loss_t2i) / 2.0


def multi_positive_clip_loss(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    accelerator: Any = None,
) -> torch.Tensor:
    """Compute multi-positive CLIP loss from a model batch with labels."""

    outputs = model(**model_inputs(inputs))
    return multi_positive_contrastive_loss(
        outputs.image_embeds,
        outputs.text_embeds,
        inputs["label"],
        clip_logit_scale(model, accelerator),
    )


def compute_training_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_config: dict[str, Any] | None,
    id_head: IdentityClassificationHead | None,
    accelerator: Any = None,
) -> torch.Tensor:
    """Compute configured contrastive loss plus optional identity loss."""

    config = loss_config or {"name": "clip_infonce", "id": {"enabled": False}}
    outputs = model(**model_inputs(batch))
    logit_scale = clip_logit_scale(model, accelerator)
    if config.get("name", "clip_infonce") == "multi_positive":
        contrastive_loss = multi_positive_contrastive_loss(
            outputs.image_embeds,
            outputs.text_embeds,
            batch["label"],
            logit_scale,
        )
    elif config.get("name") == "clip_infonce":
        contrastive_loss = clip_infonce_from_features(
            outputs.image_embeds,
            outputs.text_embeds,
            logit_scale,
        )
    else:
        raise ValueError(f"Unsupported loss name: {config.get('name')}")

    id_config = config.get("id", {})
    if not id_config.get("enabled", False):
        return contrastive_loss
    if id_head is None:
        raise ValueError("ID loss is enabled but id_head is not initialized.")
    image_id_loss = F.cross_entropy(id_head(outputs.image_embeds), batch["label"])
    text_id_loss = F.cross_entropy(id_head(outputs.text_embeds), batch["label"])
    id_loss = (image_id_loss + text_id_loss) / 2.0
    return contrastive_loss + float(id_config.get("weight", 0.5)) * id_loss


def clip_infonce_from_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute diagonal-positive CLIP InfoNCE from embedding tensors."""

    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits = logit_scale * image_features @ text_features.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    ) / 2.0


def clip_logit_scale(model: torch.nn.Module, accelerator: Any = None) -> torch.Tensor:
    """Return CLIP logit scale from wrapped or unwrapped models."""

    unwrapped_model = accelerator.unwrap_model(model) if accelerator else model
    return base_clip_model(unwrapped_model).logit_scale.exp()


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


def to_device_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move all tensor batch fields to device."""

    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop non-model fields from a model batch."""

    return {
        key: value
        for key, value in batch.items()
        if key != "label" and isinstance(value, torch.Tensor)
    }


def save_matching_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    id_head: IdentityClassificationHead | None = None,
) -> None:
    """Save PEFT adapter and Matching Engine extra trainable state."""

    checkpoint_dir = Path(path)
    model.save_pretrained(checkpoint_dir)
    core_model = base_clip_model(model)
    state: dict[str, Any] = {}
    for module_name in ("text_projection", "visual_projection"):
        module = getattr(core_model, module_name, None)
        if isinstance(module, nn.Module):
            state[module_name] = module.state_dict()
    if hasattr(core_model, "logit_scale"):
        state["logit_scale"] = core_model.logit_scale.detach().cpu()
    if id_head is not None:
        state["id_head"] = id_head.state_dict()
    torch.save(state, checkpoint_dir / EXTRA_STATE_FILE)


def load_matching_extra_state(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    id_head: IdentityClassificationHead | None = None,
) -> None:
    """Load Matching Engine extra state if present next to an adapter."""

    state_path = Path(path) / EXTRA_STATE_FILE
    if not state_path.exists():
        return
    state = torch.load(state_path, map_location=device)
    core_model = base_clip_model(model)
    for module_name in ("text_projection", "visual_projection"):
        module = getattr(core_model, module_name, None)
        if isinstance(module, nn.Module) and module_name in state:
            module.load_state_dict(state[module_name], strict=False)
    if "logit_scale" in state and hasattr(core_model, "logit_scale"):
        core_model.logit_scale.data.copy_(state["logit_scale"].to(device))
    if id_head is not None and "id_head" in state:
        id_head.load_state_dict(state["id_head"], strict=False)


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
