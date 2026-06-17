"""Fail-fast sanity checks for Matching Engine Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPConfig, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_engine.core.dataset import CLIP_CUHK_Dataset
from src.matching_engine.core.metrics import Evaluator
from src.matching_engine.core.training import clip_infonce_loss, to_device_inputs


class DummyProcessor:
    """Minimal CLIP-like processor for in-memory sanity tests."""

    def __call__(
        self,
        text: str | None = None,
        images: Image.Image | None = None,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Return CLIP-compatible tensors."""

        result: dict[str, torch.Tensor] = {}
        if images is not None:
            array = np.asarray(images.resize((32, 32)), dtype=np.float32) / 255.0
            result["pixel_values"] = torch.tensor(array).permute(2, 0, 1).unsqueeze(0)
        if text is not None:
            token_ids = [ord(char) % 99 + 1 for char in text[:77]]
            token_ids = token_ids + [0] * (77 - len(token_ids))
            result["input_ids"] = torch.tensor([token_ids], dtype=torch.long)
            result["attention_mask"] = (result["input_ids"] != 0).long()
        return result


def test_phase1_forward_and_infonce_loss_shape() -> None:
    """Run Dataset -> DataLoader -> CLIP forward -> InfoNCE loss in RAM."""

    processor = DummyProcessor()
    images = [
        Image.fromarray(np.full((48, 32, 3), 64, dtype=np.uint8)),
        Image.fromarray(np.full((32, 48, 3), 192, dtype=np.uint8)),
    ]
    samples = [
        (0, 0, images[0], "a person wearing a red shirt"),
        (1, 1, images[1], "a person carrying a black bag"),
    ]
    loader = DataLoader(
        CLIP_CUHK_Dataset(samples, processor, mode="pair"),
        batch_size=2,
    )
    batch = next(iter(loader))
    model = CLIPModel(tiny_clip_config())
    inputs = to_device_inputs(batch, torch.device("cpu"))
    outputs = model(**inputs)
    loss = clip_infonce_loss(model, inputs)

    assert outputs.image_embeds.shape == (2, 16)
    assert outputs.text_embeds.shape == (2, 16)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_evaluator_projects_peft_clip_embeddings() -> None:
    """Ensure PEFT CLIP eval features are projected into shared CLIP space."""

    processor = DummyProcessor()
    samples = [
        (
            0,
            0,
            Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8)),
            "a person wearing a red shirt",
        ),
        (
            1,
            1,
            Image.fromarray(np.full((32, 32, 3), 192, dtype=np.uint8)),
            "a person carrying a black bag",
        ),
    ]
    img_loader = DataLoader(
        CLIP_CUHK_Dataset(samples, processor, mode="image"),
        batch_size=2,
    )
    txt_loader = DataLoader(
        CLIP_CUHK_Dataset(samples, processor, mode="text"),
        batch_size=2,
    )
    model = get_peft_model(
        CLIPModel(tiny_clip_config()),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )

    qfeats, gfeats, qids, gids = Evaluator(img_loader, txt_loader)._compute_embedding(
        model
    )

    assert qfeats.shape == (2, 16)
    assert gfeats.shape == (2, 16)
    assert qids.tolist() == [0, 1]
    assert gids.tolist() == [0, 1]
    assert torch.isfinite(qfeats).all()
    assert torch.isfinite(gfeats).all()


def tiny_clip_config() -> CLIPConfig:
    """Create a tiny random CLIP config for fast CPU tests."""

    return CLIPConfig(
        projection_dim=16,
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 77,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "num_channels": 3,
        },
    )


def main() -> int:
    """Run the sanity test when this file is executed directly."""

    test_phase1_forward_and_infonce_loss_shape()
    test_evaluator_projects_peft_clip_embeddings()
    print("Phase 1 sanity test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
