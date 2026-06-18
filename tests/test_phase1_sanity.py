"""Fail-fast sanity checks for Matching Engine Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPConfig, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import eval_cuhk, train_lora
from src.matching_engine.core.clip_model import base_clip_model, build_clip_lora
from src.matching_engine.core.dataset import CLIP_CUHK_Dataset
from src.matching_engine.core.metrics import Evaluator, rank
from src.matching_engine.core.training import (
    clip_infonce_loss,
    multi_positive_contrastive_loss,
    multi_positive_clip_loss,
    to_device_inputs,
)


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
    multi_loss = multi_positive_clip_loss(model, batch)

    assert outputs.image_embeds.shape == (2, 16)
    assert outputs.text_embeds.shape == (2, 16)
    assert torch.isfinite(loss)
    assert torch.isfinite(multi_loss)
    assert loss.ndim == 0
    assert multi_loss.ndim == 0


def test_dataset_modes_and_missing_image_fail_fast(tmp_path: Path) -> None:
    """Check dataset adapter modes and missing image fail-fast behavior."""

    processor = DummyProcessor()
    img_path = tmp_path / "person.jpg"
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(img_path)
    samples = [
        (0, 0, img_path, "first caption"),
        (0, 0, img_path, "second caption"),
    ]

    pair_item = CLIP_CUHK_Dataset(samples, processor, mode="pair")[0]
    image_dataset = CLIP_CUHK_Dataset(samples, processor, mode="image")
    text_dataset = CLIP_CUHK_Dataset(samples, processor, mode="text")

    assert pair_item["pixel_values"].shape == (3, 32, 32)
    assert pair_item["input_ids"].shape == (77,)
    assert len(image_dataset) == 1
    assert len(text_dataset) == 2
    with pytest.raises(RuntimeError, match="Failed to load image"):
        CLIP_CUHK_Dataset(
            [(0, 0, tmp_path / "missing.jpg", "bad image")],
            processor,
            mode="pair",
        )[0]


def test_multi_positive_loss_backward() -> None:
    """Ensure duplicate labels form multi-positive targets and backprop works."""

    image_features = torch.randn(4, 8, requires_grad=True)
    text_features = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([1, 1, 2, 3], dtype=torch.long)
    positive_mask = labels.view(-1, 1).eq(labels.view(1, -1))

    loss = multi_positive_contrastive_loss(
        image_features,
        text_features,
        labels,
        torch.tensor(2.0),
    )
    loss.backward()

    assert positive_mask[0, 1]
    assert torch.isfinite(loss)
    assert image_features.grad is not None
    assert text_features.grad is not None


def test_build_clip_lora_trainable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build LoRA without downloads and verify trainable modules."""

    monkeypatch.setattr(
        "src.matching_engine.core.clip_model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: CLIPModel(tiny_clip_config()),
    )
    monkeypatch.setattr(
        "src.matching_engine.core.clip_model.CLIPProcessor.from_pretrained",
        lambda *_args, **_kwargs: DummyProcessor(),
    )
    model, _processor = build_clip_lora(
        {
            "model_name": "tiny",
            "lora": {
                "r": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            },
            "trainable": {
                "unfreeze_projection": True,
                "unfreeze_logit_scale": True,
            },
            "quantization": {"enabled": False},
        }
    )
    core_model = base_clip_model(model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable
    assert any("lora" in name for name in trainable)
    assert core_model.logit_scale.requires_grad
    assert any(parameter.requires_grad for parameter in core_model.text_projection.parameters())


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


def test_rank_toy_matrix() -> None:
    """Validate rank metrics on a perfect two-sample toy matrix."""

    cmc, mAP, mINP, indices = rank(
        torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        max_rank=2,
    )

    assert cmc[0].item() == 100.0
    assert mAP.item() == 100.0
    assert mINP.item() == 100.0
    assert indices.shape == (2, 2)


def test_cli_parse_smoke_and_zero_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check train/eval CLI smoke and zero-shot flags parse cleanly."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_lora.py", "--smoke-test", "--max-train-samples", "8"],
    )
    train_args = train_lora.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_cuhk.py", "--zero-shot", "--smoke-test", "--i2t"],
    )
    eval_args = eval_cuhk.parse_args()

    assert train_args.smoke_test
    assert train_args.max_train_samples == 8
    assert eval_args.zero_shot
    assert eval_args.smoke_test
    assert eval_args.i2t


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
    test_multi_positive_loss_backward()
    test_rank_toy_matrix()
    print("Phase 1 sanity test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
