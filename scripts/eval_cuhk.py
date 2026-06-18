"""Evaluate the best CLIP LoRA adapter on CUHK-PEDES test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor


def _bootstrap_project_root() -> Path:
    """Add the repository root to sys.path for direct script execution."""

    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml and src/.")


PROJECT_ROOT = _bootstrap_project_root()

from src.matching_engine.core.dataset import CLIP_CUHK_Dataset, CUHKPEDES  # noqa: E402
from src.matching_engine.core.metrics import Evaluator  # noqa: E402
from src.matching_engine.core.training import (  # noqa: E402
    load_config,
    load_matching_extra_state,
    pick_device,
    resolve_path,
)
from src.utils.logger import setup_logger  # noqa: E402


logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate CLIP LoRA on CUHK-PEDES.")
    parser.add_argument("--config", default="config/matching_engine.yaml")
    parser.add_argument("--adapter-dir", default="weights/best_adapter")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--i2t", action="store_true", help="Also report image-to-text.")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-val-texts", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run CUHK-PEDES retrieval evaluation."""

    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    adapter_dir = resolve_path(args.adapter_dir, PROJECT_ROOT)
    dataset_root = resolve_path(args.dataset_root or config["dataset_root"], PROJECT_ROOT)
    batch_size = int(args.batch_size or config["batch_size"])
    max_val_images = args.max_val_images
    max_val_texts = args.max_val_texts
    if args.smoke_test:
        max_val_images = max_val_images or 64
        max_val_texts = max_val_texts or 128
    if not args.zero_shot and not adapter_dir.exists():
        raise RuntimeError(f"Best adapter not found: {adapter_dir}")

    device = pick_device()
    logger.info(
        "Using device=%s dataset_root=%s mode=%s",
        device,
        dataset_root,
        "zero-shot" if args.zero_shot else f"adapter:{adapter_dir}",
    )
    base_model = CLIPModel.from_pretrained(config["model_name"])
    if args.zero_shot:
        model = base_model.to(device)
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
        load_matching_extra_state(model, adapter_dir, device)
    processor = CLIPProcessor.from_pretrained(config["model_name"])

    dataset = CUHKPEDES(dataset_root)
    img_loader = DataLoader(
        CLIP_CUHK_Dataset(
            dataset.test,
            processor,
            mode="image",
            max_samples=max_val_images,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    txt_loader = DataLoader(
        CLIP_CUHK_Dataset(
            dataset.test,
            processor,
            mode="text",
            max_samples=max_val_texts,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    rank1 = Evaluator(img_loader, txt_loader).eval(model, i2t_metric=args.i2t)
    logger.info("Final CUHK-PEDES test Rank-1: %.3f", rank1)
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
