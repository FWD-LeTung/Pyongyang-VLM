"""Train CLIP LoRA adapters on CUHK-PEDES."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _bootstrap_project_root() -> Path:
    """Add the repository root to sys.path for direct script execution."""

    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise RuntimeError("Cannot find project root containing pyproject.toml and src/.")


PROJECT_ROOT = _bootstrap_project_root()

from src.matching_engine.core.clip_model import build_clip_lora  # noqa: E402
from src.matching_engine.core.dataset import CLIP_CUHK_Dataset, CUHKPEDES  # noqa: E402
from src.matching_engine.core.metrics import Evaluator  # noqa: E402
from src.matching_engine.core.training import (  # noqa: E402
    append_metrics,
    load_config,
    pick_device,
    resolve_path,
    train_one_epoch,
)
from src.utils.logger import setup_logger  # noqa: E402


logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse train CLI arguments."""

    parser = argparse.ArgumentParser(description="Train CLIP LoRA on CUHK-PEDES.")
    parser.add_argument("--config", default="config/matching_engine.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run LoRA training with checkpointing and early stopping."""

    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    dataset_root = resolve_path(args.dataset_root or config["dataset_root"], PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir or config["output_weights_dir"], PROJECT_ROOT)
    batch_size = int(args.batch_size or config["batch_size"])
    epochs = int(args.epochs or config["epochs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    logger.info("Using device=%s dataset_root=%s output_dir=%s", device, dataset_root, output_dir)
    model, processor = build_clip_lora(config)
    model.to(device)

    dataset = CUHKPEDES(dataset_root)
    train_loader = DataLoader(
        CLIP_CUHK_Dataset(dataset.train, processor, mode="pair"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_img_loader = DataLoader(
        CLIP_CUHK_Dataset(dataset.val, processor, mode="image"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_txt_loader = DataLoader(
        CLIP_CUHK_Dataset(dataset.val, processor, mode="text"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    evaluator = Evaluator(val_img_loader, val_txt_loader)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
    )

    best_rank1 = 0.0
    no_improve_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
        model.save_pretrained(checkpoint_dir)

        val_rank1 = evaluator.eval(model)
        append_metrics(
            output_dir / "training_metrics.jsonl",
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rank1": val_rank1,
                "val_metrics": evaluator.last_metrics,
                "best_rank1": max(best_rank1, val_rank1),
                "checkpoint": str(checkpoint_dir),
            },
        )

        if val_rank1 > best_rank1:
            best_rank1 = val_rank1
            no_improve_epochs = 0
            model.save_pretrained(output_dir / "best_adapter")
            logger.info("New best Rank-1 %.3f at epoch %s.", best_rank1, epoch)
        else:
            no_improve_epochs += 1
            logger.info("No improvement for %s epoch(s).", no_improve_epochs)
            if no_improve_epochs >= int(config["patience"]):
                logger.info("Early stopping triggered at epoch %s.", epoch)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
