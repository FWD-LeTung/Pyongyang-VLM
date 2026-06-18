"""Train CLIP LoRA adapters on CUHK-PEDES."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
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

from src.matching_engine.core.clip_model import base_clip_model, build_clip_lora  # noqa: E402
from src.matching_engine.core.dataset import (  # noqa: E402
    CLIP_CUHK_Dataset,
    CUHKPEDES,
    build_train_augmentation,
    clip_image_size,
)
from src.matching_engine.core.metrics import Evaluator  # noqa: E402
from src.matching_engine.core.training import (  # noqa: E402
    IdentityClassificationHead,
    append_metrics,
    load_config,
    resolve_path,
    save_matching_checkpoint,
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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-val-texts", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run LoRA training with checkpointing and early stopping."""

    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    dataset_root = resolve_path(args.dataset_root or config["dataset_root"], PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir or config["output_weights_dir"], PROJECT_ROOT)
    batch_size = int(args.batch_size or config["batch_size"])
    epochs = int(args.epochs or config["epochs"])
    max_train_samples = args.max_train_samples
    max_val_images = args.max_val_images
    max_val_texts = args.max_val_texts
    if args.smoke_test:
        epochs = int(args.epochs or 1)
        max_train_samples = max_train_samples or 128
        max_val_images = max_val_images or 64
        max_val_texts = max_val_texts or 128
    output_dir.mkdir(parents=True, exist_ok=True)

    distributed_config = config.get("distributed", {})
    find_unused_parameters = (
        bool(distributed_config.get("find_unused_parameters", True))
        if isinstance(distributed_config, dict)
        else True
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters
    )
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    if accelerator.is_main_process:
        logger.info(
            "Using device=%s num_processes=%s find_unused_parameters=%s dataset_root=%s output_dir=%s",
            accelerator.device,
            accelerator.num_processes,
            find_unused_parameters,
            dataset_root,
            output_dir,
        )
        if args.smoke_test:
            logger.info("Smoke test enabled: epochs=%s max_train_samples=%s", epochs, max_train_samples)
    model, processor = build_clip_lora(config)

    dataset = CUHKPEDES(dataset_root)
    image_transform, tensor_transform = build_train_augmentation(
        config.get("augmentation", {}),
        clip_image_size(processor),
    )
    train_loader = DataLoader(
        CLIP_CUHK_Dataset(
            dataset.train,
            processor,
            mode="pair",
            image_transform=image_transform,
            tensor_transform=tensor_transform,
            max_samples=max_train_samples,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_img_loader = DataLoader(
        CLIP_CUHK_Dataset(
            dataset.val,
            processor,
            mode="image",
            max_samples=max_val_images,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_txt_loader = DataLoader(
        CLIP_CUHK_Dataset(
            dataset.val,
            processor,
            mode="text",
            max_samples=max_val_texts,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    id_head = build_id_head(model, dataset, config)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if id_head is not None:
        trainable_parameters.extend(id_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
    )
    if id_head is not None:
        model, id_head, optimizer, train_loader = accelerator.prepare(
            model,
            id_head,
            optimizer,
            train_loader,
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    best_rank1 = 0.0
    no_improve_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            accelerator,
            epoch,
            config.get("loss", {}),
            id_head,
        )
        accelerator.wait_for_everyone()

        should_stop = False
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_id_head = (
                accelerator.unwrap_model(id_head) if id_head is not None else None
            )
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            save_matching_checkpoint(unwrapped_model, checkpoint_dir, unwrapped_id_head)

            val_rank1 = evaluator.eval(unwrapped_model)
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
                save_matching_checkpoint(
                    unwrapped_model,
                    output_dir / "best_adapter",
                    unwrapped_id_head,
                )
                logger.info("New best Rank-1 %.3f at epoch %s.", best_rank1, epoch)
            else:
                no_improve_epochs += 1
                logger.info("No improvement for %s epoch(s).", no_improve_epochs)
                if no_improve_epochs >= int(config["patience"]):
                    logger.info("Early stopping triggered at epoch %s.", epoch)
                    should_stop = True

        stop_signal = torch.tensor(int(should_stop), device=accelerator.device)
        stop_signal = accelerator.reduce(stop_signal, reduction="max")
        if bool(stop_signal.item()):
            accelerator.wait_for_everyone()
            break
    return 0


def build_id_head(
    model: torch.nn.Module,
    dataset: CUHKPEDES,
    config: dict[str, Any],
) -> IdentityClassificationHead | None:
    """Build an optional identity classification head."""

    loss_config = config.get("loss", {})
    id_config = loss_config.get("id", {}) if isinstance(loss_config, dict) else {}
    if not id_config.get("enabled", False):
        return None
    embedding_dim = int(base_clip_model(model).config.projection_dim)
    num_classes = max(pid for pid, _image_id, _img_path, _caption in dataset.train) + 1
    return IdentityClassificationHead(embedding_dim, num_classes)


if __name__ == "__main__":
    raise SystemExit(main())
