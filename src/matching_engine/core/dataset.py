"""CUHK-PEDES dataset loaders for CLIP training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ImageSource: TypeAlias = str | Path | Image.Image
PairSample: TypeAlias = tuple[int, int, ImageSource, str]
DatasetMode = Literal["pair", "image", "text"]


class CUHKPEDES:
    """Read CUHK-PEDES reid_raw.json and expose train/val/test splits."""

    def __init__(self, root: str | Path) -> None:
        """Initialize dataset paths and materialize all splits."""

        self.dataset_root = Path(root)
        self.img_dir = self.dataset_root / "imgs"
        self.anno_path = self.dataset_root / "reid_raw.json"
        self._check_before_run()

        annotations = self._read_annotations()
        self.train = self._pairs(annotations, "train")
        self.val = self._pairs(annotations, "val")
        self.test = self._eval_split(annotations, "test")

    def _check_before_run(self) -> None:
        """Validate required CUHK-PEDES paths."""

        for path in (self.dataset_root, self.img_dir, self.anno_path):
            if not path.exists():
                raise RuntimeError(f"{path} is not available.")

    def _read_annotations(self) -> list[dict[str, Any]]:
        """Load reid_raw.json annotations."""

        with self.anno_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("reid_raw.json must contain a list of annotations.")
        return data

    def _pairs(self, annotations: list[dict[str, Any]], split: str) -> list[PairSample]:
        """Build training-style caption-image pairs for one split."""

        samples: list[PairSample] = []
        image_id = 0
        for annotation in annotations:
            if annotation.get("split") != split:
                continue
            pid = int(annotation["id"]) - 1
            img_path = self.img_dir / str(annotation["file_path"])
            for caption in annotation["captions"]:
                samples.append((pid, image_id, img_path, str(caption)))
            image_id += 1
        return samples

    def _eval_split(
        self,
        annotations: list[dict[str, Any]],
        split: str,
    ) -> dict[str, list[Any]]:
        """Build grouped image/text arrays for retrieval evaluation."""

        image_pids: list[int] = []
        img_paths: list[str] = []
        caption_pids: list[int] = []
        captions: list[str] = []
        for annotation in annotations:
            if annotation.get("split") != split:
                continue
            pid = int(annotation["id"])
            image_pids.append(pid)
            img_paths.append(str(self.img_dir / str(annotation["file_path"])))
            for caption in annotation["captions"]:
                caption_pids.append(pid)
                captions.append(str(caption))
        return {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions,
        }


class CLIP_CUHK_Dataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch adapter that applies a CLIP processor to CUHK-PEDES samples."""

    def __init__(
        self,
        data: list[PairSample] | dict[str, list[Any]],
        processor: Any,
        mode: DatasetMode = "pair",
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
        tensor_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Store the processor and normalize samples for the selected mode."""

        self.processor = processor
        self.mode = mode
        self.image_transform = image_transform
        self.tensor_transform = tensor_transform
        self.samples = self._build_samples(data, mode)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sample and return processor tensors plus label."""

        label, image_source, caption = self.samples[index]
        image = self._load_image(image_source) if image_source is not None else None
        if image is not None and self.image_transform is not None:
            image = self.image_transform(image)
        encoded = self.processor(
            text=caption,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        if "pixel_values" in item and self.tensor_transform is not None:
            item["pixel_values"] = self.tensor_transform(item["pixel_values"])
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item

    def _build_samples(
        self,
        data: list[PairSample] | dict[str, list[Any]],
        mode: DatasetMode,
    ) -> list[tuple[int, ImageSource | None, str | None]]:
        """Normalize pair or grouped eval data into adapter samples."""

        if isinstance(data, dict):
            return self._samples_from_eval_dict(data, mode)
        return self._samples_from_pairs(data, mode)

    @staticmethod
    def _samples_from_pairs(
        pairs: list[PairSample],
        mode: DatasetMode,
    ) -> list[tuple[int, ImageSource | None, str | None]]:
        """Create samples from train/val pair tuples."""

        if mode == "pair":
            return [(pid, img_path, caption) for pid, _image_id, img_path, caption in pairs]
        if mode == "text":
            return [(pid, None, caption) for pid, _image_id, _img_path, caption in pairs]

        seen: set[int] = set()
        samples: list[tuple[int, ImageSource | None, str | None]] = []
        for pid, image_id, img_path, _caption in pairs:
            if image_id in seen:
                continue
            seen.add(image_id)
            samples.append((pid, img_path, None))
        return samples

    @staticmethod
    def _samples_from_eval_dict(
        data: dict[str, list[Any]],
        mode: DatasetMode,
    ) -> list[tuple[int, ImageSource | None, str | None]]:
        """Create samples from grouped retrieval dictionaries."""

        if mode == "image":
            return [
                (int(pid), str(img_path), None)
                for pid, img_path in zip(data["image_pids"], data["img_paths"])
            ]
        if mode == "text":
            return [
                (int(pid), None, str(caption))
                for pid, caption in zip(data["caption_pids"], data["captions"])
            ]
        raise ValueError("Grouped eval data supports only image or text mode.")

    def _load_image(self, image_source: ImageSource) -> Image.Image:
        """Load a PIL image and fail fast with a clear path on error."""

        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        try:
            return Image.open(image_source).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_source}") from exc


def build_train_augmentation(
    config: dict[str, Any],
    image_size: int,
) -> tuple[Callable[[Image.Image], Image.Image] | None, Callable[[torch.Tensor], torch.Tensor] | None]:
    """Build optional train-only image and tensor augmentations."""

    if not config.get("enabled", False):
        return None, None

    image_transforms: list[Callable[[Image.Image], Image.Image]] = []
    crop_config = config.get("random_resized_crop", {})
    if crop_config.get("enabled", False):
        image_transforms.append(
            transforms.RandomResizedCrop(
                size=image_size,
                scale=tuple(crop_config.get("scale", [0.85, 1.0])),
            )
        )

    flip_config = config.get("horizontal_flip", {})
    if flip_config.get("enabled", False):
        image_transforms.append(
            transforms.RandomHorizontalFlip(p=float(flip_config.get("p", 0.5)))
        )

    rotation_config = config.get("rotation", {})
    if rotation_config.get("enabled", False):
        image_transforms.append(
            transforms.RandomRotation(degrees=float(rotation_config.get("degrees", 5)))
        )

    jitter_config = config.get("color_jitter", {})
    if jitter_config.get("enabled", False):
        image_transforms.append(
            transforms.ColorJitter(
                brightness=float(jitter_config.get("brightness", 0.1)),
                contrast=float(jitter_config.get("contrast", 0.1)),
                saturation=float(jitter_config.get("saturation", 0.1)),
                hue=float(jitter_config.get("hue", 0.0)),
            )
        )

    erase_config = config.get("random_erasing", {})
    tensor_transform = (
        transforms.RandomErasing(p=float(erase_config.get("p", 0.1)))
        if erase_config.get("enabled", False)
        else None
    )
    image_transform = transforms.Compose(image_transforms) if image_transforms else None
    return image_transform, tensor_transform


def clip_image_size(processor: Any) -> int:
    """Return the CLIP processor image size for augmentation crops."""

    size = getattr(getattr(processor, "image_processor", None), "size", None)
    if isinstance(size, dict):
        return int(size.get("height") or size.get("shortest_edge") or 224)
    if isinstance(size, int):
        return size
    return 224
