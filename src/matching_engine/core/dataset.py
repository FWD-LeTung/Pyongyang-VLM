"""CUHK-PEDES dataset loaders for CLIP training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from PIL import Image
from torch.utils.data import Dataset


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
    ) -> None:
        """Store the processor and normalize samples for the selected mode."""

        self.processor = processor
        self.mode = mode
        self.samples = self._build_samples(data, mode)

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sample and return processor tensors plus label."""

        label, image_source, caption = self.samples[index]
        image = self._load_image(image_source) if image_source is not None else None
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
        """Load a PIL image, falling back to the first sample image on error."""

        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        try:
            return Image.open(image_source).convert("RGB")
        except Exception:
            fallback = self._first_image_source()
            if fallback is None or fallback == image_source:
                raise
            return self._load_image(fallback)

    def _first_image_source(self) -> ImageSource | None:
        """Return first available image source in this dataset."""

        for _label, image_source, _caption in self.samples:
            if image_source is not None:
                return image_source
        return None
