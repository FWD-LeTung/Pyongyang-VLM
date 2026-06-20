"""TBPS-CLIP inference wrapper for Matching Engine Phase 2."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class TBPSCLIPEncoder:
    """Production image-text encoder backed by the official TBPS-CLIP clone.

    NumPy arrays are treated as OpenCV-style BGR images because Module 2 crops
    come from OpenCV frames. PIL images and path-like inputs are converted to RGB.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        tbps_root: str | Path | None = None,
        device: str = "cuda",
        precision: str = "fp16",
        batch_size: int = 32,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self.tbps_root = Path(tbps_root or self._default_tbps_root()).resolve()
        self.config_path = Path(config_path or self.tbps_root / "config/config.yaml")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._resolve_device(
            device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.precision = precision
        self.use_half = self.device.type == "cuda" and precision == "fp16"
        self.batch_size = max(1, int(batch_size))
        self.allow_cpu_fallback = allow_cpu_fallback
        self.checkpoint_missing_keys_allowed: list[str] = []
        self.checkpoint_missing_keys_dangerous: list[str] = []
        self.checkpoint_unexpected_keys: list[str] = []

        if not self.tbps_root.exists():
            raise FileNotFoundError(f"TBPS-CLIP root not found: {self.tbps_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TBPS-CLIP checkpoint not found: {self.checkpoint_path}"
            )

        self._install_inference_stubs()
        self._add_tbps_root_to_path()
        self._tokenize = self._import_tokenizer()
        self._official_load_checkpoint = self._import_official_loader()
        self._model = self._build_model()
        self._image_transform = self._build_image_transform()
        logger.info(
            "tbps_device=%s precision=%s",
            self.device,
            "fp16" if self.use_half else "fp32",
        )

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode and L2-normalize text embeddings."""

        if not texts:
            return torch.empty(0, self._embedding_dim(), device=self.device)
        tokens = self._tokenize(
            list(texts),
            context_length=int(self._tbps_config.experiment.text_length),
        ).to(self.device)
        features = self._model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        """Encode and L2-normalize image embeddings."""

        if not images:
            return torch.empty(0, self._embedding_dim(), device=self.device)

        batches: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            pixel_values = torch.stack(
                [self._image_transform(self._to_pil_image(image)) for image in batch_images]
            ).to(self.device)
            if self.use_half:
                pixel_values = pixel_values.half()
            features = self._model.encode_image(pixel_values)
            batches.append(F.normalize(features.float(), dim=-1))
        return torch.cat(batches, dim=0)

    def _build_model(self) -> torch.nn.Module:
        tbps_model = importlib.import_module("model.tbps_model")
        tbps_model.EDA = _NoOpEDA
        self._tbps_config = self._build_tbps_config()
        num_classes = int(getattr(self._tbps_config.model, "num_classes", 11003))
        model = tbps_model.clip_vitb(self._tbps_config, num_classes=num_classes)
        model.to(self.device)
        if self.use_half:
            model.half()
        model.eval()

        try:
            model, load_result = self._official_load_checkpoint(
                model,
                self._tbps_config,
            )
        except Exception as exc:
            logger.warning(
                "Official TBPS-CLIP checkpoint loading failed: %s. "
                "Trying local compatibility loader.",
                exc,
            )
            load_result = self._load_checkpoint_compat(model)

        model.to(self.device)
        if self.use_half:
            model.half()
        model.eval()
        self._validate_checkpoint_keys(load_result)
        logger.info("Loaded TBPS-CLIP checkpoint: %s", self.checkpoint_path)
        logger.info("TBPS-CLIP load result: %s", load_result)
        logger.info(
            "checkpoint_missing_keys_allowed=%s",
            self.checkpoint_missing_keys_allowed,
        )
        logger.info(
            "checkpoint_missing_keys_dangerous=%s",
            self.checkpoint_missing_keys_dangerous,
        )
        logger.info("checkpoint_unexpected_keys=%s", self.checkpoint_unexpected_keys)
        return model

    def _build_tbps_config(self) -> "_AttrDict":
        raw_config: dict[str, Any] = {}
        if self.config_path.exists():
            loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("TBPS-CLIP config must be a YAML mapping.")
            raw_config = loaded

        config = _to_attr_dict(raw_config)
        config.setdefault("misc", _AttrDict())
        config.setdefault("experiment", _AttrDict())
        config.setdefault("model", _AttrDict())
        config.setdefault("distributed", _AttrDict())

        experiment_defaults = {
            "input_resolution": [224, 224],
            "simclr_mlp": [512, 128, 512],
            "simclr_temperature": 0.1,
            "dropout": 0.05,
            "eda_alpha": 0.05,
            "back_trans": False,
            "backtrans_p": 0.0,
            "text_length": 77,
            "mixgen": False,
            "mvs_image": False,
            "nitc_ratio": 1.0,
            "ss": False,
            "ss_ratio": 0.0,
            "ritc": False,
            "ritc_eps": 1.0e-2,
            "ritc_ratio": 1.0,
            "mlm": False,
            "mlm_ratio": 1.0,
            "cmt_depth": 4,
            "citc": False,
            "citc_lambda1": 0.25,
            "citc_lambda2": 0.25,
            "citc_ratio": 0.0,
            "id": False,
            "id_ratio": 1.0,
        }
        model_defaults = {
            "ckpt_type": "saved",
            "saved_path": str(self.checkpoint_path),
            "checkpoint": str(self.checkpoint_path),
            "use_gather": False,
            "embed_dim": 512,
            "vocab_size": 49408,
        }
        for key, value in experiment_defaults.items():
            config.experiment.setdefault(key, value)
        for key, value in model_defaults.items():
            config.model.setdefault(key, value)

        config.device = str(self.device)
        config.model.ckpt_type = "saved"
        config.model.saved_path = str(self.checkpoint_path)
        config.model.checkpoint = str(self.checkpoint_path)
        config.model.use_gather = False
        return config

    def _load_checkpoint_compat(self, model: torch.nn.Module) -> object:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_prefixes(state_dict, prefixes=("module.", "model."))
        return model.load_state_dict(state_dict, strict=False)

    def _validate_checkpoint_keys(self, load_result: object) -> None:
        """Fail fast if retrieval-critical checkpoint keys are missing."""

        missing_keys = list(getattr(load_result, "missing_keys", []) or [])
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
        allowed_prefixes = ("simclr_mlp.", "classifier.", "mlm_head.")

        allowed_missing = [
            key for key in missing_keys if key.startswith(allowed_prefixes)
        ]
        dangerous_missing = [
            key for key in missing_keys if not key.startswith(allowed_prefixes)
        ]

        self.checkpoint_missing_keys_allowed = allowed_missing
        self.checkpoint_missing_keys_dangerous = dangerous_missing
        self.checkpoint_unexpected_keys = unexpected_keys

        if dangerous_missing:
            raise RuntimeError(
                "TBPS-CLIP checkpoint is missing retrieval-critical keys: "
                f"{dangerous_missing}"
            )

    def _build_image_transform(self) -> transforms.Compose:
        size = self._tbps_config.experiment.input_resolution
        if isinstance(size, int):
            size = (size, size)
        else:
            size = tuple(int(value) for value in size)
        logger.info(
            "image_preprocess_source=official_tbps_clip input_size=%s "
            "mean=%s std=%s bgr_to_rgb=%s",
            size,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            True,
        )
        return transforms.Compose(
            [
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _to_pil_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
            if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                array = np.moveaxis(array, 0, -1)
            return self._array_to_pil(array, assume_bgr=False)
        if isinstance(image, np.ndarray):
            return self._array_to_pil(image, assume_bgr=True)
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    def _array_to_pil(self, array: np.ndarray, *, assume_bgr: bool) -> Image.Image:
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Expected image array with 2 or 3 dimensions, got {array.ndim}.")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image array, got shape {array.shape}.")

        array = _to_uint8(array)
        if assume_bgr:
            array = array[:, :, ::-1]
        return Image.fromarray(array, mode="RGB")

    def _embedding_dim(self) -> int:
        return int(getattr(self._tbps_config.model, "embed_dim", 512))

    def _add_tbps_root_to_path(self) -> None:
        root = str(self.tbps_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _import_tokenizer(self) -> Any:
        try:
            tokenizer = importlib.import_module("text_utils.tokenizer").tokenize
            logger.info("tokenizer_source=official_tbps_clip")
            return tokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import TBPS-CLIP tokenizer. Install the official "
                "TBPS-CLIP tokenizer dependencies or run through the project "
                "environment."
            ) from exc

    @staticmethod
    def _import_official_loader() -> Any:
        return importlib.import_module("misc.build").load_checkpoint

    @staticmethod
    def _resolve_device(device: str, *, allow_cpu_fallback: bool = False) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            message = (
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use --device cpu or pass --allow-cpu-fallback."
            )
            if not allow_cpu_fallback:
                raise RuntimeError(message)
            logger.warning("%s Falling back to CPU.", message)
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _default_tbps_root() -> Path:
        return Path(__file__).resolve().parents[1] / "TBPS-CLIP"

    @staticmethod
    def _install_inference_stubs() -> None:
        """Patch training-only official imports for single-process inference."""

        if "ftfy" not in sys.modules:
            try:
                importlib.import_module("ftfy")
                logger.info("text_cleanup_source=official_ftfy")
            except ModuleNotFoundError:
                ftfy_stub = types.ModuleType("ftfy")
                ftfy_stub.fix_text = lambda text: text
                sys.modules["ftfy"] = ftfy_stub
                logger.warning("ftfy is unavailable; using identity text cleanup.")
        else:
            logger.info("text_cleanup_source=official_ftfy")

        if "easydict" not in sys.modules:
            easydict_stub = types.ModuleType("easydict")
            easydict_stub.EasyDict = _AttrDict
            _AttrDict.__module__ = "easydict"
            sys.modules["easydict"] = easydict_stub

        misc_utils = types.ModuleType("misc.utils")
        misc_utils.is_using_distributed = lambda: False
        misc_utils.is_dist_avail_and_initialized = lambda: False
        misc_utils.get_world_size = lambda: 1
        misc_utils.get_rank = lambda: 0
        misc_utils.is_master = lambda: True
        sys.modules.setdefault("misc.utils", misc_utils)

        eda_stub = types.ModuleType("model.eda")
        eda_stub.EDA = _NoOpEDA
        sys.modules.setdefault("model.eda", eda_stub)


class _NoOpEDA:
    """Training augmentation stub; retrieval inference never uses EDA."""

    def random_deletion(self, sentence: str, _p: float = 0.0) -> str:
        return sentence


class _AttrDict(dict):
    """Tiny EasyDict-compatible mapping for official TBPS config access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping or contain a state dict.")
    for key in ("model", "state_dict", "model_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Could not find a model state dict in checkpoint.")


def _strip_prefixes(
    state_dict: dict[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
