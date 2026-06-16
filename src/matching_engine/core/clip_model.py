"""CLIP + LoRA model construction for Matching Engine Phase 1."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import CLIPModel, CLIPProcessor


def build_clip_lora(config: dict[str, Any]) -> tuple[PeftModel, CLIPProcessor]:
    """Build a frozen CLIP backbone with trainable LoRA adapters."""

    quantization = config.get("quantization", {})
    model_kwargs: dict[str, Any] = {}
    if quantization.get("enabled", False):
        # TODO: Version 2 will add BitsAndBytesConfig for 4-bit/8-bit loading.
        raise NotImplementedError("Quantized CLIP loading is reserved for Version 2.")

    model = CLIPModel.from_pretrained(config["model_name"], **model_kwargs)
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model, processor


def base_clip_model(model: Any) -> CLIPModel:
    """Return the underlying CLIPModel from PEFT or plain CLIP wrappers."""

    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model
