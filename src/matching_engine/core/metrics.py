"""Retrieval metrics and evaluator for CUHK-PEDES."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from src.matching_engine.core.clip_model import base_clip_model
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def rank(
    similarity: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    max_rank: int = 10,
    get_mAP: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CMC, mAP, mINP, and sorted gallery indices."""

    max_rank = min(max_rank, similarity.size(1))
    indices = (
        torch.argsort(similarity, dim=1, descending=True)
        if get_mAP
        else torch.topk(similarity, k=max_rank, dim=1).indices
    )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1).cpu())
    valid = matches.sum(1) > 0
    if not valid.any():
        zeros = torch.zeros(max_rank)
        return zeros, torch.tensor(0.0), torch.tensor(0.0), indices

    matches = matches[valid]
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    if not get_mAP:
        return all_cmc, torch.tensor(0.0), torch.tensor(0.0), indices

    num_rel = matches.sum(1)
    cumulative = matches.cumsum(1)
    inp = [
        cumulative[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100
    precision = torch.stack(
        [cumulative[:, i] / (i + 1.0) for i in range(cumulative.shape[1])],
        dim=1,
    )
    AP = (precision * matches).sum(1) / num_rel
    mAP = AP.mean() * 100
    return all_cmc, mAP, mINP, indices


class Evaluator:
    """Compute text-to-image retrieval metrics for CLIP-style models."""

    def __init__(
        self,
        img_loader: DataLoader[dict[str, torch.Tensor]],
        txt_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Store gallery and query loaders for evaluation."""

        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.last_metrics: dict[str, dict[str, float]] = {}

    def _compute_embedding(
        self,
        model: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract text query and image gallery embeddings."""

        model.eval()
        clip_model = self._clip_core_model(model)
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats = [], [], [], []

        for batch in self.txt_loader:
            labels = batch["label"]
            inputs = self._to_device(batch, device)
            with torch.no_grad():
                text_feat = self._encode_text_features(clip_model, inputs)
            qids.append(labels.view(-1).cpu())
            qfeats.append(text_feat.cpu())

        for batch in self.img_loader:
            labels = batch["label"]
            inputs = self._to_device(batch, device)
            with torch.no_grad():
                image_feat = self._encode_image_features(clip_model, inputs)
            gids.append(labels.view(-1).cpu())
            gfeats.append(image_feat.cpu())

        return (
            torch.cat(qfeats, dim=0),
            torch.cat(gfeats, dim=0),
            torch.cat(qids, dim=0),
            torch.cat(gids, dim=0),
        )

    def eval(self, model: Any, i2t_metric: bool = False) -> float:
        """Evaluate retrieval metrics and log a PrettyTable report."""

        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        similarity = qfeats @ gfeats.t()

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        self.last_metrics = {}
        self.last_metrics["t2i"] = self._add_metric_row(
            table,
            "t2i",
            similarity,
            qids,
            gids,
        )
        if i2t_metric:
            self.last_metrics["i2t"] = self._add_metric_row(
                table,
                "i2t",
                similarity.t(),
                gids,
                qids,
            )
        for column in ("R1", "R5", "R10", "mAP", "mINP"):
            table.custom_format[column] = lambda _field, value: f"{value:.3f}"
        logger.info("\n%s", table)
        return self.last_metrics["t2i"]["R1"]

    @staticmethod
    def _to_device(
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Move model inputs to device and drop labels."""

        return {
            key: value.to(device)
            for key, value in batch.items()
            if key != "label" and isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _clip_core_model(model: Any) -> Any:
        """Return the CLIP core model behind DDP or PEFT wrappers."""

        while hasattr(model, "module"):
            model = model.module
        clip_model = base_clip_model(model)
        peft_core = getattr(getattr(clip_model, "base_model", None), "model", None)
        if peft_core is not None:
            return peft_core
        return clip_model

    @staticmethod
    def _encode_text_features(
        clip_model: Any,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode text inputs into projected CLIP embeddings."""

        if hasattr(clip_model, "text_model") and hasattr(clip_model, "text_projection"):
            text_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            output = clip_model.text_model(**text_inputs, return_dict=True)
            return Evaluator._extract_feature(
                output,
                embed_attr="text_embeds",
                projection=clip_model.text_projection,
            )
        output = clip_model.get_text_features(**inputs)
        return Evaluator._extract_feature(
            output,
            embed_attr="text_embeds",
            projection=getattr(clip_model, "text_projection", None),
        )

    @staticmethod
    def _encode_image_features(
        clip_model: Any,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode image inputs into projected CLIP embeddings."""

        if hasattr(clip_model, "vision_model") and hasattr(clip_model, "visual_projection"):
            image_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {"pixel_values", "interpolate_pos_encoding"}
            }
            output = clip_model.vision_model(**image_inputs, return_dict=True)
            return Evaluator._extract_feature(
                output,
                embed_attr="image_embeds",
                projection=clip_model.visual_projection,
            )
        output = clip_model.get_image_features(**inputs)
        return Evaluator._extract_feature(
            output,
            embed_attr="image_embeds",
            projection=getattr(clip_model, "visual_projection", None),
        )

    @staticmethod
    def _extract_feature(
        output: Any,
        embed_attr: str,
        projection: torch.nn.Module | None,
    ) -> torch.Tensor:
        """Extract a tensor feature and project pooled model outputs if needed."""

        if isinstance(output, torch.Tensor):
            return output
        embed = getattr(output, embed_attr, None)
        if isinstance(embed, torch.Tensor):
            return embed
        if isinstance(output, dict):
            embed = output.get(embed_attr)
            if isinstance(embed, torch.Tensor):
                return embed
            pooled = output.get("pooler_output")
        else:
            pooled = getattr(output, "pooler_output", None)
        if not isinstance(pooled, torch.Tensor):
            raise TypeError(f"Cannot extract {embed_attr} from {type(output).__name__}.")
        return projection(pooled) if projection is not None else pooled

    @staticmethod
    def _add_metric_row(
        table: PrettyTable,
        task: str,
        similarity: torch.Tensor,
        qids: torch.Tensor,
        gids: torch.Tensor,
    ) -> dict[str, float]:
        """Compute metrics and append one table row."""

        cmc, mAP, mINP, _indices = rank(similarity, qids, gids, max_rank=10)
        values = cmc.tolist()
        metrics = {
            "R1": float(values[0]),
            "R5": float(values[min(4, len(values) - 1)]),
            "R10": float(values[min(9, len(values) - 1)]),
            "mAP": float(mAP),
            "mINP": float(mINP),
        }
        table.add_row(
            [
                task,
                metrics["R1"],
                metrics["R5"],
                metrics["R10"],
                metrics["mAP"],
                metrics["mINP"],
            ]
        )
        return metrics
