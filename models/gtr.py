from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .timer import cfg, load_shared_timer, serializable_config, timer_base_prediction, timer_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Local GTRAdapter — replaces external dependency on 3_betterAI
# ---------------------------------------------------------------------------

class _GTRAdapter:
    """Leakage-safe retrieval adapter.

    Builds a retrieval bank from training windows and blends Timer predictions
    with matched training continuations.  No learnable parameters — purely
    nearest-neighbour retrieval over the train set.
    """

    def __init__(
        self,
        topk: int = 3,
        fusion_weight: float = 0.2,
        mode: str = "output",
        max_bank_size: int = 16384,
        bank_strategy: str = "segmented",
        distance_temperature: float = 1.0,
        dynamic_fusion: bool = True,
        dynamic_fusion_scale: float = 10.0,
        min_fusion_weight: float = 0.0,
        seed: int = 2024,
    ):
        self.topk = int(topk)
        self.fusion_weight = float(fusion_weight)
        self.mode = mode
        self.max_bank_size = int(max_bank_size)
        self.bank_strategy = bank_strategy
        self.distance_temperature = max(float(distance_temperature), 1e-6)
        self.dynamic_fusion = bool(dynamic_fusion)
        self.dynamic_fusion_scale = max(float(dynamic_fusion_scale), 1e-6)
        self.min_fusion_weight = max(0.0, min(1.0, float(min_fusion_weight)))
        self.seed = int(seed)
        self.bank_x: torch.Tensor | None = None
        self.bank_y: torch.Tensor | None = None

    # -- bank construction ---------------------------------------------------

    def _select_bank_indices(self, dataset_len: int) -> torch.Tensor:
        bank_size = min(max(self.max_bank_size, 1), int(dataset_len))
        if self.bank_strategy == "sequential" or bank_size >= dataset_len:
            return torch.arange(bank_size, dtype=torch.long)
        if self.bank_strategy == "random":
            generator = torch.Generator().manual_seed(self.seed)
            return torch.randperm(int(dataset_len), generator=generator)[:bank_size].sort().values
        return torch.linspace(0, int(dataset_len) - 1, steps=bank_size).round().long().unique()

    def _fit_from_dataset(self, dataset: Any) -> None:
        indices = self._select_bank_indices(len(dataset))
        xs, ys = [], []
        for idx in indices.tolist():
            sample = dataset[int(idx)]
            x, y = sample[:2]
            xs.append(torch.as_tensor(x).float())
            ys.append(torch.as_tensor(y).float())
        if not xs:
            raise ValueError("Cannot build GTR retrieval bank from an empty train dataset.")
        self.bank_x = torch.stack(xs, dim=0)
        self.bank_y = torch.stack(ys, dim=0)

    def _fit_from_loader_prefix(self, train_data: Any) -> None:
        xs, ys = [], []
        seen = 0
        for batch in train_data:
            remaining = self.max_bank_size - seen
            if remaining <= 0:
                break
            x = torch.as_tensor(batch[0][:remaining]).float()
            y = torch.as_tensor(batch[1][:remaining]).float()
            xs.append(x)
            ys.append(y)
            seen += x.shape[0]
        if not xs:
            raise ValueError("Cannot build GTR retrieval bank from an empty train loader.")
        self.bank_x = torch.cat(xs, dim=0)
        self.bank_y = torch.cat(ys, dim=0)

    def fit(self, train_data: Any = None, context: dict[str, Any] | None = None) -> "_GTRAdapter":
        if train_data is None:
            raise ValueError("GTRAdapter.fit requires train_data to build a leakage-safe retrieval bank.")
        dataset = getattr(train_data, "dataset", None)
        if dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            self._fit_from_dataset(dataset)
        else:
            self._fit_from_loader_prefix(train_data)
        return self

    # -- enhancement ---------------------------------------------------------

    def _dynamic_weight(self, nearest_distance: torch.Tensor) -> torch.Tensor:
        base_weight = max(0.0, min(1.0, self.fusion_weight))
        if not self.dynamic_fusion:
            return torch.full_like(nearest_distance, base_weight).reshape(-1, 1, 1)
        confidence = torch.exp(-nearest_distance / self.dynamic_fusion_scale)
        weight = base_weight * confidence
        if self.min_fusion_weight > 0.0:
            weight = torch.clamp(weight, min=self.min_fusion_weight, max=base_weight)
        return weight.reshape(-1, 1, 1)

    def enhance_output(self, y_pred: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        if self.mode not in {"output", "both"}:
            return y_pred
        if self.bank_x is None or self.bank_y is None:
            return y_pred
        x = context.get("x") if context else None
        if x is None:
            return y_pred

        bank_x = self.bank_x.to(device=x.device, dtype=x.dtype)
        bank_y = self.bank_y.to(device=y_pred.device, dtype=y_pred.dtype)

        query = x.reshape(x.shape[0], -1)
        keys = bank_x.reshape(bank_x.shape[0], -1)
        distances = torch.cdist(query, keys)
        k = min(self.topk, keys.shape[0])
        topk_dist, topk_idx = torch.topk(distances, k=k, dim=1, largest=False)

        neighbor_weights = torch.softmax(-topk_dist / self.distance_temperature, dim=1)
        retrieved = (bank_y[topk_idx] * neighbor_weights[..., None, None]).sum(dim=1)

        if retrieved.shape[1] != y_pred.shape[1]:
            retrieved = retrieved[:, : y_pred.shape[1], :]
        weight = self._dynamic_weight(topk_dist[:, 0]).to(device=y_pred.device, dtype=y_pred.dtype)
        return (1.0 - weight) * y_pred + weight * retrieved


# ---------------------------------------------------------------------------
# GTR model
# ---------------------------------------------------------------------------

class GTRNet(nn.Module):
    """Reference-compatible GTR wrapper: train-only retrieval over Timer output."""

    def __init__(self) -> None:
        super().__init__()


class GTRModel(nn.Module):
    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        self.timer_model = load_shared_timer(self.device, self.pred_len, cfg(self.config, "timer_model_dir", None) or None)
        self.contract = timer_contract(self.pred_len, cfg(self.config, "timer_model_dir", None) or None)
        self.enhancer: _GTRAdapter | None = None

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        self.timer_model.eval()
        self.enhancer = _GTRAdapter(seed=int(cfg(self.config, "seed", 2024)))
        self.enhancer.fit(train_data=train_loader)
        return {
            "status": "fit_train_only_retrieval_bank",
            "uses_timer0_base": True,
            "timer_contract": self.contract,
            "bank_shape_x": None if self.enhancer.bank_x is None else list(self.enhancer.bank_x.shape),
            "bank_shape_y": None if self.enhancer.bank_y is None else list(self.enhancer.bank_y.shape),
            "timer_frozen": True,
        }

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
        y_timer = timer_base_prediction(self.timer_model, x)
        if self.enhancer is None:
            return y_timer
        with torch.no_grad():
            final = self.enhancer.enhance_output(y_timer, context={"x": x})
        return final

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        self.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in test_loader:
                preds.append(self.forward(batch[0]).detach().cpu())
        out = torch.cat(preds, dim=0)
        return out.numpy() if return_numpy else out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": serializable_config(self.config), "timer_contract": self.contract, "uses_timer0_base": True}
        if self.enhancer is not None:
            payload["gtr_bank_x"] = self.enhancer.bank_x
            payload["gtr_bank_y"] = self.enhancer.bank_y
        torch.save(payload, path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "GTRModel":
        payload = torch.load(path, map_location=map_location or self.device)
        if payload.get("gtr_bank_x") is not None:
            self.enhancer = _GTRAdapter(seed=int(cfg(self.config, "seed", 2024)))
            self.enhancer.bank_x = payload["gtr_bank_x"]
            self.enhancer.bank_y = payload["gtr_bank_y"]
        return self


__all__ = ["GTRNet", "GTRModel"]
