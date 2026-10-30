from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _cfg(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _unpack_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("inputs")))
        y = batch.get("y", batch.get("target", batch.get("targets")))
    elif isinstance(batch, (list, tuple)):
        x = batch[0] if batch else None
        y = batch[1] if len(batch) > 1 else None
    else:
        x, y = batch, None
    if x is None or y is None:
        raise ValueError("validation batches must contain input and target tensors")
    return torch.as_tensor(x), torch.as_tensor(y)


@torch.inference_mode()
def evaluate_forecast(model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    """Return element-weighted MSE and MAE in the standardized data space."""
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for batch in loader:
        x, y = _unpack_xy(batch)
        x = x.to(device).float()
        y = y.to(device).float()
        prediction = model(x, y)
        if tuple(prediction.shape) != tuple(y.shape):
            if prediction.shape[0] != y.shape[0] or prediction.numel() != y.numel():
                raise ValueError(
                    f"validation prediction shape {tuple(prediction.shape)} does not match "
                    f"target shape {tuple(y.shape)}"
                )
            prediction = prediction.reshape(y.shape)
        error = prediction - y
        if not torch.isfinite(error).all():
            raise FloatingPointError("validation prediction contains NaN or Inf")
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        count += int(error.numel())
    if count == 0:
        raise ValueError("validation loader produced no target elements")
    return {"val_loss": squared_error / count, "val_mae": absolute_error / count}


class ValidationEarlyStopping:
    """Track and restore the validation-best trainable parameters only.

    Frozen parameters (notably the Timer backbone) are deliberately excluded
    from the in-memory snapshot. This changes training control only; model
    architecture, forward computation, and optimization losses stay untouched.
    """

    monitor = "val_mae"

    def __init__(self, model: nn.Module, config: Any) -> None:
        self.model = model
        self.enabled = bool(_cfg(config, "early_stopping", True))
        self.patience = max(1, int(_cfg(config, "early_stopping_patience", 3)))
        self.min_delta = max(0.0, float(_cfg(config, "early_stopping_min_delta", 1.0e-4)))
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self._patience_reference = float("inf")
        self._best_parameters: dict[str, torch.Tensor] = {}
        self.stopped_early = False

    def _snapshot(self) -> None:
        snapshot = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if not snapshot:
            raise ValueError("early stopping found no trainable model parameters")
        self._best_parameters = snapshot

    def update(self, metric: float, epoch: int) -> bool:
        value = float(metric)
        if not math.isfinite(value):
            raise FloatingPointError(f"{self.monitor} is NaN or Inf at epoch {epoch}")

        is_best = value < self.best_metric
        if is_best:
            self.best_metric = value
            self.best_epoch = int(epoch)
            self._snapshot()

        if value < self._patience_reference - self.min_delta:
            self._patience_reference = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        self.stopped_early = self.enabled and self.bad_epochs >= self.patience
        return is_best

    def restore_best(self) -> None:
        if not self._best_parameters:
            return
        current = dict(self.model.named_parameters())
        missing = sorted(set(self._best_parameters) - set(current))
        if missing:
            raise RuntimeError(f"cannot restore missing trainable parameters: {missing}")
        with torch.no_grad():
            for name, saved in self._best_parameters.items():
                parameter = current[name]
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))

    def annotate(self, row: dict[str, Any], *, is_best: bool) -> None:
        row.update(
            {
                "is_best": bool(is_best),
                "best_epoch": int(self.best_epoch),
                "best_val_mae": float(self.best_metric),
                "epochs_without_improvement": int(self.bad_epochs),
            }
        )

    def summary(self, epochs_ran: int) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_epoch": int(self.best_epoch),
            "best_metric": None if self.best_epoch == 0 else float(self.best_metric),
            "epochs_ran": int(epochs_ran),
            "stopped_early": bool(self.stopped_early),
            "early_stopping_enabled": bool(self.enabled),
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "restored_best_parameters": bool(self._best_parameters),
        }


__all__ = ["ValidationEarlyStopping", "evaluate_forecast"]
