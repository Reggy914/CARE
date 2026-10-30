from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .normal_head import (
    _PredictionHead,
    _base_prediction,
    _cfg,
    _flatten_time_series,
    _format_progress,
    _reference_from_xy,
    _restore_shape,
    _serializable_config,
    _unpack_batch,
)

from .timer import load_shared_timer
from utils.training_control import ValidationEarlyStopping, evaluate_forecast



class OutputCorrectionModel(nn.Module):
    """Frozen-backbone baseline with an ordinary residual correction MLP.

    The correction network sees only historical input ``x`` and frozen base
    prediction ``y_base``. It never receives future labels as features. Final
    prediction is ``y_base + residual_scale * delta``. It contains no CARE
    Anchor, Expert, or Reconciliation modules.
    """

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(_cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(_cfg(self.config, "pred_len", 96))
        self.epochs = int(_cfg(self.config, "epochs", 10))
        self.learning_rate = float(_cfg(self.config, "learning_rate", 1e-3))
        self.weight_decay = float(_cfg(self.config, "weight_decay", 0.0))
        self.residual_scale = float(_cfg(self.config, "residual_scale", 1.0))
        hidden_dim = int(_cfg(self.config, "correction_hidden_dim", 128))
        num_layers = int(_cfg(self.config, "correction_num_layers", 2))
        dropout = float(_cfg(self.config, "dropout", 0.1))
        self.backbone = _cfg(self.config, "backbone", None)
        if self.backbone is None:
            self.backbone = load_shared_timer(
                self.device,
                self.pred_len,
                _cfg(self.config, "timer_model_dir", None),
            )
        self.freeze_backbone = bool(_cfg(self.config, "freeze_backbone", True))
        self.correction_net = _PredictionHead(hidden_dim, num_layers, dropout)
        self.criterion = nn.MSELoss()
        self.to(self.device)
        self._freeze_backbone_if_needed()

    def _freeze_backbone_if_needed(self) -> None:
        if self.backbone is None or not self.freeze_backbone:
            return
        if isinstance(self.backbone, nn.Module):
            self.backbone.to(self.device)
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        x = x.to(self.device).float()
        reference = _reference_from_xy(x, y_reference, self.pred_len)
        with torch.no_grad():
            y_base = _base_prediction(self.backbone, x, reference)
        features = torch.cat([_flatten_time_series(x), _flatten_time_series(y_base)], dim=-1)
        flat_delta = self.correction_net(features, int(reference[0].numel()))
        delta = _restore_shape(flat_delta, reference)
        return y_base + self.residual_scale * delta

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        self.train()
        if self.backbone is not None and isinstance(self.backbone, nn.Module):
            self.backbone.eval()
        optimizer: torch.optim.Optimizer | None = None
        history: list[dict[str, float]] = []
        control = ValidationEarlyStopping(self, self.config)
        for epoch in range(1, self.epochs + 1):
            train_loss, optimizer = self._run_epoch(train_loader, optimizer)
            row = {"epoch": float(epoch), "train_loss": train_loss}
            if val_loader is not None:
                row.update(evaluate_forecast(self, val_loader, self.device))
                is_best = control.update(row["val_mae"], epoch)
                control.annotate(row, is_best=is_best)
            history.append(row)
            print(_format_progress("output_correction", epoch, self.epochs, row), flush=True)
            if val_loader is not None and control.stopped_early:
                print(f"[output_correction] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
                break
        if val_loader is not None:
            control.restore_best()
        return {"history": history, "training_control": control.summary(len(history))}

    def _run_epoch(self, loader: Any, optimizer: torch.optim.Optimizer | None) -> tuple[float, torch.optim.Optimizer]:
        self.train()
        total = 0.0
        count = 0
        for batch in loader:
            x, y = _unpack_batch(batch)
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            pred = self.forward(x, y)
            if optimizer is None:
                optimizer = torch.optim.AdamW(self.correction_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            loss = self.criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(y.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
        if optimizer is None:
            raise ValueError("train_loader produced no batches")
        return total / max(count, 1), optimizer

    def evaluate_loss(self, loader: Any) -> float:
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                x, y = _unpack_batch(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                pred = self.forward(x, y)
                loss = self.criterion(pred, y)
                batch_size = int(y.shape[0])
                total += float(loss.detach().cpu()) * batch_size
                count += batch_size
        return total / max(count, 1)

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        self.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = _unpack_batch(batch, require_y=False)
                x = x.to(self.device).float()
                y_ref = None if y is None else y.to(self.device).float()
                preds.append(self.forward(x, y_ref).detach().cpu())
        output = torch.cat(preds, dim=0)
        return output.numpy() if return_numpy else output

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": _serializable_config(self.config)}, path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "OutputCorrectionModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload["state_dict"])
        return self


__all__ = ["OutputCorrectionModel"]
