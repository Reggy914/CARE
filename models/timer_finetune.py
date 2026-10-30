from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .normal_head import (
    _PredictionHead,
    _cfg,
    _extract_prediction,
    _flatten_time_series,
    _format_progress,
    _match_reference_shape,
    _persistence_prediction,
    _reference_from_xy,
    _restore_shape,
    _serializable_config,
    _unpack_batch,
)

from .timer import HuggingFaceTimerZeroShotForecaster, load_timer, stabilize_timer_input
from utils.training_control import ValidationEarlyStopping, evaluate_forecast


_SAFE_PARTIAL_NAMES = ("head", "proj", "projection", "adapter", "decoder", "output", "readout", "norm")


class TimerFinetuneModel(nn.Module):
    """Ordinary supervised finetuning baseline.

    ``full`` mode leaves all backbone parameters trainable. ``partial`` mode
    freezes the backbone first, then unfreezes conventional safe trainable
    portions by name. A plain output projection head is used when needed for
    shape compatibility or when partial mode has no safe backbone parameters to
    update. It contains no CARE Anchor, Expert, or Reconciliation modules.
    """

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(_cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(_cfg(self.config, "pred_len", 96))
        self.epochs = int(_cfg(self.config, "epochs", 10))
        self.learning_rate = float(_cfg(self.config, "learning_rate", 1e-4))
        self.weight_decay = float(_cfg(self.config, "weight_decay", 0.0))
        self.grad_clip = float(_cfg(self.config, "grad_clip", 1.0))
        self.finetune_mode = str(_cfg(self.config, "finetune_mode", "partial")).lower()
        hidden_dim = int(_cfg(self.config, "projection_hidden_dim", 128))
        dropout = float(_cfg(self.config, "dropout", 0.1))
        self.backbone = _cfg(self.config, "backbone", None)
        if self.backbone is None:
            self.backbone = load_timer(
                self.device,
                self.pred_len,
                _cfg(self.config, "timer_model_dir", None),
            )
        self.use_projection = bool(_cfg(self.config, "use_projection", self.finetune_mode == "partial" or self.backbone is None))
        self.output_projection = _PredictionHead(hidden_dim, 2, dropout)
        self.criterion = nn.MSELoss()
        self.to(self.device)
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        if self.backbone is None:
            return
        if not isinstance(self.backbone, nn.Module):
            return
        self.backbone.to(self.device)
        if self.finetune_mode == "full":
            self.backbone.train()
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            lname = name.lower()
            if any(token in lname for token in _SAFE_PARTIAL_NAMES):
                param.requires_grad = True

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        x = x.to(self.device).float()
        reference = _reference_from_xy(x, y_reference, self.pred_len)
        raw = self._backbone_prediction(x, reference)
        if not self.use_projection and tuple(raw.shape) == tuple(reference.shape):
            return raw
        features = _flatten_time_series(raw)
        flat_pred = self.output_projection(features, int(reference[0].numel()))
        return _restore_shape(flat_pred, reference)

    def _backbone_prediction(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return _persistence_prediction(x, reference)
        backbone_input = x
        if isinstance(self.backbone, HuggingFaceTimerZeroShotForecaster):
            # The trainable Timer path cannot use timer_base_prediction because
            # that helper deliberately runs under inference mode. Stabilizing
            # x directly keeps backbone gradients intact while preventing
            # RevIN from dividing by zero on constant/nearly constant windows.
            backbone_input, repaired_rows = stabilize_timer_input(x)
            if repaired_rows:
                print(
                    f"[timer_finetune/Timer] stabilized "
                    f"{repaired_rows}/{x.shape[0]} degenerate input window(s)",
                    flush=True,
                )
        if hasattr(self.backbone, "forward_trainable"):
            raw = self.backbone.forward_trainable(backbone_input)
        else:
            raw = self.backbone(backbone_input) if callable(self.backbone) else None
        pred = _extract_prediction(raw)
        if pred is None:
            return _persistence_prediction(x, reference)
        pred = pred.to(x.device).float()
        if self.use_projection:
            return pred
        return _match_reference_shape(pred, reference)

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        self.train()
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
            print(_format_progress("timer_finetune", epoch, self.epochs, row), flush=True)
            if val_loader is not None and control.stopped_early:
                print(f"[timer_finetune] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
                break
        if val_loader is not None:
            control.restore_best()
        return {"history": history, "training_control": control.summary(len(history)), "finetune_mode": self.finetune_mode}

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
                params = [p for p in self.parameters() if p.requires_grad]
                if not params:
                    raise ValueError("timer_finetune has no trainable parameters; provide a trainable backbone or enable projection")
                optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
            loss = self.criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad], self.grad_clip)
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

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": _serializable_config(self.config)}, path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "TimerFinetuneModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload["state_dict"])
        return self


__all__ = ["TimerFinetuneModel"]
