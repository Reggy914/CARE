from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .timer import HuggingFaceTimerZeroShotForecaster, load_shared_timer, timer_base_prediction
from utils.training_control import ValidationEarlyStopping, evaluate_forecast


class _PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout = float(dropout)
        self.output_size: int | None = None
        self.net: nn.Sequential | None = None

    def _build(self, output_size: int, device: torch.device) -> None:
        layers: list[nn.Module] = []
        for _ in range(max(0, self.num_layers - 1)):
            layers.extend([nn.LazyLinear(self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)])
        layers.append(nn.LazyLinear(int(output_size)))
        self.net = nn.Sequential(*layers).to(device)
        self.output_size = int(output_size)

    def forward(self, features: torch.Tensor, output_size: int) -> torch.Tensor:
        if self.net is None:
            self._build(output_size, features.device)
        if self.output_size != int(output_size):
            raise ValueError(f"output size changed from {self.output_size} to {output_size}; use a consistent target shape per model instance")
        return self.net(features)


class NormalHeadModel(nn.Module):
    """Frozen-backbone baseline with a lightweight ordinary MLP head.

    The head uses only the frozen backbone prediction as its feature. If no
    backbone is supplied, a causal persistence prediction from the last history
    value is used so the class can still run shape smoke tests without data or
    pretrained assets. It contains no CARE Anchor, Expert, or Reconciliation
    modules.
    """

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(_cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(_cfg(self.config, "pred_len", 96))
        self.epochs = int(_cfg(self.config, "epochs", 10))
        self.learning_rate = float(_cfg(self.config, "learning_rate", 1e-3))
        self.weight_decay = float(_cfg(self.config, "weight_decay", 0.0))
        hidden_dim = int(_cfg(self.config, "head_hidden_dim", 128))
        num_layers = int(_cfg(self.config, "head_num_layers", 2))
        dropout = float(_cfg(self.config, "dropout", 0.1))
        self.backbone = _cfg(self.config, "backbone", None)
        if self.backbone is None:
            self.backbone = load_shared_timer(
                self.device,
                self.pred_len,
                _cfg(self.config, "timer_model_dir", None),
            )
        self.freeze_backbone = bool(_cfg(self.config, "freeze_backbone", True))
        self.head = _PredictionHead(hidden_dim, num_layers, dropout)
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
        features = _flatten_time_series(y_base)
        flat_pred = self.head(features, int(reference[0].numel()))
        return _restore_shape(flat_pred, reference)

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
            print(_format_progress("normal_head", epoch, self.epochs, row), flush=True)
            if val_loader is not None and control.stopped_early:
                print(f"[normal_head] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
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
                optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "NormalHeadModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload["state_dict"])
        return self


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _serializable_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if _is_json_scalar(v)}
    return {k: v for k, v in vars(config).items() if _is_json_scalar(v)} if hasattr(config, "__dict__") else {}


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, tuple, dict))


def _unpack_batch(batch: Any, *, require_y: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("inputs")))
        y = batch.get("y", batch.get("target", batch.get("targets")))
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
    else:
        x = batch
        y = None
    if x is None:
        raise ValueError("batch does not contain an input tensor")
    if y is None and require_y:
        raise ValueError("batch does not contain a target tensor")
    return torch.as_tensor(x), None if y is None else torch.as_tensor(y)


def _flatten_time_series(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _reference_from_xy(x: torch.Tensor, y: torch.Tensor | None, pred_len: int) -> torch.Tensor:
    if y is not None:
        return y.to(x.device).float()
    if x.ndim == 2:
        return x.new_zeros((x.shape[0], pred_len))
    if x.ndim == 3:
        return x.new_zeros((x.shape[0], pred_len, x.shape[-1]))
    return x.new_zeros((x.shape[0], pred_len))


def _restore_shape(pred_flat: torch.Tensor, reference_y: torch.Tensor) -> torch.Tensor:
    return pred_flat.reshape(reference_y.shape)


def _base_prediction(backbone: Any, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if backbone is not None:
        if isinstance(backbone, HuggingFaceTimerZeroShotForecaster):
            # Reuse the canonical Timer numerical guard, including degenerate
            # window stabilization and the existing finite-output fallback.
            # Clone after inference mode so trainable heads can save the
            # resulting features for their own backward pass.
            raw = timer_base_prediction(backbone, x).clone()
        else:
            raw = backbone(x) if callable(backbone) else None
        raw = _extract_prediction(raw)
        if raw is not None:
            return _match_reference_shape(raw.to(x.device).float(), reference)
    return _persistence_prediction(x, reference)


def _extract_prediction(raw: Any) -> torch.Tensor | None:
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, dict):
        for key in ("prediction", "predictions", "y_pred", "output", "logits"):
            if key in raw and isinstance(raw[key], torch.Tensor):
                return raw[key]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _match_reference_shape(pred: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tuple(pred.shape) == tuple(reference.shape):
        return pred
    flat = pred.reshape(pred.shape[0], -1)
    needed = int(reference[0].numel())
    if flat.shape[1] < needed:
        pad = flat[:, -1:].repeat(1, needed - flat.shape[1]) if flat.shape[1] else flat.new_zeros((flat.shape[0], needed))
        flat = torch.cat([flat, pad], dim=1)
    return flat[:, :needed].reshape(reference.shape)


def _persistence_prediction(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        last = x[:, -1:]
        return last.repeat(1, reference.shape[1]).reshape(reference.shape)
    if x.ndim == 3:
        last = x[:, -1:, :]
        return last.repeat(1, reference.shape[1], 1).reshape(reference.shape)
    flat = x.reshape(x.shape[0], -1)
    last = flat[:, -1:]
    return last.repeat(1, int(reference[0].numel())).reshape(reference.shape)


def _format_progress(model_name: str, epoch: int, epochs: int, row: dict[str, float]) -> str:
    parts = [f"[{model_name}] epoch={epoch}/{epochs}", f"train_loss={row['train_loss']:.6g}"]
    if "val_loss" in row:
        parts.append(f"val_loss={row['val_loss']:.6g}")
    if "val_mae" in row:
        parts.append(f"val_mae={row['val_mae']:.6g}")
    if "best_epoch" in row:
        parts.append(f"best_epoch={int(row['best_epoch'])}")
        parts.append(f"patience={int(row['epochs_without_improvement'])}")
    return " | ".join(parts)
