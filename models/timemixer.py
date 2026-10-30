from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.tensor_shapes import ensure_3d as _ensure_3d, restore_shape as _restore_shape
from .normal_head import _cfg, _format_progress, _serializable_config, _unpack_batch
from utils.training_control import ValidationEarlyStopping, evaluate_forecast


class _ScaleMixerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.LazyLinear(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeMixerNet(nn.Module):
    """Lightweight TimeMixer-style multi-scale temporal mixing network."""

    def __init__(self, pred_len: int, scales: list[int] | tuple[int, ...] = (1, 2, 4), hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.scales = [max(1, int(scale)) for scale in scales] or [1]
        self.blocks = nn.ModuleList([_ScaleMixerBlock(int(hidden_dim), int(num_layers), float(dropout)) for _ in self.scales])
        self.fusion = nn.Sequential(nn.LazyLinear(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), self.pred_len))

    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1:
            return x
        x_bc_l = x.permute(0, 2, 1)
        pooled = F.avg_pool1d(x_bc_l, kernel_size=scale, stride=scale, ceil_mode=True)
        return pooled.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        scale_features = []
        for scale, block in zip(self.scales, self.blocks):
            xs = self._downsample(x, scale)
            per_channel_series = xs.permute(0, 2, 1).contiguous()
            scale_features.append(block(per_channel_series))
        fused = torch.cat(scale_features, dim=-1)
        pred = self.fusion(fused)  # [B,C,P]
        return pred.permute(0, 2, 1).contiguous()


class TimeMixerModel(nn.Module):
    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(_cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(_cfg(self.config, "pred_len", 96))
        self.epochs = int(_cfg(self.config, "epochs", 10))
        self.learning_rate = float(_cfg(self.config, "learning_rate", 1e-3))
        self.weight_decay = float(_cfg(self.config, "weight_decay", 0.0))
        self.grad_clip = float(_cfg(self.config, "grad_clip", 1.0))
        self.net = TimeMixerNet(
            pred_len=self.pred_len,
            scales=list(_cfg(self.config, "scales", [1, 2, 4])),
            hidden_dim=int(_cfg(self.config, "hidden_dim", 128)),
            num_layers=int(_cfg(self.config, "num_layers", 2)),
            dropout=float(_cfg(self.config, "dropout", 0.1)),
        ).to(self.device)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
        return _restore_shape(self.net(x), x, y_reference)

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
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
            print(_format_progress("TimeMixer", epoch, self.epochs, row), flush=True)
            if val_loader is not None and control.stopped_early:
                print(f"[TimeMixer] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
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
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            loss = self.criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
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
                loss = self.criterion(self.forward(x, y), y)
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

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "TimeMixerModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload["state_dict"])
        return self


__all__ = ["TimeMixerNet", "TimeMixerModel"]
