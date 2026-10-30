from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from utils.tensor_shapes import ensure_3d as _ensure_3d, restore_shape as _restore_shape
from .normal_head import _cfg, _format_progress, _serializable_config, _unpack_batch
from utils.training_control import ValidationEarlyStopping, evaluate_forecast


class ITransformerNet(nn.Module):
    """Inverted Transformer baseline treating variables as tokens."""

    def __init__(self, pred_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.value_embedding = nn.LazyLinear(int(d_model))
        layer = nn.TransformerEncoderLayer(d_model=int(d_model), nhead=int(n_heads), dim_feedforward=int(d_ff), dropout=float(dropout), batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = nn.Linear(int(d_model), self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        variable_tokens = x.permute(0, 2, 1).contiguous()  # [B,C,L]
        encoded = self.encoder(self.value_embedding(variable_tokens))
        pred = self.head(encoded)  # [B,C,P]
        return pred.permute(0, 2, 1).contiguous()


class ITransformerModel(nn.Module):
    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(_cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(_cfg(self.config, "pred_len", 96))
        self.epochs = int(_cfg(self.config, "epochs", 10))
        self.learning_rate = float(_cfg(self.config, "learning_rate", 1e-3))
        self.weight_decay = float(_cfg(self.config, "weight_decay", 0.0))
        self.grad_clip = float(_cfg(self.config, "grad_clip", 1.0))
        self.net = ITransformerNet(
            pred_len=self.pred_len,
            d_model=int(_cfg(self.config, "d_model", 128)),
            n_heads=int(_cfg(self.config, "n_heads", 4)),
            num_layers=int(_cfg(self.config, "num_layers", 2)),
            d_ff=int(_cfg(self.config, "d_ff", 256)),
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
            print(_format_progress("iTransformer", epoch, self.epochs, row), flush=True)
            if val_loader is not None and control.stopped_early:
                print(f"[iTransformer] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
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

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "ITransformerModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload["state_dict"])
        return self


__all__ = ["ITransformerNet", "ITransformerModel"]
