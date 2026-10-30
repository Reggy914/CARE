from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


from .timer import cfg, load_shared_timer, serializable_config, timer_contract, unpack_xy
from utils.training_control import ValidationEarlyStopping, evaluate_forecast


def _model_config(config: Any) -> dict[str, Any]:
    return {
        "hidden_dim": int(cfg(config, "hidden_dim", 256)),
        "dropout": float(cfg(config, "dropout", 0.10)),
        "fusion_hidden_dim": int(cfg(config, "fusion_hidden_dim", 256)),
        "tcn_layers": int(cfg(config, "tcn_layers", 3)),
        "tcn_kernel_size": int(cfg(config, "tcn_kernel_size", 3)),
        "mask_timer_representation": bool(cfg(config, "mask_timer_representation", False)),
        "mask_regime_representation": bool(cfg(config, "mask_regime_representation", False)),
    }


class AnchorRuntime(nn.Module):
    """Build the CARE Stage-I Anchor around the real frozen Timer."""

    def __init__(self, config: Any | None = None):
        super().__init__()
        from layers import TimerBackboneWrapper, build_anchor

        self.config = config or {}
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        self.device = torch.device(cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_dir = cfg(self.config, "timer_model_dir", None)
        loaded_timer = load_shared_timer(self.device, self.pred_len, model_dir)
        inner_timer = getattr(loaded_timer, "timer_model", loaded_timer)
        hidden_size = int(getattr(getattr(inner_timer, "config", None), "hidden_size", 1024))
        self.timer_backbone = TimerBackboneWrapper(
            timer_model=inner_timer,
            target_index=0,
            pred_len=self.pred_len,
            input_token_len=int(cfg(self.config, "timer_input_token_len", 96)),
            representation_dim=hidden_size,
            freeze=True,
            use_hidden_states=True,
        )
        self.contract = timer_contract(self.pred_len, model_dir)
        result = build_anchor(
            input_dim=1,
            pred_len=self.pred_len,
            target_index=0,
            model_config=_model_config(self.config),
            timer_backbone=self.timer_backbone,
            timer_metadata=self.contract,
        )
        self.forecaster = result.model.to(self.device)
        self.metadata = result.metadata

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forecaster(x.float())


class CAREAnchor(nn.Module):
    """CARE Stage-I Anchor trained only with the Huber objective."""

    model_label = "CARE Stage I Anchor"

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        self.epochs = int(cfg(self.config, "epochs", 10))
        self.learning_rate = float(cfg(self.config, "learning_rate", 1.0e-3))
        self.weight_decay = float(cfg(self.config, "weight_decay", 0.0))
        self.grad_clip = float(cfg(self.config, "grad_clip", 1.0))
        self.huber_delta = float(cfg(self.config, "huber_delta", 1.0))
        self.net = AnchorRuntime(self.config).to(self.device)
        self.criterion = nn.HuberLoss(delta=self.huber_delta)
        # Kept as an empty serialization field because the unchanged Stage-II/III
        # checkpoint wrapper reads this attribute. No Stage-I computation uses it.
        self.tail_statistics: None = None
        self.contract = self.net.contract

    def _forward_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(x)

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        del y_reference
        x = torch.as_tensor(x, device=self.device).float()
        output = self._forward_dict(x)
        return output["forecast"].unsqueeze(-1)

    def _trainable_parameters(self) -> list[nn.Parameter]:
        timer_parameters = list(self.net.timer_backbone.timer_model.parameters())
        if any(parameter.requires_grad for parameter in timer_parameters):
            raise RuntimeError("Frozen Timer parameters unexpectedly became trainable")
        parameters = [parameter for parameter in self.net.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("Stage I has no trainable target-domain parameters")
        return parameters

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        params = self._trainable_parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        history: list[dict[str, float]] = []
        control = ValidationEarlyStopping(self, self.config)
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0
            total_absolute_error = 0.0
            total_squared_error = 0.0
            point_count = 0
            sample_count = 0
            for batch in train_loader:
                x, y = unpack_xy(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze(-1)
                forecast = self._forward_dict(x)["forecast"]
                loss = self.criterion(forecast, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                optimizer.step()

                batch_size = int(y.shape[0])
                error = forecast.detach() - y
                total_loss += float(loss.detach().cpu()) * batch_size
                total_absolute_error += float(error.abs().sum().cpu())
                total_squared_error += float(error.square().sum().cpu())
                point_count += int(error.numel())
                sample_count += batch_size

            row = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(sample_count, 1),
                "train_level_loss": total_loss / max(sample_count, 1),
                "train_mae": total_absolute_error / max(point_count, 1),
                "train_mse": total_squared_error / max(point_count, 1),
            }
            if val_loader is not None:
                row.update(evaluate_forecast(self, val_loader, self.device))
                is_best = control.update(row["val_mae"], epoch)
                control.annotate(row, is_best=is_best)
            history.append(row)
            print(
                f"[{self.model_label}] epoch={epoch}/{self.epochs} "
                f"train_huber={row['train_loss']:.6g} "
                f"val_loss={row.get('val_loss', float('nan')):.6g} "
                f"val_mae={row.get('val_mae', float('nan')):.6g} "
                f"best_epoch={control.best_epoch} patience={control.bad_epochs}",
                flush=True,
            )
            if val_loader is not None and control.stopped_early:
                print(
                    f"[{self.model_label}] early stopping at epoch={epoch}; "
                    f"best_epoch={control.best_epoch}",
                    flush=True,
                )
                break

        if val_loader is not None:
            control.restore_best()
        return {
            "history": history,
            "training_control": control.summary(len(history)),
            "loss": {"name": "HuberLoss", "delta": self.huber_delta},
            "forecast_parameterization": "frozen_timer_plus_tcn_concat_fusion_anchor",
            "optimizer": "Adam",
            "learning_rate": self.learning_rate,
            "timer_contract": self.contract,
            "timer_frozen": True,
            "timer_fallback_allowed": False,
            "anchor_metadata": self.net.metadata,
        }

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        self.eval()
        preds: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in test_loader:
                preds.append(self.forward(batch[0]).detach().cpu())
        out = torch.cat(preds, dim=0)
        return out.numpy() if return_numpy else out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        filtered = {
            key: value
            for key, value in self.state_dict().items()
            if ".timer_model." not in key
        }
        torch.save(
            {
                "state_dict_without_timer": filtered,
                "config": serializable_config(self.config),
                "timer_contract": self.contract,
                "anchor_metadata": self.net.metadata,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "CAREAnchor":
        payload = torch.load(path, map_location=map_location or self.device)
        incompatible = self.load_state_dict(payload["state_dict_without_timer"], strict=False)
        invalid_missing = [key for key in incompatible.missing_keys if ".timer_model." not in key]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Invalid Stage-I checkpoint: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        self.tail_statistics = None
        return self


__all__ = ["AnchorRuntime", "CAREAnchor"]
