"""Standalone Stage-II Expert ablation with no Stage-I dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers import ExpertNetwork, reset_expert_training_rng, rng_isolated_module

from .timer import cfg, serializable_config, unpack_xy


class CAREExpert(nn.Module):
    """CARE Stage II trained alone from history with the original MSE protocol."""

    model_label = "Expert only"
    forecast_parameterization = "stage_ii_history_lstm_direct_forecast"

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.config = config or {}
        self.device = torch.device(
            cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        configured_epochs = int(cfg(self.config, "epochs", 20))
        expert_max_epochs = int(cfg(self.config, "expert_max_epochs", 20))
        self.epochs = min(max(1, configured_epochs), max(1, expert_max_epochs), 20)
        self.weight_decay = float(cfg(self.config, "weight_decay", 0.0))
        self.grad_clip = float(cfg(self.config, "grad_clip", 1.0))
        self.expert_learning_rate = float(
            cfg(self.config, "expert_learning_rate", cfg(self.config, "learning_rate", 1.0e-3))
        )
        self.expert_patience = max(1, int(cfg(self.config, "expert_early_stopping_patience", 5)))
        self.expert_min_delta = max(0.0, float(cfg(self.config, "expert_early_stopping_min_delta", 1.0e-4)))
        branch_seed = int(cfg(self.config, "seed", 2024)) + int(
            cfg(self.config, "expert_seed_offset", 2701)
        )
        self.expert_training_seed = branch_seed
        self.expert = rng_isolated_module(
            lambda: ExpertNetwork(
                pred_len=self.pred_len,
                hidden_size=int(cfg(self.config, "expert_hidden_size", 128)),
                num_layers=int(cfg(self.config, "expert_num_layers", 2)),
                dropout=float(cfg(self.config, "expert_dropout", 0.10)),
                target_index=0,
            ),
            branch_seed,
            self.device,
        )

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        del y_reference
        x = torch.as_tensor(x, device=self.device).float()
        return self.expert(x).unsqueeze(-1)

    @torch.inference_mode()
    def _validation_metrics(self, loader: Any) -> tuple[float, float]:
        self.expert.eval()
        squared_error = 0.0
        absolute_error = 0.0
        count = 0
        for batch in loader:
            x, y = unpack_xy(batch)
            x = x.to(self.device).float()
            target = y.to(self.device).float().squeeze(-1)
            error = self.expert(x) - target
            if not torch.isfinite(error).all():
                raise FloatingPointError("Expert-only validation produced NaN or Inf")
            squared_error += float(error.square().sum().cpu())
            absolute_error += float(error.abs().sum().cpu())
            count += int(error.numel())
        if count == 0:
            raise ValueError("Expert-only validation loader produced no target elements")
        return squared_error / count, absolute_error / count

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        if val_loader is None:
            raise ValueError("Expert only requires validation data for checkpoint selection")
        reset_expert_training_rng(self.expert_training_seed, self.device)
        parameters = list(self.expert.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.expert_learning_rate,
            weight_decay=self.weight_decay,
        )
        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] = {}
        patience_reference = float("inf")
        bad_epochs = 0
        stopped_early = False
        early_stopping_enabled = bool(cfg(self.config, "early_stopping", True))

        for epoch in range(1, self.epochs + 1):
            self.expert.train()
            train_squared_error = 0.0
            count = 0
            for batch in train_loader:
                x, y = unpack_xy(batch)
                x = x.to(self.device).float()
                target = y.to(self.device).float().squeeze(-1)
                prediction = self.expert(x)
                mse = F.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
                optimizer.step()
                batch_size = int(target.shape[0])
                count += batch_size
                train_squared_error += float(mse.detach().cpu()) * batch_size

            if count == 0:
                raise ValueError("Expert-only training loader produced no batches")
            val_mse, val_mae = self._validation_metrics(val_loader)
            is_best = val_mse < best_metric
            if is_best:
                best_metric = val_mse
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.expert.state_dict().items()
                }
            if val_mse < patience_reference - self.expert_min_delta:
                patience_reference = val_mse
                bad_epochs = 0
            else:
                bad_epochs += 1
            row = {
                "stage": "expert",
                "epoch": float(epoch),
                "train_loss": train_squared_error / count,
                "train_mse": train_squared_error / count,
                "val_loss": val_mse,
                "val_mae": val_mae,
                "is_best": bool(is_best),
                "best_epoch": int(best_epoch),
                "best_val_mse": float(best_metric),
                "epochs_without_improvement": int(bad_epochs),
            }
            history.append(row)
            print(
                f"[{self.model_label}] epoch={epoch}/{self.epochs} "
                f"train_mse={row['train_mse']:.6g} val_mse={val_mse:.6g} "
                f"val_mae={val_mae:.6g} best_epoch={best_epoch} patience={bad_epochs}",
                flush=True,
            )
            stopped_early = early_stopping_enabled and bad_epochs >= self.expert_patience
            if stopped_early:
                print(
                    f"[{self.model_label}] early stopping at epoch={epoch}; best_epoch={best_epoch}",
                    flush=True,
                )
                break

        if not best_state:
            raise RuntimeError("Expert-only training did not produce a finite validation checkpoint")
        self.expert.load_state_dict(best_state, strict=True)
        return {
            "history": history,
            "training_control": {
                "monitor": "val_mse",
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "epochs_ran": len(history),
                "stopped_early": bool(stopped_early),
                "early_stopping_enabled": bool(early_stopping_enabled),
                "patience": int(self.expert_patience),
                "min_delta": float(self.expert_min_delta),
                "restored_best_parameters": True,
            },
            "forecast_parameterization": self.forecast_parameterization,
            "optimizer": "AdamW",
            "learning_rate": self.expert_learning_rate,
            "loss": "MSE",
            "stage_i_loaded": False,
            "stage_iii_enabled": False,
            "training_rng": {
                "seed": int(self.expert_training_seed),
                "reset_before_fit": True,
                "anchor_history_independent": True,
            },
        }

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        self.expert.eval()
        predictions: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in test_loader:
                x = torch.as_tensor(batch[0], device=self.device).float()
                predictions.append(self.expert(x).unsqueeze(-1).cpu())
        output = torch.cat(predictions, dim=0)
        return output.numpy() if return_numpy else output

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.expert.state_dict(),
                "config": serializable_config(self.config),
                "forecast_parameterization": self.forecast_parameterization,
                "stage_i_loaded": False,
                "stage_iii_enabled": False,
            },
            path,
        )

    def load(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "CAREExpert":
        payload = torch.load(path, map_location=map_location or self.device)
        self.expert.load_state_dict(payload["state_dict"], strict=True)
        return self


__all__ = ["CAREExpert"]
