"""CARE: Anchor, Expert, and validation-only Reconciliation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from layers import (
    ExpertNetwork,
    reset_expert_training_rng,
    rng_isolated_module,
    select_reconciliation_profile,
)

from .care_anchor import CAREAnchor
from .timer import cfg, serializable_config, unpack_xy


class CARE(CAREAnchor):
    """Paper-aligned Anchor, independent LSTM Expert, and Stage-III reconciliation."""

    model_label = "CARE"
    final_prediction_phase = "reconciled"
    forecast_parameterization = "anchor_plus_rho_times_expert_minus_anchor"
    reconciliation_enabled = True

    def __init__(self, config: Any | None = None):
        super().__init__(config)
        expert_seed = int(cfg(self.config, "seed", 2024)) + int(
            cfg(self.config, "expert_seed_offset", 2701)
        )
        self.expert_training_seed = expert_seed
        self.expert = rng_isolated_module(
            lambda: ExpertNetwork(
                pred_len=self.pred_len,
                hidden_size=int(cfg(self.config, "expert_hidden_size", 128)),
                num_layers=int(cfg(self.config, "expert_num_layers", 2)),
                dropout=float(cfg(self.config, "expert_dropout", 0.10)),
                target_index=0,
            ),
            expert_seed,
            self.device,
        )
        self.expert_learning_rate = float(cfg(self.config, "expert_learning_rate", 1.0e-3))
        self.expert_epochs = min(
            max(1, self.epochs),
            max(1, int(cfg(self.config, "expert_max_epochs", 20))),
            20,
        )
        self.expert_patience = max(1, int(cfg(self.config, "expert_early_stopping_patience", 5)))
        self.expert_min_delta = max(0.0, float(cfg(self.config, "expert_early_stopping_min_delta", 1.0e-4)))
        self.reconciliation_mae_tolerance = float(cfg(self.config, "reconciliation_mae_tolerance", 0.025))
        self.reconciliation_group_count = max(1, int(cfg(self.config, "reconciliation_group_count", 4)))
        self.reconciliation_safety_blocks = max(1, int(cfg(self.config, "reconciliation_safety_blocks", 8)))
        self.reconciliation_safety_z = max(0.0, float(cfg(self.config, "reconciliation_safety_z", 1.645)))
        self.reconciliation_grid_size = max(2, int(cfg(self.config, "reconciliation_grid_size", 101)))
        if not 0.0 <= self.reconciliation_mae_tolerance <= 1.0:
            raise ValueError("reconciliation_mae_tolerance must be in [0, 1]")
        self.register_buffer("reconciliation_profile", torch.zeros(self.pred_len, dtype=torch.float32))
        self._prediction_phase = "anchor"
        self.reconciliation_diagnostics: dict[str, Any] | None = None

    def _expert_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (5): the Expert consumes historical x and no Anchor-side input."""
        return self.expert(x)

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        del y_reference
        x = torch.as_tensor(x, device=self.device).float()
        anchor_output = self._forward_dict(x)
        anchor = anchor_output["forecast"]
        if self._prediction_phase == "anchor":
            return anchor.unsqueeze(-1)
        expert_forecast = self._expert_prediction(x)
        if self._prediction_phase == "equal_fusion":
            return (0.5 * anchor + 0.5 * expert_forecast).unsqueeze(-1)
        profile = self.reconciliation_profile.to(device=anchor.device, dtype=anchor.dtype).unsqueeze(0)
        return (anchor + profile * (expert_forecast - anchor)).unsqueeze(-1)

    @torch.inference_mode()
    def _collect_branches(self, loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        self.expert.eval()
        truths: list[torch.Tensor] = []
        anchor_predictions: list[torch.Tensor] = []
        expert_predictions: list[torch.Tensor] = []
        for batch in loader:
            x, y = unpack_xy(batch)
            x = x.to(self.device).float()
            target = y.to(self.device).float().squeeze(-1)
            anchor = self._forward_dict(x)["forecast"]
            expert = self._expert_prediction(x)
            truths.append(target.cpu())
            anchor_predictions.append(anchor.cpu())
            expert_predictions.append(expert.cpu())
        if not truths:
            raise ValueError("CARE reconciliation loader produced no batches")
        return tuple(torch.cat(values, dim=0).numpy() for values in (truths, anchor_predictions, expert_predictions))

    @torch.inference_mode()
    def _evaluate_expert(self, loader: Any) -> tuple[float, float]:
        self.expert.eval()
        squared_error = 0.0
        absolute_error = 0.0
        count = 0
        for batch in loader:
            x, y = unpack_xy(batch)
            x = x.to(self.device).float()
            target = y.to(self.device).float().squeeze(-1)
            error = self._expert_prediction(x) - target
            if not torch.isfinite(error).all():
                raise FloatingPointError("Stage-II Expert validation produced NaN or Inf")
            squared_error += float(error.square().sum().cpu())
            absolute_error += float(error.abs().sum().cpu())
            count += int(error.numel())
        if count == 0:
            raise ValueError("Stage-II Expert validation loader produced no target elements")
        return squared_error / count, absolute_error / count

    def _fit_expert(self, train_loader: Any, val_loader: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.expert.parameters():
            parameter.requires_grad_(True)
        reset_expert_training_rng(self.expert_training_seed, self.device)
        parameters = [parameter for parameter in self.expert.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.expert_learning_rate, weight_decay=self.weight_decay)
        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] = {}
        patience_reference = float("inf")
        bad_epochs = 0
        stopped_early = False
        early_stopping_enabled = bool(cfg(self.config, "early_stopping", True))

        for epoch in range(1, self.expert_epochs + 1):
            self.net.eval()
            self.expert.train()
            train_squared_error = 0.0
            count = 0
            for batch in train_loader:
                x, y = unpack_xy(batch)
                x = x.to(self.device).float()
                target = y.to(self.device).float().squeeze(-1)
                prediction = self._expert_prediction(x)
                mse = F.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
                optimizer.step()
                batch_size = int(target.shape[0])
                count += batch_size
                train_squared_error += float(mse.detach().cpu()) * batch_size

            if count == 0:
                raise ValueError("Stage-II Expert training loader produced no batches")
            val_mse, val_mae = self._evaluate_expert(val_loader)
            row: dict[str, Any] = {
                "stage": "expert",
                "epoch": float(epoch),
                "train_loss": train_squared_error / count,
                "train_mse": train_squared_error / count,
                "val_loss": val_mse,
                "val_mae": val_mae,
            }
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
            row.update(
                {
                    "is_best": bool(is_best),
                    "best_epoch": int(best_epoch),
                    "best_val_mse": float(best_metric),
                    "epochs_without_improvement": int(bad_epochs),
                }
            )
            history.append(row)
            print(
                f"[{self.model_label}:expert] epoch={epoch}/{self.expert_epochs} "
                f"train_mse={row['train_mse']:.6g} val_mse={val_mse:.6g} "
                f"val_mae={val_mae:.6g} best_epoch={best_epoch} patience={bad_epochs}",
                flush=True,
            )
            stopped_early = early_stopping_enabled and bad_epochs >= self.expert_patience
            if stopped_early:
                print(f"[{self.model_label}:expert] early stopping at epoch={epoch}; best_epoch={best_epoch}", flush=True)
                break

        if not best_state:
            raise RuntimeError("Stage-II Expert did not produce a finite validation checkpoint")
        self.expert.load_state_dict(best_state, strict=True)
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        return history, {
            "monitor": "val_mse",
            "best_epoch": int(best_epoch),
            "best_metric": float(best_metric),
            "epochs_ran": len(history),
            "stopped_early": bool(stopped_early),
            "early_stopping_enabled": bool(early_stopping_enabled),
            "patience": int(self.expert_patience),
            "min_delta": float(self.expert_min_delta),
            "restored_best_parameters": True,
        }

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        if val_loader is None:
            raise ValueError("CARE requires validation data for checkpoints and reconciliation")
        self._prediction_phase = "anchor"
        self.reconciliation_profile.zero_()
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)
        anchor_result = super().fit(train_loader, val_loader)
        anchor_history = [dict(row, stage="anchor") for row in anchor_result.get("history", [])]
        expert_history, expert_control = self._fit_expert(train_loader, val_loader)
        truth, anchor, expert_forecast = self._collect_branches(val_loader)
        selection = (
            select_reconciliation_profile(
                truth,
                anchor,
                expert_forecast,
                group_count=self.reconciliation_group_count,
                mae_tolerance=self.reconciliation_mae_tolerance,
                blocks=self.reconciliation_safety_blocks,
                z_value=self.reconciliation_safety_z,
                grid_size=self.reconciliation_grid_size,
            )
            if self.reconciliation_enabled
            else None
        )
        self.reconciliation_profile.zero_()
        if selection is not None:
            self.reconciliation_profile.copy_(torch.tensor(selection.profile, dtype=self.reconciliation_profile.dtype))
        selection_payload = None if selection is None else selection.to_dict()
        self.reconciliation_diagnostics = selection_payload
        self._prediction_phase = self.final_prediction_phase
        if selection is None:
            print(
                f"[{self.model_label}:fusion] fixed Anchor/Expert weights=(0.5, 0.5); "
                "no fusion parameters trained",
                flush=True,
            )
        else:
            print(
                f"[{self.model_label}:reconcile] path={selection.path} gamma={selection.gamma:.4f} "
                f"profile_mean={np.mean(selection.profile):.4f} upper={selection.upper_bound:.6g} "
                f"anchor_MSE={selection.anchor_validation_mse:.6g} selected_MSE={selection.validation_mse:.6g}",
                flush=True,
            )
        result = copy.deepcopy(anchor_result)
        result.update(
            {
                "history": anchor_history + expert_history,
                "training_control": {
                    "monitor": "anchor_MAE_then_expert_MSE_then_horizon_profile_safety",
                    "anchor": anchor_result.get("training_control", {}),
                    "expert": expert_control,
                    "selection": selection_payload,
                },
                "forecast_parameterization": self.forecast_parameterization,
                "optimizer": "disjoint_Adam_anchor_then_AdamW_expert",
                "stage_ii": {
                    "architecture": "two_layer_history_only_lstm",
                    "input": "historical_x_only",
                    "loss": "MSE",
                    "optimizer": "AdamW",
                    "learning_rate": self.expert_learning_rate,
                    "max_epochs": self.expert_epochs,
                    "best_checkpoint_monitor": "val_mse",
                    "anchor_frozen_during_training": True,
                    "training_rng_seed": int(self.expert_training_seed),
                    "training_rng_reset_before_fit": True,
                },
                "reconciliation": {
                    "rng_isolation": True,
                    "expert_seed_offset": int(cfg(self.config, "expert_seed_offset", 2701)),
                    "selection": selection_payload,
                    "selection_data": "validation_only",
                    "test_labels_used_for_selection": False,
                },
            }
        )
        if self.final_prediction_phase == "equal_fusion":
            result["fixed_equal_fusion"] = {
                "formula": "0.5 * anchor + 0.5 * expert",
                "anchor_weight": 0.5,
                "expert_weight": 0.5,
                "trainable_fusion_parameters": 0,
                "weight_selection_data": None,
                "test_labels_used_for_weights": False,
            }
        return result

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        self.eval()
        self._prediction_phase = self.final_prediction_phase
        predictions: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in test_loader:
                x = batch[0].to(self.device).float()
                anchor = self._forward_dict(x)["forecast"]
                expert = self._expert_prediction(x)
                if self._prediction_phase == "equal_fusion":
                    prediction = 0.5 * anchor + 0.5 * expert
                else:
                    profile = self.reconciliation_profile.to(anchor).unsqueeze(0)
                    prediction = anchor + profile * (expert - anchor)
                predictions.append(prediction.unsqueeze(-1).cpu())
        output = torch.cat(predictions, dim=0)
        if self._prediction_phase == "equal_fusion":
            self.last_prediction_diagnostics = {
                "formula": "0.5 * anchor + 0.5 * expert",
                "anchor_weight": 0.5,
                "expert_weight": 0.5,
                "trainable_fusion_parameters": 0,
                "test_labels_used_for_weights": False,
            }
        else:
            self.last_prediction_diagnostics = {
                "horizon_profile": self.reconciliation_profile.detach().cpu().tolist(),
                "profile_mean": float(self.reconciliation_profile.mean().detach().cpu()),
                "formula": "anchor + rho * (expert - anchor)",
                "validation_horizon_profile": self.reconciliation_diagnostics or {},
            }
        return output.numpy() if return_numpy else output

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        filtered = {key: value for key, value in self.state_dict().items() if ".timer_model." not in key}
        torch.save(
            {
                "state_dict_without_timer": filtered,
                "config": serializable_config(self.config),
                "timer_contract": self.contract,
                "anchor_metadata": self.net.metadata,
                "tail_statistics": self.tail_statistics,
                "reconciliation_diagnostics": self.reconciliation_diagnostics,
                "forecast_parameterization": self.forecast_parameterization,
            },
            path,
        )

    def load(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "CARE":
        payload = torch.load(path, map_location=map_location or self.device)
        state = payload["state_dict_without_timer"]
        incompatible = self.load_state_dict(state, strict=False)
        invalid_missing = [key for key in incompatible.missing_keys if ".timer_model." not in key]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Invalid CARE checkpoint: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        self.tail_statistics = payload.get("tail_statistics")
        diagnostics = payload.get("reconciliation_diagnostics")
        self.reconciliation_diagnostics = None if diagnostics is None else dict(diagnostics)
        self._prediction_phase = self.final_prediction_phase
        for module in (self.net, self.expert):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return self


class CAREEqualFusion(CARE):
    """Stage I+II with the fixed, parameter-free 0.5/0.5 prediction rule."""

    model_label = "CARE (Stage I+II equal fusion)"
    final_prediction_phase = "equal_fusion"
    forecast_parameterization = "fixed_0.5_anchor_plus_0.5_expert"

    reconciliation_enabled = False


__all__ = [
    "CARE",
    "CAREEqualFusion",
]
