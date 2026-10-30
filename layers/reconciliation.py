"""Low-dimensional horizon-profile functional reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReconciliationSelection:
    path: str
    gamma: float
    profile: tuple[float, ...]
    group_alphas: tuple[float, ...]
    validation_mae: float
    validation_mse: float
    anchor_validation_mae: float
    anchor_validation_mse: float
    expert_validation_mae: float
    expert_validation_mse: float
    mae_tolerance: float
    upper_bound: float
    paired_mean_change: float
    block_standard_error: float
    block_count: int
    candidate_count: int
    feasible_count: int
    group_count: int
    z_value: float

    @property
    def identity(self) -> bool:
        return not any(abs(value) > 0.0 for value in self.profile)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "identity": self.identity}


def _optimal_alpha(residual: np.ndarray, direction: np.ndarray) -> float:
    curvature = float(np.mean(np.square(direction)))
    if curvature <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(-np.mean(residual * direction) / curvature, 0.0, 1.0))


def _block_bound(candidate: np.ndarray, baseline: np.ndarray, blocks: int, z_value: float) -> dict[str, float | int]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    count = min(max(1, int(blocks)), len(difference))
    means = np.asarray([part.mean() for part in np.array_split(difference, count) if len(part)])
    standard_error = 0.0 if len(means) <= 1 else float(means.std(ddof=1) / np.sqrt(len(means)))
    mean = float(means.mean())
    return {
        "upper_bound": mean + float(z_value) * standard_error,
        "paired_mean_change": mean,
        "block_standard_error": standard_error,
        "block_count": int(len(means)),
    }


def select_reconciliation_profile(
    truth: np.ndarray,
    anchor: np.ndarray,
    expert: np.ndarray,
    *,
    group_count: int = 4,
    mae_tolerance: float = 0.025,
    blocks: int = 8,
    z_value: float = 1.645,
    grid_size: int = 101,
) -> ReconciliationSelection:
    """Choose the formal global/grouped profile using validation only.

    Every candidate must satisfy the blockwise MAE-safety certificate.
    """

    y, anchor_forecast, expert_forecast = (
        np.asarray(value, dtype=np.float64) for value in (truth, anchor, expert)
    )
    if y.shape != anchor_forecast.shape or y.shape != expert_forecast.shape or y.ndim != 2 or y.size == 0:
        raise ValueError("reconciliation arrays must share non-empty [N,H] shape")
    if not all(np.isfinite(value).all() for value in (y, anchor_forecast, expert_forecast)):
        raise FloatingPointError("reconciliation arrays contain NaN or Inf")
    if group_count <= 0 or not 0.0 <= mae_tolerance <= 1.0 or blocks <= 0 or z_value < 0 or grid_size < 2:
        raise ValueError("invalid horizon-profile settings")
    residual = anchor_forecast - y
    direction = expert_forecast - anchor_forecast
    horizon = y.shape[1]
    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(np.arange(horizon), min(group_count, horizon))]
    global_alpha = _optimal_alpha(residual, direction)
    global_profile = np.full(horizon, global_alpha, dtype=np.float64)
    grouped_profile = np.zeros(horizon, dtype=np.float64)
    group_alphas: list[float] = []
    for group in groups:
        alpha = _optimal_alpha(residual[:, group], direction[:, group])
        grouped_profile[group] = alpha
        group_alphas.append(alpha)
    anchor_window_mae = np.mean(np.abs(residual), axis=1)
    anchor_mae = float(anchor_window_mae.mean())
    anchor_mse = float(np.mean(np.square(residual)))
    expert_mae = float(np.mean(np.abs(expert_forecast - y)))
    expert_mse = float(np.mean(np.square(expert_forecast - y)))
    budget = anchor_mae * float(mae_tolerance)
    best_profile = np.zeros(horizon, dtype=np.float64)
    best_key = (anchor_mse, anchor_mae, 0, 0.0)
    best_path = "identity"
    best_gamma = 0.0
    best_mae = anchor_mae
    best_mse = anchor_mse
    best_certificate = {
        "upper_bound": 0.0,
        "paired_mean_change": 0.0,
        "block_standard_error": 0.0,
        "block_count": min(int(blocks), len(y)),
    }
    feasible = 1
    candidates = 1
    paths = (("global", global_profile), ("grouped", grouped_profile))
    for path_order, (path, base_profile) in enumerate(paths, 1):
        for gamma in np.linspace(0.0, 1.0, int(grid_size))[1:]:
            candidates += 1
            profile = float(gamma) * base_profile
            prediction = anchor_forecast + direction * profile[None, :]
            window_mae = np.mean(np.abs(prediction - y), axis=1)
            certificate = _block_bound(window_mae, anchor_window_mae, blocks, z_value)
            if float(certificate["upper_bound"]) > budget + 1.0e-12:
                continue
            feasible += 1
            mae = float(window_mae.mean())
            mse = float(np.mean(np.square(prediction - y)))
            key = (mse, mae, path_order, float(gamma))
            if key < best_key:
                best_key = key
                best_profile = profile.copy()
                best_path = path
                best_gamma = float(gamma)
                best_mae = mae
                best_mse = mse
                best_certificate = certificate
    return ReconciliationSelection(
        path=best_path,
        gamma=best_gamma,
        profile=tuple(float(value) for value in best_profile),
        group_alphas=tuple(float(value) for value in group_alphas),
        validation_mae=best_mae,
        validation_mse=best_mse,
        anchor_validation_mae=anchor_mae,
        anchor_validation_mse=anchor_mse,
        expert_validation_mae=expert_mae,
        expert_validation_mse=expert_mse,
        mae_tolerance=float(mae_tolerance),
        upper_bound=float(best_certificate["upper_bound"]),
        paired_mean_change=float(best_certificate["paired_mean_change"]),
        block_standard_error=float(best_certificate["block_standard_error"]),
        block_count=int(best_certificate["block_count"]),
        candidate_count=int(candidates),
        feasible_count=int(feasible),
        group_count=len(groups),
        z_value=float(z_value),
    )
