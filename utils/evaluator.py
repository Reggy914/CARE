from __future__ import annotations

import math
from typing import Any

import numpy as np


def _forecast_2d(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, H] or [N, H, 1], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def build_evaluation_region_masks(
    y_true: Any,
    last_observed: Any,
    statistics: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build fixed test regions from train-only thresholds and test truth.

    The thresholds and center must be fitted on chronological training targets.
    Test truth is used only to assign evaluation regions, never to tune a model.
    """
    truth = _forecast_2d("y_true", y_true)
    previous = np.asarray(last_observed, dtype=np.float64).reshape(-1)
    if previous.shape[0] != truth.shape[0]:
        raise ValueError(
            f"last_observed length {previous.shape[0]} does not match forecast rows {truth.shape[0]}"
        )
    if not np.isfinite(previous).all():
        raise ValueError("last_observed contains NaN or Inf")
    if str(statistics.get("fit_scope")) != "train_only":
        raise ValueError("evaluation region thresholds must be fitted on train_only data")

    q_delta = float(statistics["q_delta_raw"])
    q_amplitude = float(statistics["q_amplitude_raw"])
    center = float(statistics["center_raw"])
    if not all(np.isfinite([q_delta, q_amplitude, center])) or q_delta < 0.0 or q_amplitude < 0.0:
        raise ValueError("invalid evaluation region statistics")

    differences = np.empty_like(truth)
    differences[:, 0] = truth[:, 0] - previous
    differences[:, 1:] = truth[:, 1:] - truth[:, :-1]
    boundary = np.abs(differences) >= q_delta
    high_amplitude = np.abs(truth - center) >= q_amplitude
    tail = boundary | high_amplitude
    return {
        "tail": tail,
        "boundary": boundary,
        "high_amplitude": high_amplitude,
    }


def _region_error_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != truth.shape:
        raise ValueError(f"{prefix} mask shape {selected.shape} does not match truth {truth.shape}")
    count = int(selected.sum())
    if count == 0:
        return {f"{prefix}_MAE": float("nan"), f"{prefix}_MSE": float("nan"), f"{prefix}_RMSE": float("nan")}
    error = pred[selected] - truth[selected]
    mse = float(np.mean(error ** 2))
    return {
        f"{prefix}_MAE": float(np.mean(np.abs(error))),
        f"{prefix}_MSE": mse,
        f"{prefix}_RMSE": float(math.sqrt(mse)),
    }


def compute_diagnostic_metrics(
    y_true: Any,
    y_pred: Any,
    region_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return tail-risk and boundary metrics on shared, data-defined regions."""
    truth = _forecast_2d("y_true", y_true)
    pred = _forecast_2d("y_pred", y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"metric shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    absolute_error = np.abs(pred - truth).reshape(-1)
    result = {
        "P90_AE": float(np.quantile(absolute_error, 0.90)),
        "P95_AE": float(np.quantile(absolute_error, 0.95)),
        "P99_AE": float(np.quantile(absolute_error, 0.99)),
        "Max_AE": float(np.max(absolute_error)),
    }
    for prefix, key in (
        ("Tail", "tail"),
        ("HighAmplitude", "high_amplitude"),
        ("Boundary", "boundary"),
    ):
        if key not in region_masks:
            raise ValueError(f"region_masks is missing {key!r}")
        result.update(_region_error_metrics(truth, pred, region_masks[key], prefix))
    return result


def compute_metrics(y_true: Any, y_pred: Any, epsilon: float = 1e-8) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"metric shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    diff = pred - truth
    abs_error = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    truth_flat = truth.reshape(-1)
    diff_flat = diff.reshape(-1)
    denom = float(np.sum((truth_flat - np.mean(truth_flat)) ** 2))
    r2 = 0.0 if denom <= epsilon else float(1.0 - np.sum(diff_flat ** 2) / (denom + epsilon))
    return {
        "MAE": float(np.mean(abs_error)),
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAPE": float(np.mean(abs_error / (np.abs(truth) + epsilon)) * 100.0),
        "R2": r2,
    }


__all__ = ["build_evaluation_region_masks", "compute_diagnostic_metrics", "compute_metrics"]
