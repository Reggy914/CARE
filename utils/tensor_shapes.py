"""Small tensor-shape helpers shared by trainable forecasting baselines."""

from __future__ import annotations

import torch


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim == 3:
        return x
    raise ValueError(f"expected input shape [B,L] or [B,L,C], got {tuple(x.shape)}")


def restore_shape(
    prediction: torch.Tensor,
    x: torch.Tensor,
    reference_y: torch.Tensor | None,
) -> torch.Tensor:
    if reference_y is not None:
        if reference_y.ndim == 2:
            return prediction[..., 0]
        if reference_y.ndim == 3:
            return prediction.reshape(reference_y.shape)
    if x.ndim == 2:
        return prediction[..., 0]
    return prediction


__all__ = ["ensure_3d", "restore_shape"]
