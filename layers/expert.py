"""Paper-aligned Stage-II Expert and RNG-isolated construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
from torch import nn


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def rng_isolated_module(
    factory: Callable[[], ModuleT],
    seed: int,
    device: torch.device | str | None = None,
) -> ModuleT:
    """Build and place a module without advancing caller CPU/CUDA RNG streams."""

    target = None if device is None else torch.device(device)
    devices: list[int] = []
    if target is not None and target.type == "cuda":
        devices = [torch.cuda.current_device() if target.index is None else int(target.index)]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        module = factory()
        return module if target is None else module.to(target)


def reset_expert_training_rng(seed: int, device: torch.device | str | None = None) -> None:
    """Start Stage-II optimization from a branch-local deterministic RNG state.

    Stage II is defined as an independent history-only Expert.  Resetting here
    prevents an earlier Anchor fit from changing Expert shuffling/dropout while
    preserving the controller's target/seed-level reproducibility contract.
    """

    torch.manual_seed(int(seed))
    target = None if device is None else torch.device(device)
    if torch.cuda.is_available() and (target is None or target.type == "cuda"):
        torch.cuda.manual_seed_all(int(seed))


class ExpertNetwork(nn.Module):
    """Two-layer history-only LSTM Expert from CARE Eqs. (5)-(6)."""

    def __init__(
        self,
        *,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        target_index: int = 0,
    ) -> None:
        super().__init__()
        if pred_len <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("invalid Stage-II Expert settings")
        if int(hidden_size) != 128:
            raise ValueError("CARE fixes the Stage-II LSTM hidden dimension at 128")
        if int(num_layers) != 2:
            raise ValueError("CARE requires exactly two Stage-II LSTM layers")
        self.pred_len = int(pred_len)
        self.target_index = int(target_index)
        lstm_dropout = 0.0 if int(num_layers) <= 1 else float(dropout)
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.direct_head = nn.Linear(int(hidden_size), self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or not 0 <= self.target_index < x.shape[-1]:
            raise ValueError("Stage-II Expert expects [B,L,F] with a valid target index")
        series = x[:, :, self.target_index].unsqueeze(-1)
        _, (hidden, _) = self.encoder(series)
        return self.direct_head(hidden[-1])
