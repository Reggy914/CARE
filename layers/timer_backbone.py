from __future__ import annotations

import torch
from torch import nn


class TimerBackboneWrapper(nn.Module):
    def __init__(
        self,
        timer_model: nn.Module,
        target_index: int,
        pred_len: int,
        input_token_len: int,
        representation_dim: int,
        freeze: bool = True,
        use_hidden_states: bool = True,
    ) -> None:
        super().__init__()
        self.timer_model = timer_model
        self.target_index = int(target_index)
        self.pred_len = int(pred_len)
        self.input_token_len = int(input_token_len)
        self.representation_dim = int(representation_dim)
        self.use_hidden_states = bool(use_hidden_states)
        self.frozen = bool(freeze)
        if self.frozen:
            self.timer_model.eval()
            for parameter in self.timer_model.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.timer_model.eval()
        return self

    def _target_series(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            series = x[:, :, self.target_index]
        elif x.ndim == 2:
            series = x
        else:
            raise ValueError(f"Timer input must be [B, L, F] or [B, L], got {tuple(x.shape)}")
        usable = (series.shape[1] // self.input_token_len) * self.input_token_len
        if usable < self.input_token_len:
            raise ValueError(f"Input length must be at least {self.input_token_len} for Timer")
        return series[:, -usable:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward_timer(self.timer_model, self._target_series(x), self.pred_len)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hidden_states:
            raise RuntimeError("Paper-aligned Stage I requires Timer hidden-state features")
        series = self._target_series(x)
        outputs = self.timer_model(
            input_ids=series,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            max_output_length=self.pred_len,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "The real Timer model did not expose hidden states; "
                "forecast-projection fallback is forbidden"
            )
        return hidden_states[-1][:, -1, :]


def forward_timer(model: nn.Module, x: torch.Tensor, pred_len: int = 96) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Timer direct input must be [B, L], got {tuple(x.shape)}")
    outputs = model(input_ids=x, return_dict=True, max_output_length=pred_len)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Timer model did not return prediction logits")
    if logits.ndim != 2 or logits.shape[1] != pred_len:
        raise RuntimeError(f"Timer logits must have shape [B, {pred_len}], got {tuple(logits.shape)}")
    return logits


def extract_timer_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "extract_features"):
        return model.extract_features(x)
    raise TypeError("Timer feature extraction requires TimerBackboneWrapper")
