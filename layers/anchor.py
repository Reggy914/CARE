from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .timer_backbone import extract_timer_features


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.left_padding, 0))
        y = self.conv(y)
        y = self.dropout(F.gelu(self.norm(y)))
        return y + self.residual(x)


class CausalRegimeTCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        in_channels = input_dim
        for layer_idx in range(layers):
            blocks.append(CausalConvBlock(in_channels, hidden_dim, kernel_size, dilation=2**layer_idx, dropout=dropout))
            in_channels = hidden_dim
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.blocks(y)
        return self.pool(y[:, :, -1])


class AnchorNetwork(nn.Module):
    """CARE Stage-I Anchor: frozen Timer + target TCN + concat fusion."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        target_index: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        tcn_layers: int = 3,
        tcn_kernel_size: int = 3,
        mask_timer_representation: bool = False,
        mask_regime_representation: bool = False,
        timer_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if timer_backbone is None:
            raise ValueError("Paper-aligned Stage I requires a real frozen Timer backbone")
        if not bool(getattr(timer_backbone, "frozen", False)):
            raise ValueError("Paper-aligned Stage I requires timer_backbone.frozen=True")
        timer_model = getattr(timer_backbone, "timer_model", None)
        if timer_model is None:
            raise TypeError("Timer backbone must expose its pretrained timer_model")
        if any(parameter.requires_grad for parameter in timer_model.parameters()):
            raise ValueError("All pretrained Timer parameters must be frozen")
        if int(hidden_dim) != 256:
            raise ValueError("CARE fixes the Stage-I hidden dimension at 256")
        if int(tcn_layers) != 3:
            raise ValueError("CARE requires exactly three causal TCN blocks")
        if int(tcn_kernel_size) != 3:
            raise ValueError("CARE fixes the TCN kernel size at 3")
        if abs(float(dropout) - 0.1) > 1.0e-12:
            raise ValueError("CARE fixes Stage-I dropout at 0.1")

        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.target_index = int(target_index)
        self.mask_timer_representation = bool(mask_timer_representation)
        self.mask_regime_representation = bool(mask_regime_representation)
        self.timer_backbone = timer_backbone

        timer_dim = int(getattr(self.timer_backbone, "representation_dim", hidden_dim))
        self.timer_projection = nn.Linear(timer_dim, hidden_dim) if timer_dim != hidden_dim else nn.Identity()
        self.regime_tcn = CausalRegimeTCNEncoder(input_dim, hidden_dim, tcn_layers, tcn_kernel_size, dropout)

        # Phi_A in Eq. (3). The paper does not prescribe its internal width;
        # the existing 256-dimensional Stage-I setting is retained.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, int(fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(int(fusion_hidden_dim)),
        )
        # f_A in Eq. (3): one direct H-step Anchor prediction head.
        self.prediction_head = nn.Linear(int(fusion_hidden_dim), pred_len)

    def _timer_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = extract_timer_features(self.timer_backbone, x)
        return self.timer_projection(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"AnchorNetwork expects [B, L, F], got {tuple(x.shape)}")
        z_timer = self._timer_embedding(x)
        z_regime = self.regime_tcn(x)
        z_timer_used = torch.zeros_like(z_timer) if self.mask_timer_representation else z_timer
        z_regime_used = torch.zeros_like(z_regime) if self.mask_regime_representation else z_regime
        dual_source_representation = torch.cat([z_timer_used, z_regime_used], dim=-1)
        fused_representation = self.fusion_mlp(dual_source_representation)
        forecast = self.prediction_head(fused_representation)
        return {
            "forecast": forecast,
            "level_forecast": forecast,
            "timer_embedding": z_timer,
            "regime_embedding": z_regime,
            "timer_embedding_used": z_timer_used,
            "regime_embedding_used": z_regime_used,
            "dual_source_representation": dual_source_representation,
            "fused_representation": fused_representation,
        }


@dataclass(frozen=True)
class AnchorBuildResult:
    model: AnchorNetwork
    metadata: dict[str, Any]


def build_anchor(
    input_dim: int,
    pred_len: int,
    target_index: int,
    model_config: dict[str, Any],
    timer_backbone: nn.Module | None = None,
    timer_metadata: dict[str, Any] | None = None,
) -> AnchorBuildResult:
    mask_timer = bool(model_config.get("mask_timer_representation", False))
    mask_regime = bool(model_config.get("mask_regime_representation", False))
    model = AnchorNetwork(
        input_dim=input_dim,
        pred_len=pred_len,
        target_index=target_index,
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.1)),
        fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 256)),
        tcn_layers=int(model_config.get("tcn_layers", 3)),
        tcn_kernel_size=int(model_config.get("tcn_kernel_size", 3)),
        mask_timer_representation=mask_timer,
        mask_regime_representation=mask_regime,
        timer_backbone=timer_backbone,
    )
    metadata = {
        "architecture": "care_frozen_timer_plus_tcn_concat_fusion_anchor",
        "timer_backbone_used": True,
        "timer_fallback_allowed": False,
        "timer_metadata": timer_metadata or {},
        "representations": ["timer_embedding", "regime_embedding"],
        "fusion": "concat_then_mlp",
        "heads": ["forecast"],
        "target_encoder": "three_block_causal_dilated_tcn",
        "tcn_dilations": [1, 2, 4],
        "ablation_flags": ["mask_timer_representation", "mask_regime_representation"],
        "representation_ablation": {
            "timer_representation_used": not mask_timer,
            "regime_representation_used": not mask_regime,
            "masked_representation_value": "zeros_like",
            "both_masked_prediction_path": "zero_concat_through_existing_fusion_and_prediction_head",
            "additional_features_or_modules": False,
        },
    }
    return AnchorBuildResult(model=model, metadata=metadata)
