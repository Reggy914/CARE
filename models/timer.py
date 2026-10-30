from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMER_MODEL_DIR = PROJECT_ROOT / "pretrained" / "timer-base-84m"

_TIMER_CACHE: dict[tuple[str, int, str], nn.Module] = {}


def cfg(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def checkpoint_file_hashes(model_dir: str | Path | None = TIMER_MODEL_DIR) -> dict[str, str]:
    model_dir = Path(TIMER_MODEL_DIR if model_dir is None else model_dir).resolve()
    patterns = ["config.json", "*.safetensors", "*.bin", "*.index.json", "configuration_timer.py", "modeling_timer.py", "ts_generation_mixin.py"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(model_dir.glob(pattern)))
    hashes: dict[str, str] = {}
    for path in sorted(set(files)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def timer_contract(pred_len: int = 96, model_dir: str | Path | None = TIMER_MODEL_DIR) -> dict[str, Any]:
    model_dir = Path(TIMER_MODEL_DIR if model_dir is None else model_dir).resolve()
    hashes = checkpoint_file_hashes(model_dir)
    weight_hashes = {path: value for path, value in hashes.items() if path.endswith((".safetensors", ".bin"))}
    return {
        "timer_checkpoint_path": str(model_dir),
        "timer_checkpoint_realpath": str(model_dir.resolve()),
        "timer_model_class": "HuggingFaceTimerZeroShotForecaster(AutoModelForCausalLM remote Timer)",
        "timer_loader": "transformers.AutoModelForCausalLM.from_pretrained",
        "trust_remote_code": True,
        "local_files_only": True,
        "input_len": 96,
        "pred_len": int(pred_len),
        "config_sha256": hashes.get(str(model_dir / "config.json")),
        "weight_sha256": weight_hashes,
        "all_checkpoint_sha256": hashes,
    }


class HuggingFaceTimerZeroShotForecaster(nn.Module):
    def __init__(self, timer_model: nn.Module, pred_len: int):
        super().__init__()
        self.timer_model = timer_model
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x.squeeze(-1)
        generated = self.timer_model.generate(series, max_new_tokens=self.pred_len)
        generated = generated[:, -self.pred_len :]
        return generated.unsqueeze(-1)

    def forward_trainable(self, x: torch.Tensor) -> torch.Tensor:
        series = x.squeeze(-1)
        output = self.timer_model(
            input_ids=series,
            max_output_length=self.pred_len,
            revin=True,
            return_dict=True,
        )
        return output.logits.unsqueeze(-1)


def load_timer(device: torch.device | str, pred_len: int = 96, model_dir: str | Path | None = TIMER_MODEL_DIR) -> nn.Module:
    device = torch.device(device)
    model_dir = Path(TIMER_MODEL_DIR if model_dir is None else model_dir).resolve()
    from transformers import AutoModelForCausalLM

    timer = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model = HuggingFaceTimerZeroShotForecaster(timer, int(pred_len)).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_shared_timer(device: torch.device | str, pred_len: int = 96, model_dir: str | Path | None = TIMER_MODEL_DIR) -> nn.Module:
    device = torch.device(device)
    model_dir = Path(TIMER_MODEL_DIR if model_dir is None else model_dir).resolve()
    key = (str(device), int(pred_len), str(model_dir))
    cached = _TIMER_CACHE.get(key)
    if cached is not None:
        return cached
    model = load_timer(device, pred_len, model_dir)
    _TIMER_CACHE[key] = model
    return model


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_timer_prediction(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2:
        pred = pred.unsqueeze(-1)
    return pred


def stabilize_timer_input(x: torch.Tensor, *, relative_std_floor: float = 1e-6) -> tuple[torch.Tensor, int]:
    """Make degenerate Timer windows safe for the remote RevIN normalization.

    Timer divides every input window by its own standard deviation. Constant or
    nearly constant windows therefore yield NaN predictions. Only such rows are
    replaced with a deterministic, mean-preserving micro-ramp; normal windows
    are returned unchanged.
    """
    if x.ndim != 3 or x.shape[-1] != 1:
        return x, 0

    series = x.squeeze(-1)
    finite_mask = torch.isfinite(series)
    has_nonfinite = ~finite_mask.all(dim=1)
    finite_values = torch.where(finite_mask, series, torch.zeros_like(series))
    finite_counts = finite_mask.sum(dim=1).clamp_min(1)
    row_mean = finite_values.sum(dim=1) / finite_counts
    cleaned = torch.where(finite_mask, series, row_mean.unsqueeze(1))
    row_std = cleaned.std(dim=1, unbiased=True)
    row_scale = cleaned.abs().amax(dim=1).clamp_min(1.0)
    low_variance = ~torch.isfinite(row_std) | (row_std < row_scale * relative_std_floor)
    repair_rows = has_nonfinite | low_variance
    if not repair_rows.any():
        return x, 0

    repaired = cleaned.clone()
    ramp = torch.linspace(-0.5, 0.5, steps=cleaned.shape[1], device=cleaned.device, dtype=cleaned.dtype)
    amplitude = row_scale[repair_rows] * relative_std_floor
    repaired[repair_rows] = row_mean[repair_rows].unsqueeze(1) + amplitude.unsqueeze(1) * ramp
    return repaired.unsqueeze(-1), int(repair_rows.sum().item())

# Preserve the former private name for out-of-tree callers.
_stabilize_timer_input = stabilize_timer_input


def timer_base_prediction(timer_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    timer_model.eval()
    stable_x, repaired_rows = stabilize_timer_input(x)
    if repaired_rows:
        print(f"[Timer] stabilized {repaired_rows}/{x.shape[0]} degenerate input window(s)", flush=True)
    original_device = stable_x.device
    timer_device = _module_device(timer_model)
    with torch.inference_mode():
        pred = _normalize_timer_prediction(timer_model(stable_x.to(timer_device).float()))
    if torch.isfinite(pred).all():
        return pred.to(original_device)
    if timer_device.type == "cuda":
        pred_len = int(getattr(timer_model, "pred_len", pred.shape[1] if pred.ndim >= 2 else 96))
        cpu_timer = load_shared_timer("cpu", pred_len, TIMER_MODEL_DIR)
        cpu_timer.eval()
        with torch.inference_mode():
            cpu_x = stable_x.detach().cpu().float()
            cpu_pred = _normalize_timer_prediction(cpu_timer(cpu_x))
            if not torch.isfinite(cpu_pred).all():
                row_finite = torch.isfinite(cpu_pred).flatten(1).all(dim=1)
                fixed = cpu_pred.clone()
                for row_idx in torch.nonzero(~row_finite, as_tuple=False).flatten().tolist():
                    sample_pred = _normalize_timer_prediction(cpu_timer(cpu_x[row_idx : row_idx + 1]))
                    fixed[row_idx : row_idx + 1] = sample_pred
                cpu_pred = fixed
        if torch.isfinite(cpu_pred).all():
            return cpu_pred.to(original_device)
    return pred.to(original_device)


class TimerModel:
    def __init__(self, config: Any | None = None):
        self.config = config or {}
        self.device = torch.device(cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        self.timer_model = load_shared_timer(self.device, self.pred_len, cfg(self.config, "timer_model_dir", TIMER_MODEL_DIR))
        self.contract = timer_contract(self.pred_len, cfg(self.config, "timer_model_dir", TIMER_MODEL_DIR))

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        self.timer_model.eval()
        return {"status": "zero_shot_no_training", "timer_contract": self.contract, "missing_keys": 0, "unexpected_keys": 0, "mismatched_keys": 0}

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        preds: list[torch.Tensor] = []
        self.timer_model.eval()
        with torch.inference_mode():
            for batch in test_loader:
                x = torch.as_tensor(batch[0], device=self.device).float()
                preds.append(timer_base_prediction(self.timer_model, x).detach().cpu())
        out = torch.cat(preds, dim=0)
        return out.numpy() if return_numpy else out


def unpack_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("inputs")))
        y = batch.get("y", batch.get("target", batch.get("targets")))
    else:
        x = batch[0]
        y = batch[1]
    return torch.as_tensor(x), torch.as_tensor(y)


def serializable_config(config: Any) -> dict[str, Any]:
    source = config if isinstance(config, dict) else vars(config) if hasattr(config, "__dict__") else {}
    return {k: v for k, v in source.items() if isinstance(v, (str, int, float, bool, list, tuple, dict)) or v is None}
