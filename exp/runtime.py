"""Shared checkpoint-loading helpers for evaluation and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from exp.runner import (
    apply_external_model_paths,
    load_experiment_config,
    model_class,
    model_runtime_config,
)


def checkpoint_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    config = payload.get("config", {})
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config must be a mapping: {path}")
    return dict(config)


def load_model(
    *,
    model_id: str,
    checkpoint: Path | None,
    timer_model_dir: Path | None,
    teacher_dir: Path | None,
    device: str,
    data_dir: Path,
    target_column: str,
    input_len: int,
    pred_len: int,
    stride: int,
    batch_size: int,
    cache_dir: Path,
    split_info: dict[str, Any],
) -> Any:
    saved = checkpoint_config(checkpoint)
    for key in (
        "device",
        "processed_path",
        "raw_path",
        "timer_model_dir",
        "teacher_dir",
        "distill_cache_dir",
    ):
        saved.pop(key, None)

    config: dict[str, Any] = {
        **model_runtime_config(model_id),
        **load_experiment_config(),
        **saved,
        "device": device,
        "dataset": "ele",
        "target_column": target_column,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "processed_path": str(data_dir / "ele_processed.csv"),
        "split_info": split_info,
        "distill_cache_dir": str(cache_dir),
    }
    if timer_model_dir is not None:
        config = apply_external_model_paths(
            config,
            argparse_namespace(timer_model_dir=timer_model_dir),
        )
    if teacher_dir is not None:
        config["teacher_dir"] = str(teacher_dir.expanduser().resolve())

    model = model_class(model_id)(config)
    if checkpoint is None:
        if model_id != "timer":
            raise ValueError(f"--checkpoint is required for model {model_id!r}")
    else:
        loader = getattr(model, "load", None)
        if not callable(loader):
            raise ValueError(f"model {model_id!r} does not support checkpoint loading")
        loader(checkpoint, map_location=device)
    if hasattr(model, "eval"):
        model.eval()
    return model


class argparse_namespace:
    """Minimal namespace used by the runner's path override helper."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
