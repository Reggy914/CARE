"""Preprocessing and data loading for the project's only supported dataset: ELE."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ELE_DATASET_ID = "ele"



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELE_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ele"
DEFAULT_RAW_ELE_PATH = PROJECT_ROOT / "data" / "raw" / "ele.csv"
DEFAULT_ELE_PREPROCESS_DIR = PROJECT_ROOT / "data" / "processed" / "ele"


def _parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        median = float(numeric.dropna().median()) if numeric.notna().any() else 0.0
        unit = "ms" if median > 1e11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def preprocess_ele_csv(raw_path: str | Path = DEFAULT_RAW_ELE_PATH, processed_dir: str | Path = DEFAULT_ELE_PREPROCESS_DIR, *, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2) -> dict[str, Any]:
    raw_path = Path(raw_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raise FileNotFoundError(f"ele raw CSV not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if df.shape[1] < 2:
        raise ValueError("ele.csv must contain a timestamp column and at least one current column")

    timestamp_col = str(df.columns[0])
    profile: dict[str, Any] = {
        "raw_path": str(raw_path),
        "timestamp_col": timestamp_col,
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "missing_before": {str(k): int(v) for k, v in df.isna().sum().items()},
    }

    df[timestamp_col] = _parse_timestamp(df[timestamp_col])
    bad_timestamp = int(df[timestamp_col].isna().sum())
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(drop=True)

    current_cols: list[str] = []
    skipped_cols: list[dict[str, str]] = []
    cleaned = pd.DataFrame({"timestamp": df[timestamp_col]})
    for col in df.columns[1:]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            skipped_cols.append({"column": str(col), "reason": "all values empty or non-numeric"})
            continue
        numeric = numeric.interpolate(limit_direction="both").ffill().bfill()
        if numeric.isna().any():
            skipped_cols.append({"column": str(col), "reason": "remaining missing values after interpolation"})
            continue
        cleaned[str(col)] = numeric.astype("float32")
        current_cols.append(str(col))

    if not current_cols:
        raise ValueError("no valid current columns found in ele.csv")

    cleaned.to_csv(processed_dir / "ele_processed.csv", index=False)
    n = len(cleaned)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    split_info = {
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {"train_rows": train_end, "val_rows": val_end - train_end, "test_rows": n - val_end, "total_rows": n},
        "time_ranges": {
            "train": [str(cleaned["timestamp"].iloc[0]), str(cleaned["timestamp"].iloc[max(train_end - 1, 0)])] if train_end else None,
            "val": [str(cleaned["timestamp"].iloc[train_end]), str(cleaned["timestamp"].iloc[max(val_end - 1, train_end)])] if val_end > train_end else None,
            "test": [str(cleaned["timestamp"].iloc[val_end]), str(cleaned["timestamp"].iloc[-1])] if n > val_end else None,
        },
        "chronological": True,
    }
    profile.update({
        "processed_rows": int(n),
        "processed_columns": int(cleaned.shape[1]),
        "bad_timestamp_rows": bad_timestamp,
        "duplicate_timestamps_removed": int(before_dedup - len(df)),
        "time_start": str(cleaned["timestamp"].iloc[0]),
        "time_end": str(cleaned["timestamp"].iloc[-1]),
        "valid_current_columns": current_cols,
        "skipped_columns": skipped_cols,
        "missing_after": {str(k): int(v) for k, v in cleaned.isna().sum().items()},
    })
    columns = {"timestamp_col": "timestamp", "source_timestamp_col": timestamp_col, "current_cols": current_cols, "skipped_cols": skipped_cols}
    for name, payload in [("data_profile.json", profile), ("columns.json", columns), ("split_info.json", split_info)]:
        with (processed_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    return {"processed_path": str(processed_dir / "ele_processed.csv"), "profile": profile, "columns": columns, "split_info": split_info}



@dataclass(frozen=True)
class LoaderBundle:
    """ELE loaders and the train-fitted scaling metadata used by evaluation."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    target_column: str
    split_info: dict[str, Any]
    y_test_scaled: np.ndarray


def _read_metadata(directory: Path, filename: str) -> dict[str, Any]:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"ELE metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ELE metadata must be a JSON object: {path}")
    return payload


def ele_target_columns(ele_data_dir: str | Path = DEFAULT_ELE_DATA_DIR) -> list[str]:
    """Return the usable ELE current columns in their stored order."""

    directory = Path(ele_data_dir)
    columns = _read_metadata(directory, "columns.json")
    timestamp = str(columns.get("timestamp_col", "timestamp"))
    excluded = {str(value) for value in columns.get("excluded_columns", [])} | {timestamp}
    targets = [
        str(value)
        for value in columns.get("feature_columns", columns.get("current_cols", []))
        if str(value) not in excluded
    ]
    if not targets:
        raise ValueError(f"{directory}: columns.json contains no usable ELE target columns")
    if len(targets) != len(set(targets)):
        raise ValueError(f"{directory}: columns.json contains duplicate ELE target columns")
    return targets


def _make_windows(
    values: np.ndarray,
    input_len: int,
    pred_len: int,
    stride: int,
    maximum: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if input_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("input_len, pred_len, and stride must be positive")
    available = len(values) - input_len - pred_len + 1
    if available <= 0:
        raise ValueError(
            f"split has {len(values)} rows, but input_len + pred_len is "
            f"{input_len + pred_len}"
        )
    starts = np.arange(0, available, stride, dtype=np.int64)
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("maximum window count must be positive when provided")
        starts = starts[:maximum]
    x = np.stack([values[start : start + input_len] for start in starts]).astype(np.float32)
    y = np.stack(
        [values[start + input_len : start + input_len + pred_len, 0] for start in starts]
    ).astype(np.float32)
    return x, y, starts


def build_ele_loaders(
    ele_data_dir: str | Path = DEFAULT_ELE_DATA_DIR,
    *,
    target_column: str | None = None,
    input_len: int = 96,
    pred_len: int = 96,
    stride: int = 1,
    batch_size: int = 128,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
) -> LoaderBundle:
    """Build chronological, split-local, univariate ELE forecasting loaders."""

    directory = Path(ele_data_dir)
    if directory.is_file():
        raise ValueError(f"ele_data_dir must be the ELE directory, not a file: {directory}")
    csv_path = directory / "ele_processed.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"ELE processed CSV not found: {csv_path}")

    columns = _read_metadata(directory, "columns.json")
    split = _read_metadata(directory, "split_info.json")
    targets = ele_target_columns(directory)
    target_column = target_column or targets[0]
    if target_column not in targets:
        raise ValueError(f"unknown ELE target {target_column!r}; available: {targets}")

    header = pd.read_csv(csv_path, nrows=0)
    timestamp = str(columns.get("timestamp_col", "timestamp"))
    if len(header.columns) < 2:
        raise ValueError(f"ELE processed CSV needs a timestamp and target columns: {csv_path}")
    if str(header.columns[0]) != timestamp:
        raise ValueError(f"{csv_path}: first column must be {timestamp!r}")
    if target_column not in header.columns:
        raise ValueError(f"{csv_path}: target {target_column!r} is missing")
    frame = pd.read_csv(csv_path, usecols=[target_column])
    if frame.empty or not pd.api.types.is_numeric_dtype(frame[target_column]):
        raise ValueError(f"{csv_path}: target {target_column!r} is empty or non-numeric")

    values_raw = frame[[target_column]].to_numpy(dtype=np.float64)
    if not np.isfinite(values_raw).all():
        raise ValueError(f"{csv_path}: target {target_column!r} contains NaN or infinity")

    if split.get("chronological") is not True:
        raise ValueError(f"{directory}: ELE split must be chronological")
    counts = split.get("counts", {})
    train_rows = int(counts.get("train_rows", -1))
    val_rows = int(counts.get("val_rows", -1))
    test_rows = int(counts.get("test_rows", -1))
    total_rows = int(counts.get("total_rows", len(values_raw)))
    if min(train_rows, val_rows, test_rows) <= 0:
        raise ValueError(f"{directory}: split_info.json must define positive train/val/test rows")
    if total_rows != len(values_raw) or train_rows + val_rows + test_rows != len(values_raw):
        raise ValueError(
            f"{directory}: split counts do not match ELE CSV rows "
            f"({train_rows}+{val_rows}+{test_rows} != {len(values_raw)})"
        )

    train_end = train_rows
    val_end = train_rows + val_rows
    scaler = StandardScaler().fit(values_raw[:train_end])
    values = scaler.transform(values_raw).astype(np.float32)
    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    x_train, y_train, train_starts = _make_windows(
        train_values, input_len, pred_len, stride, max_train_windows
    )
    x_val, y_val, val_starts = _make_windows(
        val_values, input_len, pred_len, stride, max_val_windows
    )
    x_test, y_test, test_starts = _make_windows(
        test_values, input_len, pred_len, stride, max_test_windows
    )
    val_starts += train_end
    test_starts += val_end

    training_targets = values_raw[:train_end, 0]
    center_raw = float(np.median(training_targets))
    q_delta_raw = float(np.quantile(np.abs(np.diff(training_targets)), 0.90))
    q_amplitude_raw = float(np.quantile(np.abs(training_targets - center_raw), 0.90))

    def make_loader(
        x: np.ndarray, y: np.ndarray, starts: np.ndarray, *, shuffle: bool
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(starts.astype(np.int64)),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    info = {
        "dataset": ELE_DATASET_ID,
        "target_column": target_column,
        "processed_path": str(csv_path),
        "input_len": input_len,
        "pred_len": pred_len,
        "stride": stride,
        "scaler_fit": "train_only",
        "rows": {"train": train_rows, "val": val_rows, "test": test_rows},
        "windows": {
            "train": len(x_train),
            "val": len(x_val),
            "test": len(x_test),
        },
        "evaluation_region_statistics": {
            "source": "chronological_training_targets_only",
            "fit_scope": "train_only",
            "representation": "raw_target_space",
            "center_raw": center_raw,
            "q_delta_raw": q_delta_raw,
            "q_amplitude_raw": q_amplitude_raw,
            "delta_quantile": 0.90,
            "amplitude_quantile": 0.90,
            "boundary_definition": "abs(true_first_difference) >= train_q90_abs_difference",
            "high_amplitude_definition": "abs(true_value - train_median) >= train_q90_abs_deviation",
            "tail_definition": "boundary OR high_amplitude",
        },
        "sample_id": {
            "definition": "global ELE CSV input-window start row",
            "train_preview": train_starts[:5].tolist(),
            "val_preview": val_starts[:5].tolist(),
            "test_preview": test_starts[:5].tolist(),
        },
    }
    return LoaderBundle(
        train_loader=make_loader(x_train, y_train, train_starts, shuffle=True),
        val_loader=make_loader(x_val, y_val, val_starts, shuffle=False),
        test_loader=make_loader(x_test, y_test, test_starts, shuffle=False),
        scaler=scaler,
        target_column=target_column,
        split_info=info,
        y_test_scaled=y_test,
    )


__all__ = [
    "DEFAULT_ELE_DATA_DIR",
    "DEFAULT_ELE_PREPROCESS_DIR",
    "DEFAULT_RAW_ELE_PATH",
    "ELE_DATASET_ID",
    "LoaderBundle",
    "build_ele_loaders",
    "preprocess_ele_csv",
    "ele_target_columns",
]
