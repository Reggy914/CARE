from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_predictions(output_dir: Path, y_true: Any, y_pred: Any, sample_id: Any | None = None, max_csv_points: int = 200000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    np.save(output_dir / "y_true.npy", truth)
    np.save(output_dir / "y_pred.npy", pred)
    sample = None if sample_id is None else np.asarray(sample_id, dtype=np.int64)
    if sample is not None:
        np.save(output_dir / "sample_id.npy", sample)
    t_flat = truth.reshape(-1)
    p_flat = pred.reshape(-1)
    n = min(len(t_flat), len(p_flat), int(max_csv_points))
    pd.DataFrame({"index": np.arange(n), "y_true": t_flat[:n], "y_pred": p_flat[:n]}).to_csv(output_dir / "prediction.csv", index=False)
