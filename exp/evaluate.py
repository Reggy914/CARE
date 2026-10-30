"""Evaluate a trained CARE-compatible checkpoint on an ELE test split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from data_provider.ele import build_ele_loaders
from exp.runner import concatenate_evaluation_data, inverse
from exp.runtime import load_model
from utils.evaluator import (
    build_evaluation_region_masks,
    compute_diagnostic_metrics,
    compute_metrics,
)
from utils.predictor import save_predictions


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--model", default="care")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--timer-model-dir", type=Path)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-windows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bundle = build_ele_loaders(
        data_dir,
        target_column=args.target_column,
        input_len=args.input_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        max_test_windows=args.max_test_windows,
    )
    model = load_model(
        model_id=args.model,
        checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
        timer_model_dir=args.timer_model_dir,
        teacher_dir=args.teacher_dir,
        device=args.device,
        data_dir=data_dir,
        target_column=args.target_column,
        input_len=args.input_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        cache_dir=output_dir / "cache",
        split_info=bundle.split_info,
    )

    truth_scaled, sample_ids, last_observed_scaled = concatenate_evaluation_data(
        bundle.test_loader
    )
    predicted_scaled = np.asarray(model.predict(bundle.test_loader), dtype=np.float32)
    truth = inverse(bundle.scaler, truth_scaled)
    predicted = inverse(bundle.scaler, predicted_scaled)
    last_observed = inverse(bundle.scaler, last_observed_scaled)

    region_statistics = copy.deepcopy(bundle.split_info["evaluation_region_statistics"])
    masks = build_evaluation_region_masks(truth, last_observed, region_statistics)
    metrics = {
        **compute_metrics(truth, predicted),
        **compute_diagnostic_metrics(truth, predicted, masks),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / "predictions", truth, predicted, sample_ids)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "data_dir": str(data_dir),
        "target_column": args.target_column,
        "device": args.device,
        "window_count": int(len(sample_ids)),
        "metrics_file": str(output_dir / "metrics.json"),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": metrics, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
