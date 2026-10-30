"""Run checkpoint inference over an ELE train, validation, or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.ele import build_ele_loaders
from exp.runner import inverse
from exp.runtime import load_model


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--model", default="care")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--timer-model-dir", type=Path)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inference")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    limits = {
        "max_train_windows": args.max_windows if args.split == "train" else None,
        "max_val_windows": args.max_windows if args.split == "val" else None,
        "max_test_windows": args.max_windows if args.split == "test" else None,
    }
    bundle = build_ele_loaders(
        data_dir,
        target_column=args.target_column,
        input_len=args.input_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        **limits,
    )
    source_loader = {
        "train": bundle.train_loader,
        "val": bundle.val_loader,
        "test": bundle.test_loader,
    }[args.split]
    loader = DataLoader(source_loader.dataset, batch_size=args.batch_size, shuffle=False)
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
    prediction_scaled = np.asarray(model.predict(loader), dtype=np.float32)
    prediction = inverse(bundle.scaler, prediction_scaled)
    sample_ids = np.concatenate(
        [batch[2].detach().cpu().numpy() for batch in loader],
        axis=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", prediction)
    np.save(output_dir / "sample_id.npy", sample_ids)
    manifest = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "data_dir": str(data_dir),
        "target_column": args.target_column,
        "split": args.split,
        "device": args.device,
        "window_count": int(len(sample_ids)),
        "prediction_shape": list(prediction.shape),
    }
    (output_dir / "inference.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
