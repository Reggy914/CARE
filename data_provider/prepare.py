"""Command-line preprocessing for the ELE dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ele import preprocess_ele_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw ELE CSV into CARE's chronological processed format."
    )
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value <= 0.0 for value in ratios):
        parser.error("train, validation, and test ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1.0e-9:
        parser.error("train, validation, and test ratios must sum to 1")
    return args


def main() -> int:
    args = parse_args()
    result = preprocess_ele_csv(
        args.raw_csv.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
