#!/usr/bin/env python3
"""List usable target columns from a processed ELE directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.ele import ele_target_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    for name in ele_target_columns(args.data_dir.expanduser().resolve()):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
