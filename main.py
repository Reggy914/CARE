#!/usr/bin/env python3
"""Unified command-line entry point for CARE."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "preprocess": "data_provider.prepare",
    "train": "exp.runner",
    "evaluate": "exp.evaluate",
    "inference": "exp.inference",
}


def print_help() -> None:
    print(
        "usage: python main.py {preprocess,train,evaluate,inference} [options]\n\n"
        "commands:\n"
        "  preprocess  Prepare a raw ELE CSV for CARE\n"
        "  train       Train and evaluate configured models\n"
        "  evaluate    Evaluate a saved checkpoint\n"
        "  inference   Generate predictions from a saved checkpoint\n\n"
        "Run 'python main.py <command> --help' for command-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return 0
    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"unknown command: {command!r}", file=sys.stderr)
        print_help()
        return 2
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    module = importlib.import_module(module_name)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
