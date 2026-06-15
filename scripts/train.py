#!/usr/bin/env python3
"""Train edge-deployable models for industrial fault detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import list_fault_scenarios
from src.train_models import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Edge AI fault detection models")
    parser.add_argument(
        "--scenario",
        default="IR - 7",
        help="Fault scenario key (e.g. 'IR - 7', 'OR - 21')",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Rows per class for faster training (None = full dataset)",
    )
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:", ", ".join(list_fault_scenarios()))
        return

    print(f"Training Edge AI models for scenario: {args.scenario}")
    print("  → Random Forest (CPU edge classifier)...", flush=True)
    results = train_all(args.scenario, sample_size=args.sample_size)
    print(json.dumps(results, indent=2))
    print("\nArtifacts saved to models/. Run scripts/quantize.py next.")


if __name__ == "__main__":
    main()
