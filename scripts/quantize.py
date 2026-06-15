#!/usr/bin/env python3
"""Quantize autoencoder models for edge deployment (TFLite)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edge_quantization import quantize_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge AI model quantization pipeline")
    parser.add_argument("--scenario", default="IR - 7")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    print(f"Quantizing autoencoder for edge deployment: {args.scenario}")
    results = quantize_autoencoder(args.scenario, sample_size=args.sample_size)
    print(json.dumps(results, indent=2))
    print("\nTFLite artifacts saved to models/")


if __name__ == "__main__":
    main()
