#!/usr/bin/env python3
"""Copy bundled deploy artifacts into models/ when missing (cloud/Docker startup)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_deploy_artifacts


def main() -> int:
    copied = ensure_deploy_artifacts()
    if copied:
        print(f"Restored {copied} artifact(s) to models/.")
    else:
        print("Models already present or no deploy bundle — nothing to restore.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
