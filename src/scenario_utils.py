"""Lightweight scenario helpers (no ML dependencies)."""

from __future__ import annotations

import re


def scenario_prefix(fault_key: str) -> str:
    return re.sub(r"[\s\-_]+", "_", fault_key.strip()).strip("_")
