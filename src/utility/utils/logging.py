"""Structured logging and experiment reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json_metrics(metrics: Dict[str, Any], filepath: str | Path) -> None:
    """Save dictionary of metrics to a formatted JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_section(title: str, width: int = 70) -> None:
    """Print an ASCII visual section header."""
    print("=" * width)
    print(f" {title.upper()} ".center(width, " "))
    print("=" * width)
