"""Benchmark manifest loading for validation runs."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "benchmarks" / "public_full_stack.json"
)


def load_manifest(path: str | None = None) -> dict:
    """Load a validation manifest from disk."""
    manifest_path = Path(path) if path else DEFAULT_MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    metrics = manifest.get("metrics", [])
    scenarios = manifest.get("scenarios", {})
    if not metrics:
        raise ValueError("Validation manifest does not define any metrics")
    if not scenarios:
        raise ValueError("Validation manifest does not define any scenarios")

    for metric in metrics:
        domain = metric.get("domain")
        key = metric.get("key")
        if domain not in scenarios:
            raise ValueError(f"Unknown metric domain in manifest: {domain}")
        if not key:
            raise ValueError(f"Metric is missing a key: {metric!r}")

    manifest["_path"] = str(manifest_path)
    return manifest
