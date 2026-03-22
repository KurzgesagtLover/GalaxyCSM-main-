"""Metric evaluation helpers for validation reports."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite


def evaluate_metric(metric: dict, actual: float) -> dict:
    """Compare one simulated value against its observed benchmark."""
    observed = float(metric["observed"])
    tolerance = metric["tolerance"]
    tol_mode = tolerance["mode"]
    tol_value = float(tolerance["value"])
    actual = float(actual)

    signed_diff = actual - observed
    abs_diff = abs(signed_diff)
    rel_diff = None if observed == 0 else abs_diff / abs(observed)

    if tol_mode == "abs":
        passed = abs_diff <= tol_value
        normalized_error = abs_diff / tol_value if tol_value > 0 else float("inf")
    elif tol_mode == "rel":
        if rel_diff is None:
            raise ValueError(f"Relative tolerance is invalid for zero observed value: {metric['id']}")
        passed = rel_diff <= tol_value
        normalized_error = rel_diff / tol_value if tol_value > 0 else float("inf")
    else:
        raise ValueError(f"Unsupported tolerance mode: {tol_mode}")

    if not isfinite(normalized_error):
        normalized_error = float("inf")

    return {
        "id": metric["id"],
        "domain": metric["domain"],
        "label": metric["label"],
        "unit": metric["unit"],
        "source": metric.get("source", ""),
        "observed": observed,
        "actual": actual,
        "signed_diff": signed_diff,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "tolerance": {
            "mode": tol_mode,
            "value": tol_value,
        },
        "passed": passed,
        "status": "pass" if passed else "fail",
        "normalized_error": normalized_error,
    }


def summarize_results(results: list[dict]) -> dict:
    """Build total and per-domain pass/fail summaries."""
    by_domain = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    passed = 0
    failed = 0

    for result in results:
        domain_summary = by_domain[result["domain"]]
        domain_summary["total"] += 1
        if result["passed"]:
            passed += 1
            domain_summary["passed"] += 1
        else:
            failed += 1
            domain_summary["failed"] += 1

    worst = sorted(
        results,
        key=lambda item: item["normalized_error"],
        reverse=True,
    )[:5]

    return {
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "pass_rate": (passed / len(results)) if results else 0.0,
        "domains": dict(by_domain),
        "worst_metrics": worst,
    }
