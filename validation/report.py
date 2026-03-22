"""Report rendering for validation pipeline output."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude < 1e-3:
        return f"{value:.3e}"
    if magnitude >= 100:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.4f}"
    return f"{value:.5f}"


def _fmt_tolerance(tolerance: dict) -> str:
    mode = tolerance["mode"]
    value = float(tolerance["value"])
    suffix = "abs" if mode == "abs" else "rel"
    return f"{_fmt_number(value)} {suffix}"


def format_markdown_report(report: dict) -> str:
    """Render a human-readable markdown report."""
    summary = report["summary"]
    lines = [
        "# GalaxyCSM Validation Report",
        "",
        f"- Manifest: `{report['metadata']['manifest_label']}`",
        f"- Generated: `{report['metadata']['generated_at']}`",
        f"- Passed: `{summary['passed']}/{summary['total']}` ({summary['pass_rate'] * 100:.1f}%)",
        "",
        "## Domain Summary",
        "",
        "| Domain | Passed | Failed | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain, stats in report["summary"]["domains"].items():
        lines.append(
            f"| {domain} | {stats['passed']} | {stats['failed']} | {stats['total']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Mismatches",
            "",
            "| Metric | Simulated | Observed | Abs diff | Tolerance | Status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary["worst_metrics"]:
        lines.append(
            "| {id} | {actual} | {observed} | {abs_diff} | {tol} | {status} |".format(
                id=item["id"],
                actual=_fmt_number(item["actual"]),
                observed=_fmt_number(item["observed"]),
                abs_diff=_fmt_number(item["abs_diff"]),
                tol=_fmt_tolerance(item["tolerance"]),
                status=item["status"],
            )
        )

    for domain in ("galaxy", "stellar", "planet"):
        domain_results = [item for item in report["results"] if item["domain"] == domain]
        if not domain_results:
            continue
        lines.extend(
            [
                "",
                f"## {domain.title()} Metrics",
                "",
                "| Metric | Simulated | Observed | Signed diff | Rel diff | Tolerance | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in domain_results:
            rel_pct = "-" if item["rel_diff"] is None else f"{item['rel_diff'] * 100:.1f}%"
            lines.append(
                "| {id} | {actual} | {observed} | {signed_diff} | {rel_diff} | {tol} | {status} |".format(
                    id=item["id"],
                    actual=_fmt_number(item["actual"]),
                    observed=_fmt_number(item["observed"]),
                    signed_diff=_fmt_number(item["signed_diff"]),
                    rel_diff=rel_pct,
                    tol=_fmt_tolerance(item["tolerance"]),
                    status=item["status"],
                )
            )

    galaxy_context = report["actuals"]["galaxy"]["context"]
    planet_context = report["actuals"]["planet"]["context"]
    lines.extend(
        [
            "",
            "## Scenario Notes",
            "",
            f"- Galaxy solar comparison used the nearest annulus at `{_fmt_number(galaxy_context['nearest_solar_annulus_kpc'])} kpc`.",
            f"- Planet benchmark used a differentiated surface reservoir with effective Bond albedo `{_fmt_number(planet_context['albedo_bond_effective'])}`.",
        ]
    )
    return "\n".join(lines)


def write_report_files(report: dict, output_dir: str | Path) -> dict:
    """Write JSON and markdown report files to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "validation_report.json"
    md_path = out_dir / "validation_report.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    md_path.write_text(format_markdown_report(report), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }
