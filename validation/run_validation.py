"""CLI entry point for the GalaxyCSM validation pipeline."""

from __future__ import annotations

import argparse
import json

from .pipeline import run_validation_pipeline
from .report import format_markdown_report, write_report_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GalaxyCSM validation benchmarks against public reference values."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a validation manifest JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where markdown and JSON reports should be written.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Stdout format when --output-dir is not provided.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any benchmark fails.",
    )
    args = parser.parse_args()

    report = run_validation_pipeline(args.manifest)

    if args.output_dir:
        paths = write_report_files(report, args.output_dir)
        print(format_markdown_report(report))
        print("")
        print(f"Wrote JSON report to {paths['json']}")
        print(f"Wrote Markdown report to {paths['markdown']}")
    elif args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_markdown_report(report))

    return 1 if args.strict and report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
