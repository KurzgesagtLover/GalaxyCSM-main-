"""Observational validation using a public open-cluster metallicity catalogue.

This complements the built-in benchmark manifest:
  - stellar / planetary checks still use public nominal reference values
  - galactic metallicity is compared directly against an external catalogue

The observational catalogue is Zhang et al. (2024), VizieR J/A+A/692/A212,
which provides open-cluster Galactocentric distances and [Fe/H].
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from gce.solver import GCESolver

from .benchmarks import load_manifest
from .pipeline import run_validation_pipeline

OPEN_CLUSTER_CATALOG_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/692/A212/tablea1.dat"
OPEN_CLUSTER_README = "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A+A/692/A212?format=html&tex=true"
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent / "cache" / "J_A+A_692_A212_tablea1.dat"
)


def _parse_float(field: str) -> float | None:
    text = field.strip()
    if not text:
        return None
    return float(text)


def _parse_int(field: str) -> int | None:
    text = field.strip()
    if not text:
        return None
    return int(text)


def _download_catalog(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "GalaxyCSM observational validation"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise

        # Fallback for local Python installs that do not ship an up-to-date
        # certificate bundle. The source is a fixed public-read CDS archive URL.
        insecure_context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=60, context=insecure_context) as response:
            payload = response.read()
    target_path.write_bytes(payload)


def load_open_cluster_catalog(
    cache_path: str | Path | None = None,
    force_download: bool = False,
) -> list[dict]:
    """Load the Zhang+2024 open-cluster metallicity catalogue."""
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if force_download or not path.exists():
        _download_catalog(OPEN_CLUSTER_CATALOG_URL, path)

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cluster = line[0:19].strip()
        if not cluster:
            continue

        rgc_kpc = _parse_float(line[45:51])
        feh_ann = _parse_float(line[101:119])
        feh_ann_err = _parse_float(line[120:137])
        feh_mcmc = _parse_float(line[138:159])
        feh_mcmc_err = _parse_float(line[160:177])
        nmemb = _parse_int(line[178:180]) or 0

        if rgc_kpc is None:
            continue
        if feh_mcmc is not None:
            feh = feh_mcmc
            feh_err = feh_mcmc_err
            feh_method = "mcmc"
        elif feh_ann is not None:
            feh = feh_ann
            feh_err = feh_ann_err
            feh_method = "ann"
        else:
            continue

        records.append(
            {
                "cluster": cluster,
                "rgc_kpc": float(rgc_kpc),
                "feh": float(feh),
                "feh_err": None if feh_err is None else float(feh_err),
                "nmemb": int(nmemb),
                "feh_method": feh_method,
            }
        )
    return records


def _weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted y = slope * x + intercept fit, returning slope uncertainty."""
    if x.size < 2:
        raise ValueError("At least two samples are required for a linear fit")

    safe_sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    weights = 1.0 / np.square(safe_sigma)
    design = np.column_stack([x, np.ones_like(x)])
    xtwx = design.T @ (weights[:, None] * design)
    xtwy = design.T @ (weights * y)
    beta = np.linalg.solve(xtwx, xtwy)
    cov = np.linalg.inv(xtwx)
    slope = float(beta[0])
    intercept = float(beta[1])
    slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    return slope, intercept, slope_err


def _simulate_galaxy_profile(manifest_path: str | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    config = manifest["scenarios"]["galaxy"]
    solver = GCESolver(config["params"])
    result = solver.solve()
    radii = np.asarray(result["radius"], dtype=float)
    feh = np.asarray(result["XH"]["Fe"], dtype=float)[:, -1]
    solar_radius = float(config["solar_radius_kpc"])
    return {
        "manifest_path": manifest["_path"],
        "radii_kpc": radii,
        "final_feh": feh,
        "solar_radius_kpc": solar_radius,
    }


def _weighted_mean(values: np.ndarray, sigma: np.ndarray) -> float:
    weights = 1.0 / np.square(np.clip(sigma, 1e-6, None))
    return float(np.average(values, weights=weights))


def run_open_cluster_validation(
    manifest_path: str | None = None,
    cache_path: str | Path | None = None,
    force_download: bool = False,
    min_members: int = 3,
    r_min: float = 4.0,
    r_max: float = 12.0,
) -> dict:
    """Compare the GCE [Fe/H] profile against an observed open-cluster catalogue."""
    observed_rows = load_open_cluster_catalog(cache_path=cache_path, force_download=force_download)
    observed_rows = [
        row
        for row in observed_rows
        if row["nmemb"] >= min_members and r_min <= row["rgc_kpc"] <= r_max
    ]
    if len(observed_rows) < 10:
        raise ValueError(
            f"Too few observed clusters after filtering: {len(observed_rows)}. "
            "Try lowering --min-members or widening the radius window."
        )

    x_obs = np.asarray([row["rgc_kpc"] for row in observed_rows], dtype=float)
    y_obs = np.asarray([row["feh"] for row in observed_rows], dtype=float)
    sigma_obs = np.asarray(
        [
            max(row["feh_err"], 0.03) if row["feh_err"] is not None else 0.05
            for row in observed_rows
        ],
        dtype=float,
    )

    obs_slope, obs_intercept, obs_slope_err = _weighted_linear_fit(x_obs, y_obs, sigma_obs)

    sim = _simulate_galaxy_profile(manifest_path)
    radii = sim["radii_kpc"]
    final_feh = sim["final_feh"]
    sim_at_observed = np.interp(x_obs, radii, final_feh)
    sim_slope_sampled, sim_intercept_sampled, _ = _weighted_linear_fit(
        x_obs,
        sim_at_observed,
        sigma_obs,
    )
    native_mask = (radii >= r_min) & (radii <= r_max)
    sim_native_slope = float(np.polyfit(radii[native_mask], final_feh[native_mask], 1)[0])

    residuals = sim_at_observed - y_obs
    abs_residuals = np.abs(residuals)
    reduced_chi2 = float(
        np.sum(np.square(residuals / sigma_obs)) / max(len(observed_rows) - 2, 1)
    )

    solar_radius = float(sim["solar_radius_kpc"])
    solar_model_feh = float(np.interp(solar_radius, radii, final_feh))
    solar_mask = np.abs(x_obs - solar_radius) <= 0.5
    if np.any(solar_mask):
        solar_observed_feh = _weighted_mean(y_obs[solar_mask], sigma_obs[solar_mask])
        solar_observed_scatter = float(np.std(y_obs[solar_mask], ddof=0))
        solar_cluster_count = int(np.sum(solar_mask))
    else:
        solar_observed_feh = None
        solar_observed_scatter = None
        solar_cluster_count = 0

    bin_edges = np.arange(r_min, r_max + 1.0, 1.0)
    binned_profile = []
    for idx in range(len(bin_edges) - 1):
        lo = float(bin_edges[idx])
        hi = float(bin_edges[idx + 1])
        if idx == len(bin_edges) - 2:
            in_bin = (x_obs >= lo) & (x_obs <= hi)
        else:
            in_bin = (x_obs >= lo) & (x_obs < hi)
        if not np.any(in_bin):
            continue

        center = 0.5 * (lo + hi)
        binned_profile.append(
            {
                "r_lo_kpc": lo,
                "r_hi_kpc": hi,
                "count": int(np.sum(in_bin)),
                "observed_weighted_mean_feh": _weighted_mean(y_obs[in_bin], sigma_obs[in_bin]),
                "observed_std_feh": float(np.std(y_obs[in_bin], ddof=0)),
                "simulated_feh": float(np.interp(center, radii, final_feh)),
            }
        )

    slope_delta = float(sim_slope_sampled - obs_slope)
    slope_sigma_offset = (
        None if obs_slope_err <= 0 else float(abs(slope_delta) / obs_slope_err)
    )

    if slope_sigma_offset is None:
        slope_status = "unknown"
    elif slope_sigma_offset <= 1.0:
        slope_status = "good"
    elif slope_sigma_offset <= 2.0:
        slope_status = "acceptable"
    else:
        slope_status = "tension"

    return {
        "source": {
            "label": "Zhang et al. (2024) open clusters",
            "catalog_id": "VizieR J/A+A/692/A212",
            "catalog_url": OPEN_CLUSTER_CATALOG_URL,
            "readme_url": OPEN_CLUSTER_README,
            "cache_path": str(Path(cache_path) if cache_path else DEFAULT_CACHE_PATH),
        },
        "selection": {
            "min_members": int(min_members),
            "r_min_kpc": float(r_min),
            "r_max_kpc": float(r_max),
            "n_clusters": int(len(observed_rows)),
        },
        "simulation": {
            "manifest_path": sim["manifest_path"],
            "solar_radius_kpc": solar_radius,
            "solar_feh": solar_model_feh,
            "native_slope_dex_per_kpc": sim_native_slope,
            "sampled_slope_dex_per_kpc": sim_slope_sampled,
            "sampled_intercept_dex": sim_intercept_sampled,
        },
        "observation": {
            "slope_dex_per_kpc": obs_slope,
            "slope_err_dex_per_kpc": obs_slope_err,
            "intercept_dex": obs_intercept,
            "solar_feh": solar_observed_feh,
            "solar_scatter_dex": solar_observed_scatter,
            "solar_cluster_count": solar_cluster_count,
        },
        "comparison": {
            "slope_delta_dex_per_kpc": slope_delta,
            "slope_sigma_offset": slope_sigma_offset,
            "slope_status": slope_status,
            "mean_signed_residual_dex": float(np.mean(residuals)),
            "mean_abs_residual_dex": float(np.mean(abs_residuals)),
            "median_abs_residual_dex": float(np.median(abs_residuals)),
            "rmse_dex": float(np.sqrt(np.mean(np.square(residuals)))),
            "reduced_chi2": reduced_chi2,
        },
        "profile": {
            "binned_profile": binned_profile,
        },
    }


def build_report(
    manifest_path: str | None = None,
    cache_path: str | Path | None = None,
    force_download: bool = False,
    min_members: int = 3,
    r_min: float = 4.0,
    r_max: float = 12.0,
) -> dict:
    return {
        "benchmark_validation": run_validation_pipeline(manifest_path),
        "observational_validation": run_open_cluster_validation(
            manifest_path=manifest_path,
            cache_path=cache_path,
            force_download=force_download,
            min_members=min_members,
            r_min=r_min,
            r_max=r_max,
        ),
    }


def format_markdown_report(report: dict) -> str:
    bench = report["benchmark_validation"]["summary"]
    obs = report["observational_validation"]
    sim = obs["simulation"]
    observed = obs["observation"]
    comp = obs["comparison"]

    lines = [
        "# GalaxyCSM Observational Validation",
        "",
        "## Internal Benchmarks",
        "",
        f"- Public benchmark pass rate: `{bench['passed']}/{bench['total']}` ({bench['pass_rate'] * 100:.1f}%)",
        "- Scope: stellar and planetary anchors still use public nominal reference values.",
        "",
        "## Open-Cluster Comparison",
        "",
        f"- Catalog: `{obs['source']['catalog_id']}`",
        f"- Selected clusters: `{obs['selection']['n_clusters']}` in `{obs['selection']['r_min_kpc']:.1f}-{obs['selection']['r_max_kpc']:.1f} kpc` with `Nmemb >= {obs['selection']['min_members']}`",
        f"- Observed slope: `{observed['slope_dex_per_kpc']:.4f} +/- {observed['slope_err_dex_per_kpc']:.4f} dex/kpc`",
        f"- Simulated slope (same cluster radii): `{sim['sampled_slope_dex_per_kpc']:.4f} dex/kpc`",
        f"- Simulated slope (native annuli): `{sim['native_slope_dex_per_kpc']:.4f} dex/kpc`",
        f"- Slope delta: `{comp['slope_delta_dex_per_kpc']:+.4f} dex/kpc`",
        f"- Slope assessment: `{comp['slope_status']}`",
        f"- Mean abs residual: `{comp['mean_abs_residual_dex']:.4f} dex`",
        f"- Median abs residual: `{comp['median_abs_residual_dex']:.4f} dex`",
        f"- RMSE: `{comp['rmse_dex']:.4f} dex`",
        f"- Reduced chi^2: `{comp['reduced_chi2']:.2f}`",
    ]

    if observed["solar_feh"] is not None:
        lines.extend(
            [
                "",
                "## Solar Annulus",
                "",
                f"- Observed weighted mean [Fe/H] near 8 kpc: `{observed['solar_feh']:+.4f} dex` from `{observed['solar_cluster_count']}` clusters",
                f"- Simulated [Fe/H] at 8 kpc: `{sim['solar_feh']:+.4f} dex`",
            ]
        )

    lines.extend(
        [
            "",
            "## 1 kpc Bins",
            "",
            "| Radius bin (kpc) | N | Observed [Fe/H] | Simulated [Fe/H] |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in obs["profile"]["binned_profile"]:
        lines.append(
            "| {lo:.1f}-{hi:.1f} | {count} | {obs_feh:+.4f} | {sim_feh:+.4f} |".format(
                lo=item["r_lo_kpc"],
                hi=item["r_hi_kpc"],
                count=item["count"],
                obs_feh=item["observed_weighted_mean_feh"],
                sim_feh=item["simulated_feh"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The open-cluster comparison is observational, but it only constrains the galactic metallicity field.",
            "- The simulator does not explicitly model radial migration, so residual scatter should be expected when compared to present-day cluster positions.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GalaxyCSM validation against an external open-cluster metallicity catalogue."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional benchmark manifest path for the internal validation pipeline.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional local path for the downloaded VizieR catalogue cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the observational catalogue even if the cache exists.",
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=3,
        help="Minimum number of cluster members required in the observational sample.",
    )
    parser.add_argument(
        "--r-min",
        type=float,
        default=4.0,
        help="Minimum Galactocentric radius in kpc for the observed sample.",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=12.0,
        help="Maximum Galactocentric radius in kpc for the observed sample.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Stdout format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path where the full JSON report should be written.",
    )
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        cache_path=args.cache_path,
        force_download=args.force_download,
        min_members=args.min_members,
        r_min=args.r_min,
        r_max=args.r_max,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
