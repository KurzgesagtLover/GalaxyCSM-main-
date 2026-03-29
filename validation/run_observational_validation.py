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

from gce.radial_migration import (
    DEFAULT_OPEN_CLUSTER_MEAN_BIRTH_OFFSET_KPC,
    DEFAULT_OPEN_CLUSTER_SIGMA_KPC,
    build_migration_adjusted_profile,
    sample_stellar_migration_state,
)
from gce.solver import GCESolver
from gce.stellar import _overview_alive_until, _sample_imf, _sample_popiii_imf

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
        rguiding_kpc = _parse_float(line[70:87])
        rbirth_kpc = _parse_float(line[88:100])
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
                "rguiding_kpc": None if rguiding_kpc is None else float(rguiding_kpc),
                "rbirth_kpc": None if rbirth_kpc is None else float(rbirth_kpc),
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
    gas_feh = np.asarray(result["XH"]["Fe"], dtype=float)[:, -1]
    migration_model = _cluster_migration_config(config)
    if migration_model["enabled"]:
        migrated_feh = build_migration_adjusted_profile(
            radii,
            gas_feh,
            mean_birth_offset_kpc=migration_model["mean_birth_offset_kpc"],
            sigma_kpc=migration_model["sigma_kpc"],
        )
    else:
        migrated_feh = gas_feh.copy()
    solar_radius = float(config["solar_radius_kpc"])
    return {
        "manifest_path": manifest["_path"],
        "radii_kpc": radii,
        "gas_feh": gas_feh,
        "cluster_feh": migrated_feh,
        "migration_model": migration_model,
        "solar_radius_kpc": solar_radius,
    }


def _weighted_mean(values: np.ndarray, sigma: np.ndarray) -> float:
    weights = 1.0 / np.square(np.clip(sigma, 1e-6, None))
    return float(np.average(values, weights=weights))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        return None
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return None
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    return None if not np.isfinite(corr) else float(corr)


def _fit_linear_relation(x: np.ndarray, y: np.ndarray) -> dict:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2 or np.allclose(x_arr, x_arr[0]):
        return {
            "count": int(x_arr.size),
            "slope_dex_per_kpc": None,
            "intercept_dex": None,
            "scatter_dex": None,
        }
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    residuals = y_arr - (slope * x_arr + intercept)
    return {
        "count": int(x_arr.size),
        "slope_dex_per_kpc": float(slope),
        "intercept_dex": float(intercept),
        "scatter_dex": float(np.std(residuals, ddof=0)),
    }


def _assess_section_status(passed_checks: int, total_checks: int) -> str:
    if total_checks <= 0:
        return "unknown"
    if passed_checks == total_checks:
        return "good"
    if passed_checks >= max(int(np.ceil(total_checks * 0.6)), 1):
        return "mixed"
    return "tension"


def _kinematic_validation_config(config: dict) -> dict:
    raw = config.get("stellar_kinematics_validation", {})
    age_bins_default = ((0.0, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 12.0))
    sigma_r_default = ((10.0, 25.0), (18.0, 35.0), (24.0, 46.0), (30.0, 60.0))
    sigma_z_default = ((5.0, 14.0), (8.0, 20.0), (12.0, 28.0), (16.0, 36.0))
    ecc_default = ((0.02, 0.10), (0.03, 0.14), (0.05, 0.18), (0.06, 0.24))

    def _pairs(values, fallback):
        raw_values = values if values is not None else fallback
        return [tuple(map(float, item)) for item in raw_values]

    return {
        "sample_size": int(raw.get("sample_size", 12000)),
        "seed": int(raw.get("seed", 123)),
        "solar_annulus_kpc": tuple(map(float, raw.get("solar_annulus_kpc", (6.0, 10.0)))),
        "analysis_radius_kpc": tuple(map(float, raw.get("analysis_radius_kpc", (4.0, 12.0)))),
        "age_bins_gyr": _pairs(raw.get("age_bins_gyr"), age_bins_default),
        "sigma_r_reference_km_s": _pairs(raw.get("sigma_r_reference_km_s"), sigma_r_default),
        "sigma_z_reference_km_s": _pairs(raw.get("sigma_z_reference_km_s"), sigma_z_default),
        "eccentricity_reference": _pairs(raw.get("eccentricity_reference"), ecc_default),
        "young_age_max_gyr": float(raw.get("young_age_max_gyr", 2.0)),
        "old_age_min_gyr": float(raw.get("old_age_min_gyr", 8.0)),
    }


def _draw_validation_population(config: dict) -> dict:
    solver = GCESolver(config["params"])
    solver_result = solver.solve()
    solver_params = solver.p
    kin_cfg = _kinematic_validation_config(config)
    rng = np.random.default_rng(kin_cfg["seed"])

    t_grid = np.asarray(solver_result["time"], dtype=float)
    r_grid = np.asarray(solver_result["radius"], dtype=float)
    sfr = np.asarray(solver_result["sfr"], dtype=float)
    nt, nr = len(t_grid), len(r_grid)
    dt = np.zeros(nt, dtype=float)
    if nt > 1:
        dt[1:] = np.diff(t_grid)
        dt[0] = dt[1]
    else:
        dt[0] = max(float(solver_params.get("t_max", 1.0)), 1e-3)
    dr = r_grid[1] - r_grid[0] if nr > 1 else 1.0

    weights = np.zeros((nr, nt), dtype=float)
    for ir in range(nr):
        weights[ir] = sfr[ir] * dt * 2.0 * np.pi * max(r_grid[ir], 0.1) * dr
    weights = np.clip(weights, 0.0, None)
    total_weight = max(float(np.sum(weights)), 1e-12)
    probs = (weights / total_weight).ravel()

    n_stars = max(kin_cfg["sample_size"], 1000)
    bins = rng.choice(nr * nt, size=n_stars, p=probs)
    ir_arr, it_arr = bins // nt, bins % nt

    births = t_grid[it_arr]
    current_time = float(t_grid[-1])
    star_ages = np.maximum(current_time - births, 0.0)
    birth_radius_kpc = np.asarray(r_grid[ir_arr], dtype=float)
    metallicity_grid = np.asarray(solver_result.get("metallicity_tracked", solver_result["metallicity"]), dtype=float)
    birth_metallicity = metallicity_grid[ir_arr, it_arr]
    birth_feh = np.asarray(solver_result["XH"]["Fe"], dtype=float)[ir_arr, it_arr]

    masses = _sample_imf(rng, n_stars)
    popiii_mask = (births <= max(dt[0], 0.02)) | (birth_metallicity < 1e-4)
    if np.any(popiii_mask):
        masses[popiii_mask] = _sample_popiii_imf(rng, int(np.sum(popiii_mask)))

    migration_state = sample_stellar_migration_state(
        birth_radius_kpc,
        star_ages,
        rng,
        r_min_kpc=float(r_grid[0]),
        r_max_kpc=float(r_grid[-1]),
        drift_coeff_kpc=float(solver_params.get("stellar_migration_drift_coeff_kpc", 0.18)),
        max_shift_kpc=float(solver_params.get("stellar_migration_max_shift_kpc", 0.85)),
        sigma_floor_kpc=float(solver_params.get("stellar_migration_sigma_floor_kpc", 0.04)),
        sigma_sqrt_age_kpc=float(solver_params.get("stellar_migration_sigma_sqrt_age_kpc", 0.08)),
        resonance_strength=float(solver_params.get("stellar_migration_resonance_strength", 1.0)),
        sigma_r_floor_km_s=float(solver_params.get("stellar_sigma_r_floor_km_s", 8.0)),
        sigma_r_sqrt_age_km_s=float(solver_params.get("stellar_sigma_r_sqrt_age_km_s", 9.0)),
        sigma_z_floor_km_s=float(solver_params.get("stellar_sigma_z_floor_km_s", 5.0)),
        sigma_z_sqrt_age_km_s=float(solver_params.get("stellar_sigma_z_sqrt_age_km_s", 4.0)),
        v_circ_max_km_s=float(solver_params.get("stellar_vcirc_max_km_s", 232.0)),
        v_circ_turnover_kpc=float(solver_params.get("stellar_vcirc_turnover_kpc", 1.5)),
        eccentricity_max=float(solver_params.get("stellar_eccentricity_max", 0.35)),
    )
    alive_until = births + np.asarray(
        [
            _overview_alive_until(float(mass), metallicity_z=float(metallicity_z))
            for mass, metallicity_z in zip(masses, birth_metallicity)
        ],
        dtype=float,
    )
    alive_mask = current_time <= alive_until

    return {
        "manifest_path": config.get("_manifest_path"),
        "n_stars": int(n_stars),
        "current_time_gyr": current_time,
        "age_gyr": star_ages,
        "birth_time_gyr": births,
        "mass_msun": masses,
        "birth_metallicity_z": birth_metallicity,
        "birth_feh_dex": birth_feh,
        "birth_radius_kpc": birth_radius_kpc,
        "alive_mask": alive_mask,
        **migration_state,
    }


def _downsample_arrays(rng: np.random.Generator, data: dict[str, np.ndarray], max_points: int = 700) -> dict:
    arrays = {key: np.asarray(value) for key, value in data.items()}
    size = len(next(iter(arrays.values()))) if arrays else 0
    if size <= max_points:
        return {key: array.tolist() for key, array in arrays.items()}
    idx = np.sort(rng.choice(size, size=max_points, replace=False))
    return {key: array[idx].tolist() for key, array in arrays.items()}


def run_stellar_kinematic_validation(manifest_path: str | None = None) -> dict:
    """Run broad Milky Way-like chemo-dynamical sanity checks on the synthetic disk."""
    manifest = load_manifest(manifest_path)
    config = dict(manifest["scenarios"]["galaxy"])
    config["_manifest_path"] = manifest["_path"]
    kin_cfg = _kinematic_validation_config(config)
    sample = _draw_validation_population(config)

    age = np.asarray(sample["age_gyr"], dtype=float)
    current_radius = np.asarray(sample["current_radius_kpc"], dtype=float)
    guiding_radius = np.asarray(sample["guiding_radius_kpc"], dtype=float)
    feh = np.asarray(sample["birth_feh_dex"], dtype=float)
    ecc = np.asarray(sample["orbital_eccentricity"], dtype=float)
    sigma_r = np.asarray(sample["sigma_R_km_s"], dtype=float)
    sigma_z = np.asarray(sample["sigma_z_km_s"], dtype=float)
    v_r = np.asarray(sample["v_R_km_s"], dtype=float)
    v_z = np.asarray(sample["v_z_km_s"], dtype=float)
    alive_mask = np.asarray(sample["alive_mask"], dtype=bool)

    solar_lo, solar_hi = kin_cfg["solar_annulus_kpc"]
    analysis_lo, analysis_hi = kin_cfg["analysis_radius_kpc"]
    solar_mask = alive_mask & (current_radius >= solar_lo) & (current_radius <= solar_hi)
    analysis_mask = alive_mask & (current_radius >= analysis_lo) & (current_radius <= analysis_hi)

    age_binned_stats = []
    avr_passed = 0
    avr_total = 0
    ecc_passed = 0
    ecc_total = 0
    for idx, (age_lo, age_hi) in enumerate(kin_cfg["age_bins_gyr"]):
        in_bin = solar_mask & (age >= age_lo)
        if idx == len(kin_cfg["age_bins_gyr"]) - 1:
            in_bin &= age <= age_hi
        else:
            in_bin &= age < age_hi

        sigma_r_ref = kin_cfg["sigma_r_reference_km_s"][idx]
        sigma_z_ref = kin_cfg["sigma_z_reference_km_s"][idx]
        ecc_ref = kin_cfg["eccentricity_reference"][idx]
        count = int(np.sum(in_bin))

        if count > 0:
            sigma_r_model = float(np.mean(sigma_r[in_bin]))
            sigma_z_model = float(np.mean(sigma_z[in_bin]))
            sigma_r_measured = float(np.sqrt(2.0) * np.std(v_r[in_bin], ddof=0))
            sigma_z_measured = float(np.std(v_z[in_bin], ddof=0))
            median_ecc = float(np.median(ecc[in_bin]))
            p84_ecc = float(np.quantile(ecc[in_bin], 0.84))
            mean_age = float(np.mean(age[in_bin]))
        else:
            sigma_r_model = None
            sigma_z_model = None
            sigma_r_measured = None
            sigma_z_measured = None
            median_ecc = None
            p84_ecc = None
            mean_age = 0.5 * (age_lo + age_hi)

        sigma_r_ok = (
            sigma_r_measured is not None
            and float(sigma_r_ref[0]) <= sigma_r_measured <= float(sigma_r_ref[1])
        )
        sigma_z_ok = (
            sigma_z_measured is not None
            and float(sigma_z_ref[0]) <= sigma_z_measured <= float(sigma_z_ref[1])
        )
        ecc_ok = (
            median_ecc is not None
            and float(ecc_ref[0]) <= median_ecc <= float(ecc_ref[1])
        )

        avr_passed += int(bool(sigma_r_ok)) + int(bool(sigma_z_ok))
        avr_total += 2
        ecc_passed += int(bool(ecc_ok))
        ecc_total += 1

        age_binned_stats.append(
            {
                "age_lo_gyr": float(age_lo),
                "age_hi_gyr": float(age_hi),
                "count": count,
                "mean_age_gyr": float(mean_age),
                "sigma_r_model_mean_km_s": sigma_r_model,
                "sigma_r_measured_km_s": sigma_r_measured,
                "sigma_r_reference_range_km_s": [float(sigma_r_ref[0]), float(sigma_r_ref[1])],
                "sigma_r_in_reference": bool(sigma_r_ok),
                "sigma_z_model_mean_km_s": sigma_z_model,
                "sigma_z_measured_km_s": sigma_z_measured,
                "sigma_z_reference_range_km_s": [float(sigma_z_ref[0]), float(sigma_z_ref[1])],
                "sigma_z_in_reference": bool(sigma_z_ok),
                "median_eccentricity": median_ecc,
                "p84_eccentricity": p84_ecc,
                "eccentricity_reference_range": [float(ecc_ref[0]), float(ecc_ref[1])],
                "eccentricity_in_reference": bool(ecc_ok),
            }
        )

    sigma_r_age_corr = _safe_corr(age[solar_mask], sigma_r[solar_mask])
    sigma_z_age_corr = _safe_corr(age[solar_mask], sigma_z[solar_mask])
    ecc_age_corr = _safe_corr(age[solar_mask], ecc[solar_mask])

    avr_passed += int((sigma_r_age_corr or 0.0) > 0.2) + int((sigma_z_age_corr or 0.0) > 0.2)
    avr_total += 2
    ecc_passed += int((ecc_age_corr or 0.0) > 0.15)
    ecc_total += 1

    young_mask = analysis_mask & (age <= kin_cfg["young_age_max_gyr"])
    old_mask = analysis_mask & (age >= kin_cfg["old_age_min_gyr"])
    young_fit = _fit_linear_relation(guiding_radius[young_mask], feh[young_mask])
    old_fit = _fit_linear_relation(guiding_radius[old_mask], feh[old_mask])

    young_negative_gradient = (
        young_fit["slope_dex_per_kpc"] is not None and young_fit["slope_dex_per_kpc"] < -0.02
    )
    old_flatter_than_young = (
        young_fit["slope_dex_per_kpc"] is not None
        and old_fit["slope_dex_per_kpc"] is not None
        and abs(old_fit["slope_dex_per_kpc"]) <= abs(young_fit["slope_dex_per_kpc"])
    )
    old_scatter_exceeds_young = (
        young_fit["scatter_dex"] is not None
        and old_fit["scatter_dex"] is not None
        and old_fit["scatter_dex"] >= young_fit["scatter_dex"]
    )
    feh_rg_passed = (
        int(bool(young_negative_gradient))
        + int(bool(old_flatter_than_young))
        + int(bool(old_scatter_exceeds_young))
    )
    feh_rg_total = 3

    rng = np.random.default_rng(kin_cfg["seed"] + 17)
    young_points = _downsample_arrays(
        rng,
        {
            "guiding_radius_kpc": guiding_radius[young_mask],
            "feh_dex": feh[young_mask],
            "current_radius_kpc": current_radius[young_mask],
            "age_gyr": age[young_mask],
        },
    )
    old_points = _downsample_arrays(
        rng,
        {
            "guiding_radius_kpc": guiding_radius[old_mask],
            "feh_dex": feh[old_mask],
            "current_radius_kpc": current_radius[old_mask],
            "age_gyr": age[old_mask],
        },
    )

    total_passed = avr_passed + ecc_passed + feh_rg_passed
    total_checks = avr_total + ecc_total + feh_rg_total

    return {
        "source": {
            "label": "Synthetic stellar chemo-dynamical diagnostics",
            "reference_type": "broad_milky_way_disk_envelopes",
            "manifest_path": manifest["_path"],
        },
        "selection": {
            "sample_size": int(sample["n_stars"]),
            "present_time_gyr": float(sample["current_time_gyr"]),
            "alive_stars": int(np.sum(alive_mask)),
            "solar_annulus_alive_stars": int(np.sum(solar_mask)),
            "analysis_annulus_alive_stars": int(np.sum(analysis_mask)),
            "solar_annulus_kpc": [float(solar_lo), float(solar_hi)],
            "analysis_radius_kpc": [float(analysis_lo), float(analysis_hi)],
            "young_age_max_gyr": float(kin_cfg["young_age_max_gyr"]),
            "old_age_min_gyr": float(kin_cfg["old_age_min_gyr"]),
        },
        "age_binned_stats": age_binned_stats,
        "avr": {
            "sigma_R_age_correlation": sigma_r_age_corr,
            "sigma_z_age_correlation": sigma_z_age_corr,
            "passed_checks": int(avr_passed),
            "total_checks": int(avr_total),
            "status": _assess_section_status(avr_passed, avr_total),
        },
        "eccentricity": {
            "age_correlation": ecc_age_corr,
            "passed_checks": int(ecc_passed),
            "total_checks": int(ecc_total),
            "status": _assess_section_status(ecc_passed, ecc_total),
        },
        "feh_guiding_radius": {
            "young": {
                "age_max_gyr": float(kin_cfg["young_age_max_gyr"]),
                **young_fit,
            },
            "old": {
                "age_min_gyr": float(kin_cfg["old_age_min_gyr"]),
                **old_fit,
            },
            "diagnostics": {
                "young_negative_gradient": bool(young_negative_gradient),
                "old_flatter_than_young": bool(old_flatter_than_young),
                "old_scatter_exceeds_young": bool(old_scatter_exceeds_young),
            },
            "passed_checks": int(feh_rg_passed),
            "total_checks": int(feh_rg_total),
            "status": _assess_section_status(feh_rg_passed, feh_rg_total),
        },
        "raw_samples": {
            "young_feh_guiding_radius": young_points,
            "old_feh_guiding_radius": old_points,
        },
        "overall": {
            "passed_checks": int(total_passed),
            "total_checks": int(total_checks),
            "status": _assess_section_status(total_passed, total_checks),
        },
    }


def _cluster_migration_config(config: dict) -> dict:
    raw = config.get("open_cluster_migration", {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "mean_birth_offset_kpc": float(
            raw.get("mean_birth_offset_kpc", DEFAULT_OPEN_CLUSTER_MEAN_BIRTH_OFFSET_KPC)
        ),
        "sigma_kpc": float(raw.get("sigma_kpc", DEFAULT_OPEN_CLUSTER_SIGMA_KPC)),
    }


def _summarize_migration_measurements(rows: list[dict]) -> dict:
    guiding_offsets = np.asarray(
        [
            row["rguiding_kpc"] - row["rgc_kpc"]
            for row in rows
            if row.get("rguiding_kpc") is not None
        ],
        dtype=float,
    )
    birth_offsets = np.asarray(
        [
            row["rbirth_kpc"] - row["rgc_kpc"]
            for row in rows
            if row.get("rbirth_kpc") is not None
        ],
        dtype=float,
    )

    def _summary(values: np.ndarray) -> dict:
        if values.size == 0:
            return {
                "count": 0,
                "mean_kpc": None,
                "std_kpc": None,
                "median_kpc": None,
            }
        return {
            "count": int(values.size),
            "mean_kpc": float(np.mean(values)),
            "std_kpc": float(np.std(values)),
            "median_kpc": float(np.median(values)),
        }

    return {
        "guiding_minus_current": _summary(guiding_offsets),
        "birth_minus_current": _summary(birth_offsets),
    }


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
    migration_stats = _summarize_migration_measurements(observed_rows)

    sim = _simulate_galaxy_profile(manifest_path)
    radii = sim["radii_kpc"]
    gas_feh = sim["gas_feh"]
    migrated_feh = sim["cluster_feh"]
    sim_at_observed_raw = np.interp(x_obs, radii, gas_feh)
    sim_at_observed = np.interp(x_obs, radii, migrated_feh)
    sim_gas_slope_sampled, sim_gas_intercept_sampled, _ = _weighted_linear_fit(
        x_obs,
        sim_at_observed_raw,
        sigma_obs,
    )
    sim_slope_sampled, sim_intercept_sampled, _ = _weighted_linear_fit(
        x_obs,
        sim_at_observed,
        sigma_obs,
    )
    native_mask = (radii >= r_min) & (radii <= r_max)
    sim_native_gas_slope = float(np.polyfit(radii[native_mask], gas_feh[native_mask], 1)[0])
    sim_native_slope = float(np.polyfit(radii[native_mask], migrated_feh[native_mask], 1)[0])

    gas_residuals = sim_at_observed_raw - y_obs
    residuals = sim_at_observed - y_obs
    gas_abs_residuals = np.abs(gas_residuals)
    abs_residuals = np.abs(residuals)
    gas_reduced_chi2 = float(
        np.sum(np.square(gas_residuals / sigma_obs)) / max(len(observed_rows) - 2, 1)
    )
    reduced_chi2 = float(
        np.sum(np.square(residuals / sigma_obs)) / max(len(observed_rows) - 2, 1)
    )

    solar_radius = float(sim["solar_radius_kpc"])
    solar_model_feh_raw = float(np.interp(solar_radius, radii, gas_feh))
    solar_model_feh = float(np.interp(solar_radius, radii, migrated_feh))
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
                "simulated_feh": float(np.interp(center, radii, migrated_feh)),
                "simulated_feh_raw": float(np.interp(center, radii, gas_feh)),
            }
        )

    gas_slope_delta = float(sim_gas_slope_sampled - obs_slope)
    slope_delta = float(sim_slope_sampled - obs_slope)
    gas_slope_sigma_offset = (
        None if obs_slope_err <= 0 else float(abs(gas_slope_delta) / obs_slope_err)
    )
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
            "solar_feh_raw": solar_model_feh_raw,
            "native_slope_dex_per_kpc": sim_native_slope,
            "native_gas_slope_dex_per_kpc": sim_native_gas_slope,
            "sampled_slope_dex_per_kpc": sim_slope_sampled,
            "sampled_intercept_dex": sim_intercept_sampled,
            "sampled_gas_slope_dex_per_kpc": sim_gas_slope_sampled,
            "sampled_gas_intercept_dex": sim_gas_intercept_sampled,
            "radii_kpc": radii.tolist(),
            "gas_profile_feh": gas_feh.tolist(),
            "migration_adjusted_profile_feh": migrated_feh.tolist(),
        },
        "observation": {
            "slope_dex_per_kpc": obs_slope,
            "slope_err_dex_per_kpc": obs_slope_err,
            "intercept_dex": obs_intercept,
            "solar_feh": solar_observed_feh,
            "solar_scatter_dex": solar_observed_scatter,
            "solar_cluster_count": solar_cluster_count,
        },
        "migration_model": {
            **sim["migration_model"],
            "observed_guiding_minus_current": migration_stats["guiding_minus_current"],
            "observed_birth_minus_current": migration_stats["birth_minus_current"],
        },
        "comparison": {
            "gas_only_slope_delta_dex_per_kpc": gas_slope_delta,
            "gas_only_slope_sigma_offset": gas_slope_sigma_offset,
            "slope_delta_dex_per_kpc": slope_delta,
            "slope_sigma_offset": slope_sigma_offset,
            "slope_status": slope_status,
            "gas_only_mean_signed_residual_dex": float(np.mean(gas_residuals)),
            "gas_only_mean_abs_residual_dex": float(np.mean(gas_abs_residuals)),
            "gas_only_median_abs_residual_dex": float(np.median(gas_abs_residuals)),
            "gas_only_rmse_dex": float(np.sqrt(np.mean(np.square(gas_residuals)))),
            "gas_only_reduced_chi2": gas_reduced_chi2,
            "mean_signed_residual_dex": float(np.mean(residuals)),
            "mean_abs_residual_dex": float(np.mean(abs_residuals)),
            "median_abs_residual_dex": float(np.median(abs_residuals)),
            "rmse_dex": float(np.sqrt(np.mean(np.square(residuals)))),
            "reduced_chi2": reduced_chi2,
            "mean_abs_residual_gain_dex": float(np.mean(gas_abs_residuals) - np.mean(abs_residuals)),
            "rmse_gain_dex": float(
                np.sqrt(np.mean(np.square(gas_residuals))) - np.sqrt(np.mean(np.square(residuals)))
            ),
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
        "stellar_kinematic_validation": run_stellar_kinematic_validation(
            manifest_path=manifest_path,
        ),
    }


def format_markdown_report(report: dict) -> str:
    bench = report["benchmark_validation"]["summary"]
    obs = report["observational_validation"]
    kin = report.get("stellar_kinematic_validation", {})
    sim = obs["simulation"]
    observed = obs["observation"]
    comp = obs["comparison"]
    migration = obs["migration_model"]

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
        (
            "- Effective migration kernel: "
            f"`+{migration['mean_birth_offset_kpc']:.2f} kpc` birth-radius offset, "
            f"`sigma={migration['sigma_kpc']:.2f} kpc`"
            if migration["enabled"]
            else "- Effective migration kernel: `disabled`"
        ),
        f"- Observed slope: `{observed['slope_dex_per_kpc']:.4f} +/- {observed['slope_err_dex_per_kpc']:.4f} dex/kpc`",
        f"- Simulated slope (migration-aware): `{sim['sampled_slope_dex_per_kpc']:.4f} dex/kpc`",
        f"- Simulated slope (gas only): `{sim['sampled_gas_slope_dex_per_kpc']:.4f} dex/kpc`",
        f"- Simulated slope (native annuli, migration-aware): `{sim['native_slope_dex_per_kpc']:.4f} dex/kpc`",
        f"- Slope delta: `{comp['slope_delta_dex_per_kpc']:+.4f} dex/kpc`",
        f"- Slope assessment: `{comp['slope_status']}`",
        f"- Mean abs residual: `{comp['mean_abs_residual_dex']:.4f} dex` (gas only `{comp['gas_only_mean_abs_residual_dex']:.4f}`)",
        f"- Median abs residual: `{comp['median_abs_residual_dex']:.4f} dex`",
        f"- RMSE: `{comp['rmse_dex']:.4f} dex` (gas only `{comp['gas_only_rmse_dex']:.4f}`)",
        f"- Reduced chi^2: `{comp['reduced_chi2']:.2f}`",
    ]

    if observed["solar_feh"] is not None:
        lines.extend(
            [
                "",
                "## Solar Annulus",
                "",
                f"- Observed weighted mean [Fe/H] near 8 kpc: `{observed['solar_feh']:+.4f} dex` from `{observed['solar_cluster_count']}` clusters",
                f"- Simulated [Fe/H] at 8 kpc (migration-aware): `{sim['solar_feh']:+.4f} dex`",
                f"- Simulated [Fe/H] at 8 kpc (gas only): `{sim['solar_feh_raw']:+.4f} dex`",
            ]
        )

    if kin:
        kin_overall = kin["overall"]
        avr = kin["avr"]
        ecc = kin["eccentricity"]
        feh_rg = kin["feh_guiding_radius"]
        lines.extend(
            [
                "",
                "## Stellar Chemo-Dynamical Diagnostics",
                "",
                f"- Overall status: `{kin_overall['status']}` ({kin_overall['passed_checks']}/{kin_overall['total_checks']} checks)",
                f"- Solar-annulus sample: `{kin['selection']['solar_annulus_alive_stars']}` alive stars in `{kin['selection']['solar_annulus_kpc'][0]:.1f}-{kin['selection']['solar_annulus_kpc'][1]:.1f} kpc`",
                f"- AVR status: `{avr['status']}` with age-correlation `sigma_R={avr['sigma_R_age_correlation']:.3f}` and `sigma_z={avr['sigma_z_age_correlation']:.3f}`" if avr["sigma_R_age_correlation"] is not None and avr["sigma_z_age_correlation"] is not None else f"- AVR status: `{avr['status']}`",
                f"- Age-eccentricity status: `{ecc['status']}` with correlation `{ecc['age_correlation']:.3f}`" if ecc["age_correlation"] is not None else f"- Age-eccentricity status: `{ecc['status']}`",
                f"- Young [Fe/H]-R_g slope: `{feh_rg['young']['slope_dex_per_kpc']:.4f} dex/kpc`" if feh_rg["young"]["slope_dex_per_kpc"] is not None else "- Young [Fe/H]-R_g slope: `n/a`",
                f"- Old [Fe/H]-R_g slope: `{feh_rg['old']['slope_dex_per_kpc']:.4f} dex/kpc`" if feh_rg["old"]["slope_dex_per_kpc"] is not None else "- Old [Fe/H]-R_g slope: `n/a`",
                f"- Old-vs-young flattening: `{feh_rg['diagnostics']['old_flatter_than_young']}`",
                f"- Old-vs-young scatter growth: `{feh_rg['diagnostics']['old_scatter_exceeds_young']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 1 kpc Bins",
            "",
            "| Radius bin (kpc) | N | Observed [Fe/H] | Migration-aware | Gas only |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in obs["profile"]["binned_profile"]:
        lines.append(
            "| {lo:.1f}-{hi:.1f} | {count} | {obs_feh:+.4f} | {sim_feh:+.4f} | {sim_feh_raw:+.4f} |".format(
                lo=item["r_lo_kpc"],
                hi=item["r_hi_kpc"],
                count=item["count"],
                obs_feh=item["observed_weighted_mean_feh"],
                sim_feh=item["simulated_feh"],
                sim_feh_raw=item["simulated_feh_raw"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The open-cluster comparison is observational, but it only constrains the galactic metallicity field.",
            "- The comparison now applies a lightweight effective migration kernel before sampling the present-day cluster radii.",
            "- The stellar chemo-dynamical section is a diagnostic sanity check against broad Milky Way disk envelopes, not a direct catalogue likelihood fit.",
            "- This is still not a full chemo-dynamical orbit integration, so residual scatter should remain even when the mean gradient is reproduced.",
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
