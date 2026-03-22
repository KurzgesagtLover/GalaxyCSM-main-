"""End-to-end validation pipeline for public physics benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from gce.planets import (
    build_outgassing_reservoir,
    compute_atmosphere,
    compute_physical_properties,
    differentiate_full,
)
from gce.solver import GCESolver
from gce.stellar import stellar_evolution

from .benchmarks import load_manifest
from .metrics import evaluate_metric, summarize_results


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _run_galaxy_scenario(config: dict) -> dict:
    solver = GCESolver(config["params"])
    result = solver.solve()

    radii = np.asarray(result["radius"], dtype=float)
    feh = np.asarray(result["XH"]["Fe"], dtype=float)[:, -1]
    metallicity = np.asarray(result["metallicity"], dtype=float)[:, -1]

    solar_radius = float(config["solar_radius_kpc"])
    solar_idx = _nearest_index(radii, solar_radius)
    slope_lo, slope_hi = map(float, config["slope_window_kpc"])
    slope_mask = (radii >= slope_lo) & (radii <= slope_hi)
    radial_slope = float(np.polyfit(radii[slope_mask], feh[slope_mask], 1)[0])

    return {
        "metrics": {
            "solar_feh": float(feh[solar_idx]),
            "solar_metallicity_z": float(metallicity[solar_idx]),
            "radial_feh_slope": radial_slope,
        },
        "context": {
            "nearest_solar_annulus_kpc": float(radii[solar_idx]),
            "slope_window_kpc": [slope_lo, slope_hi],
            "radii_kpc": radii.tolist(),
            "final_feh_profile": [float(value) for value in feh.tolist()],
        },
    }


def _run_stellar_scenario(config: dict) -> dict:
    state = stellar_evolution(
        float(config["mass_msun"]),
        float(config["age_gyr"]),
        metallicity_z=float(config["metallicity_z"]),
        model=config.get("stellar_model"),
    )
    return {
        "metrics": {
            "solar_teff": float(state["T_eff"]),
            "solar_luminosity_lsun": float(state["luminosity"]),
            "solar_radius_rsun": float(state["radius"]),
        },
        "context": {
            "phase": state["phase"],
            "spectral_class": state["spectral_class"],
            "age_gyr": float(config["age_gyr"]),
            "metallicity_z": float(config["metallicity_z"]),
            "stellar_model": config.get("stellar_model"),
        },
    }


def _run_planet_scenario(config: dict) -> dict:
    diff = differentiate_full(
        float(config["mass_earth"]),
        dict(config["bulk_comp"]),
        float(config["oxidation_iw"]),
        diff_progress=float(config.get("diff_progress", 1.0)),
        lv_class_params=dict(config["late_veneer"]),
    )
    surface_reservoir = build_outgassing_reservoir(diff)

    physical = compute_physical_properties(
        float(config["mass_earth"]),
        config["ptype"],
        float(config["semi_major_au"]),
        float(config["star_mass_msun"]),
        float(config["rotation_period_hr"]),
        float(config["axial_tilt_deg"]),
        float(config["ecc"]),
        float(config["age_gyr"]),
        L_star_Lsun=float(config["stellar_luminosity_lsun"]),
    )
    atmosphere = compute_atmosphere(
        surface_reservoir,
        float(config["mass_earth"]),
        config["ptype"],
        float(physical["T_eq_K"]),
        float(physical["g_mean_m_s2"]),
        float(physical["v_esc_km_s"]) * 1000.0,
        float(config["age_gyr"]),
        float(config["oxidation_iw"]),
        albedo_bond=float(physical["albedo_bond"]),
        L_star_Lsun=float(config["stellar_luminosity_lsun"]),
        star_teff=float(config["star_teff_K"]),
        semi_major_au=float(config["semi_major_au"]),
        rotation_period_hr=float(physical["rotation_effective_hr"]),
        orbital_period_days=float(physical["orbital_period_days"]),
        tidally_locked=bool(physical["tidally_locked"]),
        tidal_heating_TW=float(physical["tidal_heating_TW"]),
    )

    return {
        "metrics": {
            "earth_insolation_w_m2": float(physical["insolation_W_m2"]),
            "earth_gravity_m_s2": float(physical["g_mean_m_s2"]),
            "earth_escape_velocity_km_s": float(physical["v_esc_km_s"]),
            "earth_equilibrium_temperature_k": float(physical["T_eq_K"]),
            "earth_surface_pressure_atm": float(atmosphere["surface_pressure_atm"]),
            "earth_surface_temperature_k": float(atmosphere["surface_temp_K"]),
            "earth_n2_partial_pressure_atm": float(
                atmosphere["composition"].get("N2", {}).get("partial_P_atm", 0.0)
            ),
        },
        "context": {
            "surface_reservoir": {
                key: float(value)
                for key, value in surface_reservoir.items()
                if isinstance(value, (int, float))
            },
            "albedo_bond_initial": float(physical["albedo_bond"]),
            "albedo_bond_effective": float(atmosphere["albedo_bond_eff"]),
            "water_ocean_mass_kg": float(atmosphere["water_ocean_mass_kg"]),
            "tidally_locked": bool(physical["tidally_locked"]),
            "spin_state": physical["spin_state"],
        },
    }


def run_validation_pipeline(manifest_path: str | None = None) -> dict:
    """Run the validation pipeline and return a structured report."""
    manifest = load_manifest(manifest_path)
    scenarios = manifest["scenarios"]

    actuals = {
        "galaxy": _run_galaxy_scenario(scenarios["galaxy"]),
        "stellar": _run_stellar_scenario(scenarios["stellar"]),
        "planet": _run_planet_scenario(scenarios["planet"]),
    }

    results = []
    for metric in manifest["metrics"]:
        actual = actuals[metric["domain"]]["metrics"][metric["key"]]
        results.append(evaluate_metric(metric, actual))

    summary = summarize_results(results)
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_name": manifest["metadata"]["name"],
            "manifest_label": manifest["metadata"]["label"],
            "manifest_version": manifest["metadata"]["version"],
            "manifest_path": manifest["_path"],
        },
        "summary": summary,
        "actuals": actuals,
        "results": results,
    }
