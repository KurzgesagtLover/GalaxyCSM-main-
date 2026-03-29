"""
Helpers for approximate stellar / cluster radial migration.

The stellar migration model is intentionally lightweight but now separates:
- churning: changes in guiding radius / angular momentum
- blurring: epicyclic excursions around the guiding center
- heating: age-dependent velocity dispersion and vertical puffing
"""

from __future__ import annotations

import numpy as np

DEFAULT_OPEN_CLUSTER_MEAN_BIRTH_OFFSET_KPC = 0.75
DEFAULT_OPEN_CLUSTER_SIGMA_KPC = 0.25

DEFAULT_STELLAR_DRIFT_COEFF_KPC = 0.18
DEFAULT_STELLAR_MAX_DRIFT_KPC = 0.85
DEFAULT_STELLAR_SIGMA_FLOOR_KPC = 0.04
DEFAULT_STELLAR_SIGMA_SQRT_AGE_KPC = 0.08
DEFAULT_STELLAR_SIGMA_R_FLOOR_KM_S = 8.0
DEFAULT_STELLAR_SIGMA_R_SQRT_AGE_KM_S = 9.0
DEFAULT_STELLAR_SIGMA_Z_FLOOR_KM_S = 5.0
DEFAULT_STELLAR_SIGMA_Z_SQRT_AGE_KM_S = 4.0
DEFAULT_STELLAR_VCIRC_MAX_KM_S = 232.0
DEFAULT_STELLAR_VCIRC_TURNOVER_KPC = 1.5
DEFAULT_STELLAR_RESONANCE_STRENGTH = 1.0
DEFAULT_STELLAR_ECCENTRICITY_MAX = 0.35
DEFAULT_STELLAR_Z_FLOOR_KPC = 0.05
DEFAULT_STELLAR_Z_PER_SIGMA_Z_KPC = 0.0035


def gaussian_smooth_radial_profile(radii_kpc, values, sigma_kpc):
    """Apply an edge-preserving Gaussian blur on a 1D radial profile."""
    radii = np.asarray(radii_kpc, dtype=float)
    profile = np.asarray(values, dtype=float)
    sigma = float(max(sigma_kpc, 0.0))
    if profile.ndim != 1:
        raise ValueError("values must be a 1D radial profile")
    if profile.size < 2 or sigma <= 0.0:
        return profile.copy()

    dr = float(np.median(np.diff(radii)))
    if not np.isfinite(dr) or dr <= 0.0:
        return profile.copy()

    sigma_pix = sigma / dr
    half_width = max(int(np.ceil(3.0 * sigma_pix)), 1)
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(offsets / max(sigma_pix, 1e-8)))
    kernel /= np.sum(kernel)
    padded = np.pad(profile, (half_width, half_width), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_migration_adjusted_profile(
    radii_kpc,
    gas_phase_profile,
    mean_birth_offset_kpc=DEFAULT_OPEN_CLUSTER_MEAN_BIRTH_OFFSET_KPC,
    sigma_kpc=DEFAULT_OPEN_CLUSTER_SIGMA_KPC,
):
    """Map a birth-radius gas profile onto present-day open-cluster radii."""
    radii = np.asarray(radii_kpc, dtype=float)
    gas_profile = np.asarray(gas_phase_profile, dtype=float)
    smoothed = gaussian_smooth_radial_profile(radii, gas_profile, sigma_kpc)
    shifted_radii = radii + float(mean_birth_offset_kpc)
    return np.interp(
        shifted_radii,
        radii,
        smoothed,
        left=float(smoothed[0]),
        right=float(smoothed[-1]),
    )


def stellar_migration_mean_shift(
    age_gyr,
    drift_coeff_kpc=DEFAULT_STELLAR_DRIFT_COEFF_KPC,
    max_shift_kpc=DEFAULT_STELLAR_MAX_DRIFT_KPC,
):
    """Approximate inward churning bias amplitude for a stellar population."""
    age = np.clip(np.asarray(age_gyr, dtype=float), 0.0, None)
    mean_shift = float(max(drift_coeff_kpc, 0.0)) * np.sqrt(age)
    return np.clip(mean_shift, 0.0, float(max(max_shift_kpc, 0.0)))


def stellar_migration_sigma(
    age_gyr,
    sigma_floor_kpc=DEFAULT_STELLAR_SIGMA_FLOOR_KPC,
    sigma_sqrt_age_kpc=DEFAULT_STELLAR_SIGMA_SQRT_AGE_KPC,
):
    """Base churning diffusion scale before resonance boosting."""
    age = np.clip(np.asarray(age_gyr, dtype=float), 0.0, None)
    return float(max(sigma_floor_kpc, 0.0)) + float(max(sigma_sqrt_age_kpc, 0.0)) * np.sqrt(age)


def circular_velocity_curve(
    radii_kpc,
    v_circ_max_km_s=DEFAULT_STELLAR_VCIRC_MAX_KM_S,
    turnover_kpc=DEFAULT_STELLAR_VCIRC_TURNOVER_KPC,
):
    """A simple Milky Way-like rising-then-flat circular-speed curve."""
    radii = np.clip(np.asarray(radii_kpc, dtype=float), 0.0, None)
    vmax = float(max(v_circ_max_km_s, 1.0))
    turn = float(max(turnover_kpc, 1e-3))
    return vmax * (1.0 - np.exp(-radii / turn))


def resonance_diffusion_boost(radii_kpc, resonance_strength=DEFAULT_STELLAR_RESONANCE_STRENGTH):
    """Boost churning near simple bar / spiral resonances."""
    radii = np.asarray(radii_kpc, dtype=float)
    strength = float(max(resonance_strength, 0.0))
    if strength <= 0.0:
        return np.ones_like(radii)

    bar_cr = np.exp(-0.5 * np.square((radii - 4.5) / 0.9))
    bar_olr = np.exp(-0.5 * np.square((radii - 7.0) / 1.0))
    spiral_cr = np.exp(-0.5 * np.square((radii - 8.5) / 1.4))
    return 1.0 + strength * (0.35 * bar_cr + 0.55 * bar_olr + 0.30 * spiral_cr)


def stellar_velocity_dispersion(
    age_gyr,
    sigma_r_floor_km_s=DEFAULT_STELLAR_SIGMA_R_FLOOR_KM_S,
    sigma_r_sqrt_age_km_s=DEFAULT_STELLAR_SIGMA_R_SQRT_AGE_KM_S,
    sigma_z_floor_km_s=DEFAULT_STELLAR_SIGMA_Z_FLOOR_KM_S,
    sigma_z_sqrt_age_km_s=DEFAULT_STELLAR_SIGMA_Z_SQRT_AGE_KM_S,
):
    """Age-velocity dispersion relation for disk stars."""
    age = np.clip(np.asarray(age_gyr, dtype=float), 0.0, None)
    sigma_r = float(max(sigma_r_floor_km_s, 0.0)) + float(max(sigma_r_sqrt_age_km_s, 0.0)) * np.sqrt(age)
    sigma_phi = sigma_r / np.sqrt(2.0)
    sigma_z = float(max(sigma_z_floor_km_s, 0.0)) + float(max(sigma_z_sqrt_age_km_s, 0.0)) * np.sqrt(age)
    return sigma_r, sigma_phi, sigma_z


def sample_stellar_migration_state(
    birth_radii_kpc,
    age_gyr,
    rng,
    r_min_kpc,
    r_max_kpc,
    drift_coeff_kpc=DEFAULT_STELLAR_DRIFT_COEFF_KPC,
    max_shift_kpc=DEFAULT_STELLAR_MAX_DRIFT_KPC,
    sigma_floor_kpc=DEFAULT_STELLAR_SIGMA_FLOOR_KPC,
    sigma_sqrt_age_kpc=DEFAULT_STELLAR_SIGMA_SQRT_AGE_KPC,
    resonance_strength=DEFAULT_STELLAR_RESONANCE_STRENGTH,
    sigma_r_floor_km_s=DEFAULT_STELLAR_SIGMA_R_FLOOR_KM_S,
    sigma_r_sqrt_age_km_s=DEFAULT_STELLAR_SIGMA_R_SQRT_AGE_KM_S,
    sigma_z_floor_km_s=DEFAULT_STELLAR_SIGMA_Z_FLOOR_KM_S,
    sigma_z_sqrt_age_km_s=DEFAULT_STELLAR_SIGMA_Z_SQRT_AGE_KM_S,
    v_circ_max_km_s=DEFAULT_STELLAR_VCIRC_MAX_KM_S,
    v_circ_turnover_kpc=DEFAULT_STELLAR_VCIRC_TURNOVER_KPC,
    eccentricity_max=DEFAULT_STELLAR_ECCENTRICITY_MAX,
    z_floor_kpc=DEFAULT_STELLAR_Z_FLOOR_KPC,
    z_per_sigma_z_kpc=DEFAULT_STELLAR_Z_PER_SIGMA_Z_KPC,
):
    """Sample present-day guiding radii, radii, and kinematics for stars."""
    birth_radii = np.asarray(birth_radii_kpc, dtype=float)
    age = np.clip(np.asarray(age_gyr, dtype=float), 0.0, None)
    r_min = float(r_min_kpc)
    r_max = float(r_max_kpc)

    birth_guiding_radii = birth_radii.copy()
    mean_shift = stellar_migration_mean_shift(
        age,
        drift_coeff_kpc=drift_coeff_kpc,
        max_shift_kpc=max_shift_kpc,
    )
    base_churning_sigma = stellar_migration_sigma(
        age,
        sigma_floor_kpc=sigma_floor_kpc,
        sigma_sqrt_age_kpc=sigma_sqrt_age_kpc,
    )
    churning_sigma = base_churning_sigma * resonance_diffusion_boost(
        birth_guiding_radii,
        resonance_strength=resonance_strength,
    )

    guiding_radii = birth_guiding_radii - mean_shift + rng.normal(0.0, churning_sigma, size=birth_radii.shape)
    guiding_radii = np.clip(guiding_radii, r_min, r_max)
    churning_delta = guiding_radii - birth_guiding_radii

    circular_velocity = circular_velocity_curve(
        guiding_radii,
        v_circ_max_km_s=v_circ_max_km_s,
        turnover_kpc=v_circ_turnover_kpc,
    )
    sigma_r, sigma_phi, sigma_z = stellar_velocity_dispersion(
        age,
        sigma_r_floor_km_s=sigma_r_floor_km_s,
        sigma_r_sqrt_age_km_s=sigma_r_sqrt_age_km_s,
        sigma_z_floor_km_s=sigma_z_floor_km_s,
        sigma_z_sqrt_age_km_s=sigma_z_sqrt_age_km_s,
    )
    orbital_eccentricity = np.clip(
        sigma_r / np.maximum(np.sqrt(2.0) * circular_velocity, 1e-8),
        0.0,
        float(max(eccentricity_max, 0.0)),
    )

    radial_phase = rng.uniform(0.0, 2.0 * np.pi, size=birth_radii.shape)
    radial_amplitude = orbital_eccentricity * guiding_radii
    blurring_delta = radial_amplitude * np.cos(radial_phase)
    current_radii = np.clip(guiding_radii + blurring_delta, r_min, r_max)
    blurring_delta = current_radii - guiding_radii

    epicyclic_freq = np.sqrt(2.0) * circular_velocity / np.maximum(guiding_radii, 1e-8)
    v_r = -epicyclic_freq * radial_amplitude * np.sin(radial_phase)
    angular_momentum = guiding_radii * circular_velocity
    v_phi = angular_momentum / np.maximum(current_radii, 1e-8)
    v_z = rng.normal(0.0, sigma_z, size=birth_radii.shape)
    vertical_scale_height = float(max(z_floor_kpc, 0.0)) + float(max(z_per_sigma_z_kpc, 0.0)) * sigma_z
    current_z = rng.normal(0.0, vertical_scale_height, size=birth_radii.shape)

    return {
        'birth_guiding_radius_kpc': birth_guiding_radii,
        'guiding_radius_kpc': guiding_radii,
        'guiding_radius_delta_kpc': churning_delta,
        'current_radius_kpc': current_radii,
        'radial_migration_delta_kpc': current_radii - birth_radii,
        'radial_churning_delta_kpc': churning_delta,
        'radial_blurring_delta_kpc': blurring_delta,
        'radial_migration_mean_shift_kpc': mean_shift,
        'radial_migration_sigma_kpc': churning_sigma,
        'orbital_eccentricity': orbital_eccentricity,
        'sigma_R_km_s': sigma_r,
        'sigma_phi_km_s': sigma_phi,
        'sigma_z_km_s': sigma_z,
        'circular_velocity_km_s': circular_velocity,
        'angular_momentum_kpc_km_s': angular_momentum,
        'v_R_km_s': v_r,
        'v_phi_km_s': v_phi,
        'v_z_km_s': v_z,
        'vertical_scale_height_kpc': vertical_scale_height,
        'current_z_kpc': current_z,
    }


def sample_present_day_radii(
    birth_radii_kpc,
    age_gyr,
    rng,
    r_min_kpc,
    r_max_kpc,
    drift_coeff_kpc=DEFAULT_STELLAR_DRIFT_COEFF_KPC,
    max_shift_kpc=DEFAULT_STELLAR_MAX_DRIFT_KPC,
    sigma_floor_kpc=DEFAULT_STELLAR_SIGMA_FLOOR_KPC,
    sigma_sqrt_age_kpc=DEFAULT_STELLAR_SIGMA_SQRT_AGE_KPC,
):
    """Backward-compatible wrapper returning only radii and effective scales."""
    state = sample_stellar_migration_state(
        birth_radii_kpc,
        age_gyr,
        rng,
        r_min_kpc,
        r_max_kpc,
        drift_coeff_kpc=drift_coeff_kpc,
        max_shift_kpc=max_shift_kpc,
        sigma_floor_kpc=sigma_floor_kpc,
        sigma_sqrt_age_kpc=sigma_sqrt_age_kpc,
    )
    return (
        state['current_radius_kpc'],
        state['radial_migration_mean_shift_kpc'],
        state['radial_migration_sigma_kpc'],
    )
