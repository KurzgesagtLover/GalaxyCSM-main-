"""
Configuration and constants for Galactic Chemical Evolution model.

Contains: tracked elements, BBN abundances, solar abundances,
and default simulation parameters.
"""
import os

import numpy as np

# ============================================================
# Tracked elements (~22)
# ============================================================
ELEMENTS = [
    'H', 'He',                                        # Primordial
    'C', 'N', 'O', 'P', 'S',                          # CHNOPS
    'Fe', 'Zn', 'Mn', 'Co', 'Cu', 'Ni', 'Mo',         # Biogenic metals
    'Se', 'V', 'W',                                    # Trace metals
    'Mg', 'Eu', 'Ba',                                  # Channel tracers
    'Si', 'Al',                                        # Planet formation
]

N_ELEMENTS = len(ELEMENTS)
EL_IDX = {el: i for i, el in enumerate(ELEMENTS)}

ATOMIC_MASS = {
    'H': 1.008, 'He': 4.003, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.065, 'Fe': 55.845, 'Zn': 65.38, 'Mn': 54.938,
    'Co': 58.933, 'Cu': 63.546, 'Ni': 58.693, 'Mo': 95.95, 'Se': 78.971,
    'V': 50.942, 'W': 183.84, 'Mg': 24.305, 'Eu': 151.964, 'Ba': 137.327,
    'Si': 28.086, 'Al': 26.982,
}

# ============================================================
# BBN abundances (mass fractions) — Planck 2018
# ============================================================
BBN_ABUNDANCE = np.zeros(N_ELEMENTS)
BBN_ABUNDANCE[EL_IDX['H']]  = 0.7529
BBN_ABUNDANCE[EL_IDX['He']] = 0.2471

# ============================================================
# Solar abundances — Asplund et al. 2009, 12+log(N_X/N_H)
# ============================================================
SOLAR_LOG_EPS = {
    'H': 12.00, 'He': 10.93, 'C': 8.43, 'N': 7.83, 'O': 8.69,
    'P': 5.41,  'S': 7.12,  'Fe': 7.50, 'Zn': 4.56, 'Mn': 5.43,
    'Co': 4.99, 'Cu': 4.19, 'Ni': 6.22, 'Mo': 1.88, 'Se': 3.30,
    'V': 3.93,  'W': 0.85,  'Mg': 7.60, 'Eu': 0.52, 'Ba': 2.18,
    'Si': 7.51, 'Al': 6.45,
}

def _compute_solar_mass_fractions():
    n_ratio = {el: 10.0**(SOLAR_LOG_EPS[el] - 12.0) for el in ELEMENTS}
    total = sum(n_ratio[el] * ATOMIC_MASS[el] for el in ELEMENTS)
    arr = np.zeros(N_ELEMENTS)
    for el in ELEMENTS:
        arr[EL_IDX[el]] = n_ratio[el] * ATOMIC_MASS[el] / total
    return arr

SOLAR_X = _compute_solar_mass_fractions()
Z_SUN = 1.0 - SOLAR_X[EL_IDX['H']] - SOLAR_X[EL_IDX['He']]
Z_SUN_TRACKED = Z_SUN
Z_SUN_TOTAL = 0.0134
TRACKED_Z_TO_TOTAL_SCALE = Z_SUN_TOTAL / max(Z_SUN_TRACKED, 1e-12)

DEFAULT_GCE_T_MAX = 20.0
DEFAULT_VIEW_T_MAX = 10000.0
GCE_T_MAX_MIN = 0.1
GCE_T_MAX_MAX = 100.0

STELLAR_MODEL_OPTIONS = ('heuristic', 'precise', 'auto')
DEFAULT_STELLAR_MODEL = 'auto'
STELLAR_TRACKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'stellar_tracks')
)
DEFAULT_PRECISE_TRACK_PACK = os.path.join(
    STELLAR_TRACKS_DIR,
    'demo_precise_tracks.json',
)

# ============================================================
# Default simulation parameters
# ============================================================
DEFAULT_PARAMS = {
    # Time
    't_max': DEFAULT_GCE_T_MAX,      # Gyr
    # Spatial
    'r_min': 0.5, 'r_max': 20.0, 'dr': 1.0,   # kpc
    'radial_mixing_kpc2_gyr': 0.0,            # optional annulus-to-annulus gas mixing
    # Stellar radial migration / kinematics
    'stellar_migration_drift_coeff_kpc': 0.18,
    'stellar_migration_max_shift_kpc': 0.85,
    'stellar_migration_sigma_floor_kpc': 0.04,
    'stellar_migration_sigma_sqrt_age_kpc': 0.08,
    'stellar_migration_resonance_strength': 1.0,
    'stellar_sigma_r_floor_km_s': 8.0,
    'stellar_sigma_r_sqrt_age_km_s': 9.0,
    'stellar_sigma_z_floor_km_s': 5.0,
    'stellar_sigma_z_sqrt_age_km_s': 4.0,
    'stellar_vcirc_max_km_s': 232.0,
    'stellar_vcirc_turnover_kpc': 1.5,
    'stellar_eccentricity_max': 0.35,
    # Star formation (Kennicutt-Schmidt)
    'sfr_efficiency': 0.08,
    'sfr_exponent': 1.4,
    # Gas infall — double exponential
    'infall_tau_thick': 1.0,         # Gyr
    'infall_tau_thin': 7.0,          # Gyr
    'infall_sigma_thick0': 55.0,      # M☉/pc² normalization
    'infall_sigma_thin0': 320.0,
    'infall_rd': 2.0,                # disk scale length kpc
    # Outflow
    'outflow_eta': 1.06,             # mass-loading factor
    # r-process calibration
    'yield_r_multiplier': 1.1,       # keeps solar-zone [Eu/Fe] near the solar anchor
    'yield_s_multiplier': 1.0,
    'yield_ia_multiplier': 1.0,
    'agb_frequency_multiplier': 1.0,
    # Type Ia SN
    'ia_N_per_Msun': 2.0e-3,
    'ia_t_min': 0.15,                # Gyr — sharper knee
    'ia_dtd_slope': -1.0,
    # Neutron star mergers (r-process)
    'nsm_N_per_Msun': 3.0e-5,
    'nsm_t_min': 0.01,               # Gyr
    'nsm_dtd_slope': -1.0,
    'nsm_ejecta': 0.03,              # M☉ per event
    # Collapsar / Jet-SNe (r-process)
    # Keep collapsars sub-dominant by default so Eu is not overproduced once
    # NSM enrichment is also enabled in the same run.
    'collapsar_frac': 0.001,         # fraction of CCSNe that are collapsars
    'collapsar_ejecta': 0.05,        # M☉ r-process ejecta per event
    # IMF
    'imf': 'kroupa',
}


def coerce_solver_params(params=None):
    """Merge user parameters with defaults and enforce safe solver bounds."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    t_max = float(p['t_max'])
    if not (GCE_T_MAX_MIN <= t_max <= GCE_T_MAX_MAX):
        raise ValueError(
            f"t_max must be between {GCE_T_MAX_MIN} and {GCE_T_MAX_MAX} Gyr"
        )

    r_min = float(p['r_min'])
    r_max = float(p['r_max'])
    dr = float(p['dr'])
    radial_mixing = float(p.get('radial_mixing_kpc2_gyr', 0.0))
    if dr <= 0:
        raise ValueError("dr must be positive")
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min")
    if radial_mixing < 0:
        raise ValueError("radial_mixing_kpc2_gyr must be non-negative")

    non_negative_keys = (
        'stellar_migration_drift_coeff_kpc',
        'stellar_migration_max_shift_kpc',
        'stellar_migration_sigma_floor_kpc',
        'stellar_migration_sigma_sqrt_age_kpc',
        'stellar_migration_resonance_strength',
        'stellar_sigma_r_floor_km_s',
        'stellar_sigma_r_sqrt_age_km_s',
        'stellar_sigma_z_floor_km_s',
        'stellar_sigma_z_sqrt_age_km_s',
        'stellar_vcirc_max_km_s',
        'stellar_vcirc_turnover_kpc',
        'stellar_eccentricity_max',
    )
    for key in non_negative_keys:
        p[key] = float(p.get(key, DEFAULT_PARAMS[key]))
        if p[key] < 0:
            raise ValueError(f"{key} must be non-negative")

    p['t_max'] = t_max
    p['r_min'] = r_min
    p['r_max'] = r_max
    p['dr'] = dr
    p['radial_mixing_kpc2_gyr'] = radial_mixing
    return p


def coerce_stellar_model(model=None):
    """Resolve a stellar-track engine selector."""
    resolved = DEFAULT_STELLAR_MODEL if model is None else str(model).strip().lower()
    if resolved not in STELLAR_MODEL_OPTIONS:
        allowed = ', '.join(STELLAR_MODEL_OPTIONS)
        raise ValueError(f"stellar_model must be one of: {allowed}")
    return resolved
