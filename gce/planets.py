"""
Comprehensive rocky planet model with:
- 20-element geochemical partitioning (Wood+2006, McDonough 2003)
- Volatile depletion by condensation temperature (Lodders 2003)
- Full physical properties (mass-radius, gravity, oblateness, ESI)
- Atmosphere model with molecular speciation (Schaefer & Fegley 2010)
- Core thermal evolution with radiogenic heating (Turcotte & Schubert 2002)
- Core viscosity (Poirier 1988, de Wijs+1998)
- Magnetic field via dynamo scaling (Christensen+2009)
- Tidal heating (Peale+1979)
"""
import numpy as np
from .config import ELEMENTS, EL_IDX, SOLAR_X, N_ELEMENTS

# ============================================================
# CONSTANTS
# ============================================================
M_EARTH  = 5.972e24     # kg
R_EARTH  = 6.371e6      # m
G_CONST  = 6.674e-11    # m³/(kg·s²)
SIGMA_SB = 5.670e-8     # W/(m²·K⁴)
K_BOLTZ  = 1.381e-23    # J/K
R_GAS    = 8.314        # J/(mol·K)
AU_M     = 1.496e11     # m
L_SUN    = 3.828e26     # W
M_SUN    = 1.989e30     # kg

# ============================================================
# PARTITION COEFFICIENTS — Wood+2006, McDonough 2003, Rubie+2015
# (D_core, D_mantle, D_crust)
# ============================================================
PARTITION_COEFF = {
    'H':  (0.01, 0.05, 0.02), 'He': (0.00, 0.00, 0.00),
    'C':  (0.10, 0.60, 0.30), 'N':  (0.05, 0.50, 0.45),
    'O':  (0.05, 0.85, 0.10), 'P':  (0.50, 0.30, 0.20),
    'S':  (0.65, 0.25, 0.10), 'Fe': (0.85, 0.13, 0.02),
    'Zn': (0.10, 0.50, 0.40), 'Mn': (0.02, 0.70, 0.28),
    'Co': (0.70, 0.25, 0.05), 'Cu': (0.40, 0.35, 0.25),
    'Ni': (0.90, 0.08, 0.02), 'Mo': (0.75, 0.15, 0.10),
    'Se': (0.55, 0.30, 0.15), 'V':  (0.08, 0.72, 0.20),
    'W':  (0.80, 0.15, 0.05), 'Mg': (0.00, 0.85, 0.15),
    'Eu': (0.01, 0.60, 0.39), 'Ba': (0.00, 0.30, 0.70),
    'Si': (0.08, 0.72, 0.20), 'Al': (0.01, 0.45, 0.54),
}

# ============================================================
# 50% CONDENSATION TEMPERATURES (K) — Lodders 2003
# ============================================================
VOLATILE_T50 = {
    'H': 182, 'He': 3, 'C': 40, 'N': 123, 'O': 1350,
    'P': 1229, 'S': 664, 'Fe': 1334, 'Zn': 726, 'Mn': 1158,
    'Co': 1352, 'Cu': 1037, 'Ni': 1353, 'Mo': 1590, 'Se': 697,
    'V': 1429, 'W': 1789, 'Mg': 1354, 'Eu': 1356, 'Ba': 1455,
    'Si': 1310, 'Al': 1653,
}

# ============================================================
# RADIOGENIC ISOTOPES — Turcotte & Schubert 2002
# (half_life_gyr, heat_W_per_kg, proxy_element, concentration_ppm)
# ============================================================
RADIO_ISOTOPES = {
    'U238':  (4.468, 9.46e-5,  'Fe', 0.020),
    'U235':  (0.704, 5.69e-4,  'Fe', 0.0065),
    'Th232': (14.05, 2.64e-5,  'Fe', 0.079),
    'K40':   (1.250, 3.48e-9,  'Fe', 240.0),
}

# ============================================================
# MOLECULAR SPECIES — Schaefer & Fegley 2010
# ============================================================
MOLECULES = {
    'N2':  {'M': 28.014, 'gamma': 1.40, 'source': 'N',  'ox': 'both'},
    'O2':  {'M': 31.998, 'gamma': 1.40, 'source': 'O',  'ox': 'oxidizing'},
    'CO2': {'M': 44.009, 'gamma': 1.29, 'source': 'C',  'ox': 'oxidizing'},
    'H2O': {'M': 18.015, 'gamma': 1.33, 'source': 'H',  'ox': 'both'},
    'CH4': {'M': 16.043, 'gamma': 1.31, 'source': 'C',  'ox': 'reducing'},
    'CO':  {'M': 28.010, 'gamma': 1.40, 'source': 'C',  'ox': 'both'},
    'SO2': {'M': 64.066, 'gamma': 1.26, 'source': 'S',  'ox': 'oxidizing'},
    'H2S': {'M': 34.081, 'gamma': 1.32, 'source': 'S',  'ox': 'reducing'},
    'NH3': {'M': 17.031, 'gamma': 1.31, 'source': 'N',  'ox': 'reducing'},
    'H2':  {'M':  2.016, 'gamma': 1.41, 'source': 'H',  'ox': 'reducing'},
    'He':  {'M':  4.003, 'gamma': 1.66, 'source': 'He', 'ox': 'both'},
    'Ar':  {'M': 39.948, 'gamma': 1.67, 'source': 'K40','ox': 'both'},
    'Ne':  {'M': 20.180, 'gamma': 1.67, 'source': 'Ne', 'ox': 'both'},
    'NO2': {'M': 46.006, 'gamma': 1.29, 'source': 'N',  'ox': 'oxidizing'},
    'HCN': {'M': 27.026, 'gamma': 1.40, 'source': 'C',  'ox': 'reducing'},
    'PH3': {'M': 33.998, 'gamma': 1.13, 'source': 'P',  'ox': 'reducing'},
    'O3':  {'M': 47.998, 'gamma': 1.29, 'source': 'O',  'ox': 'oxidizing'},
}


# ============================================================
# PLANET GENERATION — Disk-based semi-analytic model
# ============================================================
from .planet_generation import (
    _generate_planets_fast,
    _generate_planets_from_disk,
    generate_planets,
)
from .planet_atmosphere import (
    _gas_giant_atmo,
    _sub_neptune_atmo,
    compute_radiation_defense,
)
from .planet_interior import (
    core_thermal_model,
    core_viscosity,
    magnetic_field,
)
from .moons import build_moon_system
from .rocky_moons import build_rocky_moon_system

# ============================================================
# VOLATILE DEPLETION — Lodders 2003
# ============================================================
def volatile_depletion(bulk_comp, T_eq):
    """Deplete volatile elements based on equilibrium temperature.

    T₅₀ values represent disk condensation temperatures (Lodders 2003).
    For planets with T_eq < ~500K, volatiles were already condensed
    into planetesimals before accretion → near-full retention.
    Depletion only matters for HOT rocky planets (T_eq > 500K)
    that formed close to their star, where the disk was too hot
    for volatile condensation.
    """
    depleted = {}
    for el, frac in bulk_comp.items():
        t50 = VOLATILE_T50.get(el, 1500)

        # Two-regime depletion:
        # 1. T_eq < 500K: "cold accretion" — volatiles were in solids, retained
        # 2. T_eq > 500K: progressive loss as disk temp exceeded condensation T
        # The transition is at ~500K because below this, essentially all
        # volatile species condense as ices/hydrates in the disk.
        if T_eq < 300:
            # Cool planets: near-full retention of everything that condensed
            # Even C (T50=40K) and N (T50=123K) were delivered as
            # organics, NH₃ ice, enstatite, etc. (Marty 2012)
            retention = 1.0
        elif T_eq < 600:
            # Warm zone: gentle depletion of most volatile elements
            # Only extremely volatile elements (He, Ne) affected
            warmth = (T_eq - 300) / 300  # 0 at 300K, 1 at 600K
            if t50 < 100:  # H, C, He, Ne — most volatile
                retention = 1.0 - 0.5 * warmth
            elif t50 < 200:  # N
                retention = 1.0 - 0.3 * warmth
            else:  # everything else: fully retained
                retention = 1.0
        else:
            # Hot rocky planets (T_eq > 600K): strong depletion
            # Sigmoid based on T_eq vs T₅₀
            scale = max(t50 * 0.15, 50)
            retention = 1.0 / (1.0 + np.exp((T_eq - t50) / scale))

        # Floor: trapped in deep mantle/core
        if el in ['C', 'N', 'H']:
            floor = 1e-3
        elif el in ['S']:
            floor = 0.01
        else:
            floor = 1e-4

        retention = max(float(retention), floor)
        depleted[el] = frac * retention

    return depleted


def estimate_equilibrium_temperature(luminosity_lsun, semi_major_au, albedo_bond=0.3):
    """Blackbody equilibrium temperature in K for a planet on a circular orbit."""
    a_au = max(float(semi_major_au), 0.01)
    alb = float(np.clip(albedo_bond, 0.0, 0.95))
    return 278.5 * max(float(luminosity_lsun), 1e-8) ** 0.25 * (1.0 - alb) ** 0.25 / np.sqrt(a_au)


_SPECTRAL_HZ_COEFFS = {
    'recent_venus': (1.776, -1.433e-4, -3.395e-9, 7.636e-12, -1.195e-15),
    'runaway_greenhouse': (1.107, -1.332e-4, -1.580e-8, 8.308e-12, -1.931e-15),
    'moist_greenhouse': (0.990, -8.177e-5, -1.706e-9, 4.324e-12, -6.646e-16),
    'max_greenhouse': (0.356, -6.171e-5, -1.698e-9, 3.198e-12, -5.575e-16),
    'early_mars': (0.320, -5.547e-5, -1.526e-9, 2.874e-12, -5.011e-16),
}


def estimate_spectral_habitable_zone(luminosity_lsun, T_eff):
    """Estimate Kopparapu-style habitable-zone boundaries in AU."""
    lum = max(float(luminosity_lsun), 1e-8)
    t_star = np.clip(float(T_eff), 2600.0, 7200.0) - 5780.0

    def _distance(key):
        s_eff_sun, a, b, c, d = _SPECTRAL_HZ_COEFFS[key]
        s_eff = s_eff_sun + a * t_star + b * t_star**2 + c * t_star**3 + d * t_star**4
        s_eff = max(s_eff, 1e-4)
        return np.sqrt(lum / s_eff)

    hz = {
        'recent_venus_au': round(float(_distance('recent_venus')), 4),
        'runaway_greenhouse_au': round(float(_distance('runaway_greenhouse')), 4),
        'moist_greenhouse_au': round(float(_distance('moist_greenhouse')), 4),
        'max_greenhouse_au': round(float(_distance('max_greenhouse')), 4),
        'early_mars_au': round(float(_distance('early_mars')), 4),
    }
    hz['conservative_inner_au'] = hz['runaway_greenhouse_au']
    hz['conservative_outer_au'] = hz['max_greenhouse_au']
    hz['optimistic_inner_au'] = hz['recent_venus_au']
    hz['optimistic_outer_au'] = hz['early_mars_au']
    return hz

# ============================================================
# PHYSICAL PROPERTIES — Zeng+2016, Valencia+2006
# ============================================================
def compute_physical_properties(mass_earth, ptype, a_AU, star_mass,
                                rot_period_hr, axial_tilt_deg, ecc,
                                age_gyr, L_star_Lsun=None,
                                esi_weights=None):
    """Compute ALL physical properties from the reference panel.

    Returns dict with ~25 physical quantities.
    References: Zeng+2016 (mass-radius), Seager+2007 (interior),
    Barnes & Fortney 2003 (oblateness), Schulze-Makuch+2011 (ESI).
    """
    if L_star_Lsun is None:
        L_star_Lsun = star_mass ** 3.5

    M = mass_earth * M_EARTH       # kg
    L_star = L_star_Lsun * L_SUN   # W

    # --- Radius (Zeng+2016 mass-radius) ---
    if ptype == 'gas_giant':
        R = R_EARTH * 11.2 * (mass_earth / 318) ** -0.04  # Jupiter analog
    elif ptype == 'mini_neptune':
        R = R_EARTH * 3.5 * (mass_earth / 10) ** 0.2
    else:
        R = R_EARTH * mass_earth ** 0.27  # rocky

    # --- Density ---
    V = (4 / 3) * np.pi * R ** 3
    rho = M / V  # kg/m³

    # --- Oblateness (Darwin-Radau, Barnes & Fortney 2003) ---
    omega = 2 * np.pi / (rot_period_hr * 3600)  # rad/s
    q = omega ** 2 * R ** 3 / (G_CONST * M)  # rotational parameter
    f = min(0.5 * q * (1 + 1.5 * (rho / 5515) ** -0.5), 0.3)  # oblateness

    R_eq = R * (1 + f / 3)
    R_pol = R * (1 - 2 * f / 3)
    R_mean = R

    # --- Gravity ---
    g_mean = G_CONST * M / R ** 2
    g_eq = G_CONST * M / R_eq ** 2 * (1 - f)
    g_pol = G_CONST * M / R_pol ** 2 * (1 + f)

    # --- Velocities ---
    v_esc = np.sqrt(2 * G_CONST * M / R)
    v_circ = v_esc / np.sqrt(2)

    # --- Insolation ---
    S = L_star / (4 * np.pi * (a_AU * AU_M) ** 2)  # W/m²
    S_earth = L_SUN / (4 * np.pi * AU_M ** 2)       # 1361 W/m²
    S_rel = S / S_earth  # relative to Earth

    # --- Bond albedo (surface-driven baseline) ---
    # The detailed atmospheric albedo is solved later in compute_atmosphere(),
    # so we do not infer an atmosphere from bulk density here.
    if ptype == 'gas_giant':
        albedo_bond = 0.34
    elif ptype == 'mini_neptune':
        albedo_bond = 0.30
    else:
        if a_AU < 0.1:  # hot rocky, no atmosphere expected
            albedo_bond = 0.10
        else:
            T_eq_prelim = estimate_equilibrium_temperature(L_star_Lsun, a_AU, 0.20)
            a_surf = 0.12  # basaltic surface
            # Temperate rocky planets with mixed cloud/ocean/land cover should
            # sit closer to Earth's Bond albedo than bare basalt.
            if 240 <= T_eq_prelim <= 320:
                a_surf = 0.30
            elif T_eq_prelim < 200:  # ice-covered
                a_surf = 0.5
            elif T_eq_prelim < 273:  # partial ice
                ice_frac = np.clip((273 - T_eq_prelim) / 100, 0, 0.7)
                a_surf = 0.12 * (1 - ice_frac) + 0.7 * ice_frac
            albedo_bond = float(np.clip(a_surf, 0.05, 0.75))

    albedo_geom = albedo_bond * 1.5

    # --- Equilibrium temperature ---
    T_eq = estimate_equilibrium_temperature(L_star_Lsun, a_AU, albedo_bond)
    T_orb_s = 2 * np.pi * np.sqrt((a_AU * AU_M) ** 3 / (G_CONST * star_mass * M_SUN))
    T_sid_s = rot_period_hr * 3600

    # --- Tidal heating (Peale, Cassen & Reynolds 1979) ---
    # Correct formula: dE/dt = -(21/2) * (k₂/Q) * (G*M_star²/R) * (R/a)⁶ * n * e²
    # where M_star = host star mass, a = semi-major axis, n = mean motion
    # Dimensions: [W] = [1] * [N·m²·kg⁻²·kg²/m] * [1] * [s⁻¹] * [1] = [W] ✓
    #
    # k₂: Love number. Fluid core → k₂ ~ 0.3, solid → k₂ ~ 0.02-0.05
    # Segatz+1988: k₂ = 1.5 / (1 + 19μ_eff/(2ρgR))
    # Earth k₂=0.299, Mars k₂=0.169, Moon k₂=0.024
    mu_rigidity = 8e10 if ptype != 'gas_giant' else 0  # Pa, mantle rigidity
    k2 = 1.5 / (1 + 19 * mu_rigidity / (2 * rho * g_mean * R)) if ptype != 'gas_giant' else 0.38
    k2 = np.clip(k2, 0.01, 0.6)

    # Q: tidal quality factor (Efroimsky 2012)
    # Earth Q≈12, Mars Q≈80-100, Moon Q≈36, Jupiter Q≈10^5
    if ptype == 'gas_giant':
        Q = 1e5
    elif ptype == 'mini_neptune':
        Q = 1e4
    else:
        T_interior_est = 4000 * mass_earth**0.3
        Q_ref = 12.0
        T_ref = 4000.0
        E_a_Q = 3e5  # activation energy J/mol
        Q = Q_ref * np.exp(E_a_Q / R_GAS * (1/max(T_interior_est, 1000) - 1/T_ref))
        Q = np.clip(Q, 5, 500)

    n_orb = 2 * np.pi / T_orb_s  # mean motion [rad/s]
    a_m = a_AU * AU_M              # semi-major axis [m]
    M_star_kg = star_mass * M_SUN  # host star mass [kg]
    # Peale+1979 Eq. 1: dE/dt = -(21/2) * (k₂/Q) * (G*M_star²/R) * (R/a)⁶ * n * e²
    E_tidal = (21.0 / 2.0) * (k2 / Q) * (G_CONST * M_star_kg**2 / R) * \
              (R / a_m)**6 * n_orb * ecc**2
    E_tidal_TW = E_tidal * 1e-12

    # --- Tidal locking / spin evolution ---
    # t_lock ~ ω a^6 I Q / (3 G M_*^2 k2 R^5)
    # (Gladman+1996; Barnes 2017 scaling form).
    moi_coeff = 0.25 if ptype == 'gas_giant' else (0.27 if ptype == 'mini_neptune' else 0.33)
    I_moi = moi_coeff * M * R**2
    omega_init = 2 * np.pi / max(T_sid_s, 1.0)
    omega_sync = n_orb
    omega_delta = abs(omega_init - omega_sync)
    if k2 > 0 and Q > 0 and omega_delta > 0:
        t_lock_s = omega_delta * I_moi * Q * a_m**6 / (
            3.0 * G_CONST * M_star_kg**2 * k2 * R**5
        )
    else:
        t_lock_s = np.inf
    t_lock_gyr = t_lock_s / 3.154e16 if np.isfinite(t_lock_s) else np.inf
    tidally_locked = bool(
        ptype in ('rocky', 'hot_rocky', 'mini_neptune')
        and age_gyr >= t_lock_gyr
    )
    rot_effective_s = T_orb_s if tidally_locked else T_sid_s
    if abs(rot_effective_s - T_orb_s) > 1:
        T_solar_s = abs(1.0 / (1.0 / rot_effective_s - 1.0 / T_orb_s))
    else:
        T_solar_s = rot_effective_s
    rotation_effective_hr = rot_effective_s / 3600.0
    if tidally_locked:
        spin_state = 'tidally_locked'
    elif rotation_effective_hr > 240:
        spin_state = 'slow_rotator'
    elif rotation_effective_hr < 12:
        spin_state = 'fast_rotator'
    else:
        spin_state = 'intermediate_rotator'

    # --- ESI (Schulze-Makuch+2011) ---
    ew = esi_weights or {}
    w_r   = ew.get('w_radius', 0.57)
    w_rho = ew.get('w_density', 1.07)
    w_ve  = ew.get('w_escape', 0.70)
    w_t   = ew.get('w_temp', 5.58)
    def _esi(x, x_e, w):
        return (1 - abs((x - x_e) / (x + x_e))) ** w
    esi_r = _esi(R / R_EARTH, 1.0, w_r)
    esi_rho = _esi(rho, 5515, w_rho)
    esi_ve = _esi(v_esc, 11186, w_ve)
    esi_t = _esi(max(T_eq + 33, 100), 288, w_t)
    esi = float((esi_r * esi_rho * esi_ve * esi_t) ** 0.25)

    return {
        'R_eq_km': round(R_eq * 1e-3, 1),
        'R_mean_km': round(R_mean * 1e-3, 1),
        'R_pol_km': round(R_pol * 1e-3, 1),
        'R_eq_Re': round(R_eq / R_EARTH, 4),
        'R_mean_Re': round(R / R_EARTH, 4),
        'oblateness': round(float(f), 5),
        'mass_kg': float(f'{M:.3e}'),
        'density_kg_m3': round(rho, 1),
        'insolation_W_m2': round(float(S), 1),
        'insolation_rel': round(float(S_rel), 4),
        'T_eq_K': round(float(T_eq), 1),
        'ESI': round(esi, 4),
        'albedo_bond': round(float(albedo_bond), 3),
        'albedo_geom': round(float(albedo_geom), 3),
        'rotation_period_hr': round(rot_period_hr, 2),
        'rotation_effective_hr': round(float(rotation_effective_hr), 2),
        'solar_day_hr': round(float(T_solar_s / 3600), 2),
        'axial_tilt_deg': round(axial_tilt_deg, 1),
        'g_eq_m_s2': round(float(g_eq), 3),
        'g_mean_m_s2': round(float(g_mean), 3),
        'g_pol_m_s2': round(float(g_pol), 3),
        'v_esc_km_s': round(float(v_esc * 1e-3), 2),
        'v_circ_km_s': round(float(v_circ * 1e-3), 2),
        'tidal_heating_TW': round(float(E_tidal_TW), 4),
        'tidal_lock_timescale_gyr': round(float(t_lock_gyr), 6) if np.isfinite(t_lock_gyr) else 1e12,
        'tidally_locked': tidally_locked,
        'spin_state': spin_state,
        'age_gyr': round(age_gyr, 4),
        'orbital_period_days': round(float(T_orb_s / 86400), 2),
    }


# ============================================================
# ATMOSPHERE MODEL — Schaefer & Fegley 2010, Zahnle+2010
# ============================================================
def compute_atmosphere(bulk_comp, mass_earth, ptype, T_eq, g_surface,
                       v_esc, age_gyr, oxidation_iw, albedo_bond=0.3,
                       L_star_Lsun=1.0, star_teff=5778, semi_major_au=1.0,
                       star_radius_rsun=None, rotation_period_hr=None,
                       orbital_period_days=None, tidally_locked=False,
                       tidal_heating_TW=0.0, flare_boost=None,
                       surface_relative_humidity=0.65, biotic_o2_atm=None):
    """Compute full atmospheric composition and properties.

    Molecular speciation from C-H-O-N-S-P system based on
    oxidation state (Schaefer & Fegley 2010).
    Photolysis lifetimes from stellar UV template (photolysis module).
    Jeans + XUV escape (Catling & Zahnle 2009, Lammer+2003).
    Greenhouse from CO₂/H₂O/CH₄ (Wordsworth & Pierrehumbert 2013).
    Optional knobs support sub-saturated rocky atmospheres and explicitly
    oxygenated modern-Earth analogues without changing the abiogenic default.
    """
    # === GAS GIANTS / MINI-NEPTUNES ===
    if ptype == 'gas_giant':
        return _gas_giant_atmo(mass_earth)
    if ptype == 'mini_neptune':
        return _sub_neptune_atmo(mass_earth, bulk_comp)

    # === ROCKY PLANETS ===
    R = R_EARTH * mass_earth ** 0.27
    M = mass_earth * M_EARTH
    A_surface = 4 * np.pi * R ** 2
    rotation_period_hr = float(max(rotation_period_hr if rotation_period_hr is not None else 24.0, 0.5))
    orbital_period_days = float(max(orbital_period_days if orbital_period_days is not None else 365.25, 0.05))
    tidal_heating_TW = float(max(tidal_heating_TW, 0.0))
    base_albedo_bond = float(np.clip(albedo_bond, 0.0, 0.95))
    albedo_bond = base_albedo_bond
    surface_relative_humidity = float(np.clip(surface_relative_humidity, 0.05, 1.0))
    biotic_o2_atm = None if biotic_o2_atm is None else float(max(biotic_o2_atm, 0.0))
    slow_rotator = bool(tidally_locked or rotation_period_hr >= 240.0)
    tidal_flux_w_m2 = tidal_heating_TW * 1e12 / max(A_surface, 1.0)

    def _atmosphere_state(local_species_mass):
        atm_mass_local = max(sum(max(v, 0.0) for v in local_species_mass.values()), 1e6)
        P_surface_local = atm_mass_local * g_surface / A_surface
        P_surface_atm_local = P_surface_local / 101325
        P_surface_bar_local = P_surface_local / 1e5

        moles_local = {}
        for sp, mass_kg in local_species_mass.items():
            mol = MOLECULES.get(sp)
            if mol and mass_kg > 0:
                moles_local[sp] = mass_kg / (mol['M'] * 1e-3)

        total_moles_local = max(sum(moles_local.values()), 1.0)
        composition_local = {}
        for sp, n_mol in sorted(moles_local.items(), key=lambda x: -x[1]):
            frac = n_mol / total_moles_local
            if frac > 1e-8:
                composition_local[sp] = {
                    'vol_frac': frac,
                    'pct': frac * 100.0,
                    'partial_P_atm': frac * P_surface_atm_local,
                    'mass_kg': local_species_mass[sp],
                }

        mu_local = sum(MOLECULES[sp]['M'] * moles_local.get(sp, 0.0)
                       for sp in moles_local if sp in MOLECULES) / total_moles_local
        mu_local = max(mu_local, 2.0)

        gamma_avg_local = sum(MOLECULES[sp]['gamma'] * moles_local.get(sp, 0.0)
                              for sp in moles_local if sp in MOLECULES) / total_moles_local
        gamma_avg_local = max(gamma_avg_local, 1.2)

        return {
            'atm_mass': atm_mass_local,
            'P_surface': P_surface_local,
            'P_surface_atm': P_surface_atm_local,
            'P_surface_bar': P_surface_bar_local,
            'moles': moles_local,
            'composition': composition_local,
            'mu': mu_local,
            'gamma_avg': gamma_avg_local,
        }

    def _greenhouse_response(state, surface_temp_ref):
        P_CO2_atm = state['composition'].get('CO2', {}).get('partial_P_atm', 0.0)
        P_H2O_atm = state['composition'].get('H2O', {}).get('partial_P_atm', 0.0)
        P_CH4_atm = state['composition'].get('CH4', {}).get('partial_P_atm', 0.0)
        P_H2_atm = state['composition'].get('H2', {}).get('partial_P_atm', 0.0)
        P_total = state['P_surface_atm']

        dT_co2 = 7.4 * np.log(1 + P_CO2_atm / 2.8e-4) if P_CO2_atm > 0 else 0.0
        if P_CO2_atm > 10:
            dT_co2 += 200 * np.log10(P_CO2_atm / 10)

        dT_h2o_base = 12 * (P_H2O_atm / 0.01) ** 0.3 if P_H2O_atm > 0 else 0.0
        cc_amplification = 1.0 + 1.2 * np.clip((surface_temp_ref - max(T_eq, 180.0)) / 120.0, 0.0, 2.5)
        dT_h2o = min(180.0, dT_h2o_base * cc_amplification) if P_H2O_atm > 0 else 0.0

        dT_ch4 = min(15.0, 3.0 * (P_CH4_atm / 1e-6) ** 0.2) if P_CH4_atm > 0 else 0.0
        dT_h2_cia = min(40.0, 5.0 * P_H2_atm ** 0.4) if P_H2_atm > 0.001 else 0.0
        dT_background = 0.0
        if 0.2 <= P_total <= 10.0 and P_H2O_atm > 1e-4:
            # Dry background gas plus water vapor broadening gives temperate,
            # N2-dominated atmospheres extra greenhouse leverage even when CO2
            # stays low.
            dT_background = min(12.5, 8.5 * np.log1p(P_total / 0.3))

        if P_CO2_atm > 0 and P_H2O_atm > 0:
            ratio = min(P_H2O_atm / max(P_CO2_atm, 1e-8), 100.0)
            f_overlap = 1.0 - 0.12 * np.log(1 + ratio)
            f_overlap = np.clip(f_overlap, 0.6, 1.0)
        else:
            f_overlap = 1.0

        f_broadening = 1.0 + 0.15 * np.log(1 + max(P_total, 0.0))
        f_broadening = min(f_broadening, 2.5)

        # Slow rotators and synchronous planets can sustain optically thick
        # substellar cloud decks that raise planetary albedo near the inner HZ
        # (Yang+2013, Kopparapu+2016). Thin locked atmospheres instead
        # redistribute heat poorly and can suffer nightside cold trapping.
        rotation_cloud_cooling = 0.0
        if slow_rotator and P_total > 0.1 and surface_temp_ref > 240.0:
            cooling_strength = np.clip((surface_temp_ref - 240.0) / 90.0, 0.0, 1.5)
            pressure_term = np.clip(np.log1p(P_total) / np.log(6.0), 0.0, 1.0)
            h2o_term = np.clip(np.log1p(P_H2O_atm / 0.01), 0.0, 2.0)
            rotation_cloud_cooling = (10.0 + 10.0 * pressure_term + 4.0 * h2o_term) * cooling_strength
            if tidally_locked:
                rotation_cloud_cooling *= 1.25
            rotation_cloud_cooling = min(rotation_cloud_cooling, 40.0)

        redistribution_penalty = 0.0
        if tidally_locked and P_total < 0.3:
            thinness = np.clip((0.3 - P_total) / 0.3, 0.0, 1.0)
            cold_bias = np.clip((260.0 - surface_temp_ref) / 60.0, 0.0, 1.0)
            redistribution_penalty = min(30.0, 8.0 + 18.0 * thinness * (0.5 + cold_bias))

        dT_tidal = min(
            80.0,
            tidal_flux_w_m2 / max(4.0 * SIGMA_SB * max(surface_temp_ref, 200.0) ** 3, 1e-8)
        )

        greenhouse_local = max(
            0.0,
            (dT_co2 + dT_h2o) * f_overlap + dT_ch4 + dT_h2_cia + dT_background
            + redistribution_penalty + dT_tidal - rotation_cloud_cooling
        )
        greenhouse_local = min(greenhouse_local * f_broadening, 800.0)
        return {
            'greenhouse': greenhouse_local,
            'P_CO2_atm': P_CO2_atm,
            'P_H2O_atm': P_H2O_atm,
            'cc_amplification': cc_amplification,
            'rotation_cloud_cooling': rotation_cloud_cooling,
            'redistribution_penalty': redistribution_penalty,
            'tidal_heating_K': dT_tidal,
        }

    # --- Element-specific outgassing efficiencies ---
    # Applied to the mobile volatile inventory, not the full mantle+crust
    # elemental reservoir. Calibrated to keep Earth-like rocky planets near
    # plausible long-term atmospheric inventories rather than producing
    # globally common 10^2-10^3 atm N2/CO2 envelopes.
    # References: Marty (2012), Halliday (2013)
    f_age = min(1.0, (age_gyr / 4.5) ** 0.5) * mass_earth ** 0.15
    OUTGAS_EFF = {
        'N': 0.23 * f_age,   # Slightly lower long-term N2 degassing keeps the
                             # modern-Earth anchor near 0.78 atm N2 without
                             # materially changing abiogenic atmospheres.
        'C': 0.05 * f_age,   # BSE C ~120 ppm; most C stays as carbonate in crust
                              # Earth total surface C ~ 0.01 wt% BSE (Dasgupta+2013)
        'H': 0.25 * f_age,   # BSE H ~100 ppm, need ~1.4e21 H₂O (ocean+atm)
        'S': 2e-4 * f_age,   # S mostly in core, small atmospheric fraction
        'O': 0.01 * f_age,   # O in silicates, only trace free O degassed
        'P': 5e-5 * f_age,   # P locked in apatite, tiny atmospheric fraction
    }

    # --- Chemical speciation (oxidation-state dependent) ---
    ox_frac = np.clip((oxidation_iw + 3) / 6, 0, 1)

    C_bulk = bulk_comp.get('C', 0)
    N_bulk = bulk_comp.get('N', 0)
    H_bulk = bulk_comp.get('H', 0)
    S_bulk = bulk_comp.get('S', 0)
    P_bulk = bulk_comp.get('P', 0)

    # The mantle+crust reservoir can contain much more total C/N than the
    # volatile-bearing portion that actually exchanges with the surface over
    # Gyr timescales. Keep low-BSE inventories available to degas, but strongly
    # sublinearize high reservoir fractions so late-veneer-rich rocky planets do
    # not routinely build 10^2-10^3 atm N2/CO2 atmospheres.
    c_mobile_cap = 1.5e-4 * mass_earth ** 0.25
    n_mobile_cap = 2.5e-6 * mass_earth ** 0.2
    c_redox_factor = 0.85 + 0.30 * ox_frac
    n_redox_factor = 0.75 + 0.50 * ox_frac
    C_mobile = min(C_bulk, c_mobile_cap + 0.03 * C_bulk) * c_redox_factor
    N_mobile = min(N_bulk, n_mobile_cap + 0.002 * N_bulk) * n_redox_factor
    H_mobile = H_bulk
    S_mobile = S_bulk * (0.03 + 0.04 * (1.0 - ox_frac))
    P_mobile = P_bulk * 0.02

    species_mass = {}
    degas_time = max(age_gyr * 1e9, 1e6)

    # ---- STABLE species (photolytically resistant) -- accumulate over time ----
    C_total = C_mobile * M * OUTGAS_EFF['C']
    # CO₂: dominant C-bearing volcanic gas even at low fO₂ (Gaillard+2014)
    # At IW-2 (Earth): mix of CO₂ and CO/CH₄. At IW+1 (Venus): nearly all CO₂
    # Even at IW-6 (reducing): ~10% of C degasses as CO₂ via equilibration
    co2_frac = max(0.10, ox_frac * 0.85 + 0.10)  # minimum 10%, max ~95%
    species_mass['CO2'] = C_total * co2_frac * (44.009 / 12.011)

    N_total = N_mobile * M * OUTGAS_EFF['N']
    # N₂ is thermally very stable — always dominant N species in volcanic gas
    species_mass['N2'] = N_total * (0.55 + 0.35 * ox_frac) * (28.014 / 14.007)

    H_total = H_mobile * M * OUTGAS_EFF['H']
    # H₂O fraction: oxidizing → more H₂O, reducing → more H₂
    h2o_frac = max(0.20, 0.3 + 0.5 * ox_frac)  # 20-80%
    species_mass['H2O'] = H_total * h2o_frac * (18.015 / 1.008)
    species_mass['H2']  = H_total * (1 - h2o_frac) * 0.5 * (2.016 / 1.008)

    # ---- UV-LABILE species: steady-state M = source_rate × τ_photo ----
    # Photolysis lifetimes now computed from stellar UV template (photolysis module)
    from .photolysis import (
        build_uv_template,
        compute_photolysis,
        estimate_flare_boost,
        integrated_xuv_history,
    )

    C_rate = C_total / degas_time
    N_rate = N_total / degas_time
    H_rate = H_total / degas_time
    S_total = S_mobile * M * OUTGAS_EFF['S']
    S_rate = S_total / degas_time
    P_total = P_mobile * M * OUTGAS_EFF['P']
    P_rate = P_total / degas_time
    T_surf_est = max(T_eq + 20, 200)

    # Source masses (before photolysis steady-state) for column density estimate
    labile_sources = {
        'CH4': C_rate * (1 - ox_frac) * 0.7 * (16.043/12.011) * degas_time * 0.001,
        'CO': C_rate * 0.05 * (28.010/12.011) * degas_time * 0.001,
        'HCN': C_rate * (1 - ox_frac) * 0.02 * (27.026/12.011) * degas_time * 0.001,
        'NH3': N_rate * (1 - ox_frac) * 0.1 * (17.031/14.007) * degas_time * 0.001,
        'NO2': N_rate * ox_frac * 0.005 * (46.006/14.007) * degas_time * 0.001,
        'SO2': S_rate * ox_frac * 0.5 * (64.066/32.065) * degas_time * 0.001,
        'H2S': S_rate * (1 - ox_frac) * 0.3 * (34.081/32.065) * degas_time * 0.001,
        'PH3': P_rate * (1 - ox_frac) * 0.5 * (33.998/30.974) * degas_time * 0.001,
    }
    # Combine stable + labile for column density calculation
    all_species_for_cols = dict(species_mass)
    all_species_for_cols.update(labile_sources)

    # Scale height estimate for column density computation
    mu_est = 28.0  # N₂-dominated estimate
    R_gas = 8.314
    H_est = R_gas * T_surf_est / (mu_est * 1e-3 * g_surface)

    # Compute photolysis rates from stellar UV template
    photo_result = compute_photolysis(
        T_eff=star_teff, age_gyr=age_gyr, semi_major_au=semi_major_au,
        species_mass_kg=all_species_for_cols,
        R_planet_m=R, g_surface=g_surface, scale_height_m=H_est,
        flare_boost=estimate_flare_boost(star_teff, age_gyr) if flare_boost is None else flare_boost,
        luminosity_lsun=L_star_Lsun, radius_rsun=star_radius_rsun,
    )

    # Extract column-integrated τ for each labile species (yr)
    def _get_tau(sp, default_yr):
        """Get column-integrated photolysis lifetime, with fallback."""
        sp_data = photo_result['species'].get(sp, {})
        tau = sp_data.get('tau_column_yr', default_yr)
        if tau <= 0 or tau > 1e10:
            tau = default_yr
        return tau

    # Steady-state: M = source_rate × τ_column
    species_mass['CH4'] = C_rate * (1-ox_frac) * 0.7 * (16.043/12.011) * _get_tau('CH4', 9.1)
    species_mass['CO']  = C_rate * 0.05 * (28.010/12.011) * _get_tau('CO', 0.3)
    species_mass['HCN'] = C_rate * (1-ox_frac) * 0.02 * (27.026/12.011) * _get_tau('HCN', 3.0)
    species_mass['NH3'] = N_rate * (1-ox_frac) * 0.1 * (17.031/14.007) * _get_tau('NH3', 0.02)
    species_mass['NO2'] = N_rate * ox_frac * 0.005 * (46.006/14.007) * _get_tau('NO2', 0.003)
    species_mass['SO2'] = S_rate * ox_frac * 0.5 * (64.066/32.065) * _get_tau('SO2', 0.03)
    species_mass['H2S'] = S_rate * (1-ox_frac) * 0.3 * (34.081/32.065) * _get_tau('H2S', 0.005)
    species_mass['PH3'] = P_rate * (1-ox_frac) * 0.5 * (33.998/30.974) * _get_tau('PH3', 0.001)

    # Noble gases — absolute amounts from radioactive decay and primordial trapping
    # Ar-40: from K-40 decay (t_{1/2}=1.25 Gyr). Earth: 6.6e16 kg Ar from
    #   ~240 ppm K in BSE → scale with planet mass, K abundance, and age.
    K_bulk = bulk_comp.get('K', 0.00024)  # K mass fraction (BSE ~240 ppm)
    K40_frac = 0.000117  # K-40 / K total
    K40_total = K_bulk * M * K40_frac  # kg of K-40
    t_half_K40 = 1.25  # Gyr
    K40_decayed = K40_total * (1 - 2**(-age_gyr / t_half_K40))
    # Only ~10.7% of K-40 decays produce Ar-40; ~3% degassed to atmosphere
    Ar40_produced = K40_decayed * 0.107 * (40/40)  # mass ratio ≈ 1
    degas_noble = 0.03 * f_age  # degassing efficiency
    species_mass['Ar'] = Ar40_produced * degas_noble  # Earth: ~6.6e16 kg

    # He-4: from U/Th decay, but mostly escapes — set small initial amount
    species_mass['He'] = Ar40_produced * degas_noble * 0.003  # trace

    # Ne-20: primordial, degassed from mantle. Earth: ~6.5e13 kg
    # Ne/Ar ratio in Earth's atmosphere ≈ 0.001 (Marty 2012)
    # Scale Ne from Ar production rather than from total mass
    species_mass['Ne'] = species_mass.get('Ar', 0) * 0.001

    # --- Atmospheric escape (Jeans + energy-limited XUV) ---
    # References:
    #   Hunten (1973)         — diffusion-limited escape
    #   Catling & Zahnle 2009 — Jeans escape formalism
    #   Lammer+2003           — energy-limited escape
    #   Owen & Wu (2017)      — XUV-driven mass loss
    #   Ribas+2005            — XUV luminosity evolution L_XUV ∝ t^(-1.23)
    #
    # Two regimes:
    #   1. Energy-limited (XUV-driven): dominant for light species (H, H₂, He)
    #      during early stellar evolution when L_XUV is high.
    #      dM/dt = ε * π * R_XUV² * F_XUV / (G * M_planet / R_planet)
    #   2. Jeans (thermal): always active, dominates for heavier species
    #      after XUV flux declines.
    # -----------------------------------------------------------
    T_exo = max(T_eq * 3 + 500, 800)
    T_exo = min(T_exo, 8000)

    xuv_history = integrated_xuv_history(
        star_teff, age_gyr, semi_major_au, luminosity_lsun=L_star_Lsun
    )
    t_sat_gyr = xuv_history['t_sat_gyr']
    F_xuv_sat = xuv_history['F_xuv_sat_erg_cm2_s']
    integrated_xuv_fluence = xuv_history['xuv_fluence_erg_cm2']
    # Efficiency factor (Owen & Wu 2017): ε ~ 0.1-0.3
    eps_xuv = 0.15

    # XUV absorption radius ~ 1.1-1.5 * R_planet
    R_xuv = 1.3 * R

    # Gravitational binding potential (escape energy per unit mass)
    phi_grav = G_CONST * M / R  # J/kg

    for sp in list(species_mass.keys()):
        mol = MOLECULES.get(sp)
        if mol is None:
            continue
        m_mol = mol['M'] * 1.66e-27  # kg per molecule
        m_mol_amu = mol['M']

        # === 1. Jeans escape ===
        # Jeans parameter λ = (v_esc / v_thermal)²
        v_th = np.sqrt(2 * K_BOLTZ * T_exo / m_mol)
        lam = (v_esc / v_th) ** 2 if v_th > 0 else 100

        # Jeans escape flux: Φ_J = n_exo * v_th * (1 + λ) * exp(-λ) / (2√π)
        # Simplified: fraction lost per Gyr
        if lam < 1.5:
            # Essentially unbound
            jeans_loss_rate = 10.0  # lose everything in < 0.1 Gyr
        elif lam < 6:
            jeans_loss_rate = 2.0 * np.exp(-lam) * (1 + lam)  # per Gyr
        elif lam < 15:
            jeans_loss_rate = 0.5 * np.exp(-lam) * (1 + lam)  # per Gyr, slow
        else:
            jeans_loss_rate = 0.0  # negligible for heavy molecules

        # Continuous loss over age
        jeans_retention = np.exp(-jeans_loss_rate * age_gyr)
        jeans_retention = np.clip(jeans_retention, 1e-6, 1.0)

        # === 2. Energy-limited XUV escape (Lammer+2003) ===
        # Important for H₂ (2 amu), He (4 amu), H₂O (18 amu — drag)
        # Mass loss rate: dM/dt = ε * π * R_xuv² * F_XUV / φ_grav
        # Integrate over stellar XUV evolution:
        # ∫ F_XUV dt from t_sat to age:
        #   = F_sat * t_sat * [1 + (age/t_sat)^(-0.23) / 0.23] (approx)
        # Total mass lost ~ ε * π * R_xuv² * F_XUV_integrated / φ_grav

        # Crossover mass: molecules lighter than M_cross can be dragged out
        # M_cross ≈ m_H + kT * Φ_diff / (b * g)  (Hunten 1973)
        # For Earth: crossover ~ 8-15 amu during early active phase
        # Simplified: species with M < 8 amu escape efficiently via XUV
        xuv_retention = 1.0
        if m_mol_amu < 20:  # potential XUV escape candidate
            # Total mass escapable per unit area: fluence * ε / φ_grav
            mass_loss_per_area = integrated_xuv_fluence * eps_xuv * 1e-4 / phi_grav  # kg/m²
            total_mass_lost = mass_loss_per_area * np.pi * R_xuv**2  # kg

            # Lighter species escape preferentially
            # Mass-dependent efficiency: heavier molecules have lower cross-section
            mass_eff = np.exp(-m_mol_amu / 4.0)  # e-folding at 4 amu
            effective_loss = total_mass_lost * mass_eff

            if species_mass[sp] > 0:
                xuv_retention = np.clip(
                    1.0 - effective_loss / species_mass[sp], 1e-6, 1.0
                )

        # Combined retention
        species_mass[sp] *= min(jeans_retention, xuv_retention)

    # --- INITIAL H₂O condensation ---
    # Water condenses only if the pre-feedback climate is below the critical
    # point. The later greenhouse solution iterates on surface temperature so
    # the vapor inventory is coupled to the evolving climate state.
    P_CO2_pre = species_mass.get('CO2', 0) * g_surface / A_surface / 101325
    dT_co2_pre = 7.4 * np.log(1 + P_CO2_pre / 2.8e-4)
    if P_CO2_pre > 10:
        dT_co2_pre += 200 * np.log10(P_CO2_pre / 10)
    P_H2O_pre = species_mass.get('H2O', 0) * g_surface / A_surface / 101325
    dT_h2o_pre = min(200, 12 * (P_H2O_pre / 0.01) ** 0.3) if P_H2O_pre > 0 else 0
    T_pre_check = T_eq + dT_co2_pre + dT_h2o_pre

    P0 = 101325
    L_vap = 2.26e6 * 18.015e-3
    allows_ocean = T_pre_check < 647
    ocean_mass = 0.0
    if allows_ocean:
        T_cond = max(T_eq + 20, 200)
        P_sat = P0 * np.exp(-L_vap / R_GAS * (1 / T_cond - 1 / 373))
        H2O_atm_max = surface_relative_humidity * P_sat * A_surface / g_surface
        if species_mass.get('H2O', 0) > H2O_atm_max:
            ocean_mass = species_mass['H2O'] - H2O_atm_max
            species_mass['H2O'] = H2O_atm_max

    # --- Geological CO₂ drawdown (pre-ocean) ---
    if age_gyr > 0.1:
        geo_time_factor = min(age_gyr / 0.3, 1.0)
        geo_T_factor = np.clip(1.5 - T_eq / 600, 0.1, 1.0)
        geo_sink = np.clip(geo_time_factor * geo_T_factor * 0.9, 0, 0.95)
        species_mass['CO2'] *= (1 - geo_sink)

    # --- Iterative CO₂ weathering + vapor adjustment ---
    for _iter in range(8):
        P_CO2_iter = species_mass.get('CO2', 0) * g_surface / A_surface / 101325
        P_H2O_iter = species_mass.get('H2O', 0) * g_surface / A_surface / 101325

        dT_co2_est = 7.4 * np.log(1 + P_CO2_iter / 2.8e-4)
        if P_CO2_iter > 10:
            dT_co2_est += 200 * np.log10(P_CO2_iter / 10)
        dT_h2o_est = min(100, 12 * (P_H2O_iter / 0.01) ** 0.3) if P_H2O_iter > 0 else 0
        T_surface_est = T_eq + dT_co2_est + dT_h2o_est + 15

        if ocean_mass > 0:
            ocean_rel = min(ocean_mass / 1.4e21, 1.0)
            time_factor = min(age_gyr / 0.5, 1.0)
            T_dep = np.clip((T_surface_est - 273) / 50, 0.5, 3.0)
            weathering_frac = 1.0 - 1.0 / (1.0 + 500 * ocean_rel * time_factor * T_dep)
            weathering_frac = np.clip(weathering_frac, 0.0, 0.99999)
            species_mass['CO2'] *= (1 - weathering_frac)

            P_sat_new = P0 * np.exp(-L_vap / R_GAS * (1 / max(T_surface_est, 200) - 1 / 373))
            H2O_atm_new = surface_relative_humidity * P_sat_new * A_surface / g_surface
            H2O_total = species_mass.get('H2O', 0) + ocean_mass
            if H2O_total > H2O_atm_new:
                species_mass['H2O'] = H2O_atm_new
                ocean_mass = H2O_total - H2O_atm_new
            else:
                species_mass['H2O'] = H2O_total
                ocean_mass = 0.0

    # --- Ocean dissolved gas sinks (Henry's Law) ---
    species_mass.pop('H2O_ocean', None)
    if ocean_mass > 0:
        ocean_rel = min(ocean_mass / 1.4e21, 1.0)
        solubility = {'SO2': 0.95, 'NH3': 0.90, 'H2S': 0.80, 'NO2': 0.70,
                      'HCN': 0.60, 'PH3': 0.10}
        for sp, max_frac in solubility.items():
            if sp in species_mass and species_mass[sp] > 0:
                species_mass[sp] *= (1 - max_frac * ocean_rel)

    # --- Abiogenic O₂ source-sink balance ---
    # Catling & Zahnle (2020): abiogenic O₂ is set by the competition between
    # H escape / photolysis sources and explicit reducing sinks from volcanic
    # gases and crustal oxidation, not by direct proportionality to bulk H₂O.
    solar_uv_ref = build_uv_template(
        5778, 4.5, 1.0, luminosity_lsun=1.0, radius_rsun=1.0
    )['F_UV_total']
    uv_norm = np.clip(photo_result['F_UV_total'] / max(solar_uv_ref, 1e-6), 1e-4, 1e4)
    h2_rate_kg_yr = H_rate * (1 - h2o_frac) * 0.5 * (2.016 / 1.008)
    co_rate_kg_yr = C_rate * 0.05 * (28.010 / 12.011)
    h2s_rate_kg_yr = S_rate * (1 - ox_frac) * 0.3 * (34.081 / 32.065)
    red_flux_mol_yr = (
        0.5 * h2_rate_kg_yr / 2.016e-3
        + 0.5 * co_rate_kg_yr / 28.010e-3
        + 1.5 * h2s_rate_kg_yr / 34.081e-3
    )
    crustal_sink = np.clip((bulk_comp.get('Fe', 0.06) / 0.06) ** 0.35, 0.5, 3.0)
    crustal_sink *= 1.0 + 4.0 * (1.0 - ox_frac)
    h2o_inventory_rel = max(species_mass.get('H2O', 0.0) + ocean_mass, 0.0) / 1.4e21
    co2_inventory_rel = max(species_mass.get('CO2', 0.0), 0.0) / 1e20
    escape_drive = np.clip(
        (F_xuv_sat / 500.0) ** 0.25
        * (integrated_xuv_fluence / max(500.0 * 0.1 * 3.15e16, 1.0)) ** 0.15
        * (11186.0 / max(v_esc, 2000.0)) ** 0.4,
        0.05, 12.0
    )
    source_pal = 2e-7 * uv_norm ** 0.7 * escape_drive * (0.5 + 0.5 * ox_frac)
    source_pal *= h2o_inventory_rel ** 0.6 + 0.2 * co2_inventory_rel ** 0.4
    sink_scale = 1.0 + red_flux_mol_yr / 3e10 + crustal_sink
    abiogenic_o2_pal = np.clip(source_pal / sink_scale, 1e-14, 3e-6)
    species_mass['O2'] = abiogenic_o2_pal * 1.18e18
    O3_factor = max(abiogenic_o2_pal, 1e-15) ** 0.4 * 2.5e-6
    species_mass['O3'] = species_mass['O2'] * O3_factor

    def _solve_surface_climate(base_t_eq, local_species_mass, total_water_inventory,
                               ocean_mass_local, initial_temp=None, max_iter=10):
        T_surface_local = max(base_t_eq + 5.0 if initial_temp is None else initial_temp, 180.0)
        greenhouse_terms_local = {
            'greenhouse': 0.0,
            'P_CO2_atm': 0.0,
            'P_H2O_atm': 0.0,
            'cc_amplification': 1.0,
            'rotation_cloud_cooling': 0.0,
            'redistribution_penalty': 0.0,
            'tidal_heating_K': 0.0,
        }
        iterations_local = 0
        for _iter in range(max_iter):
            if allows_ocean and total_water_inventory > 0:
                P_sat_loop = P0 * np.exp(-L_vap / R_GAS * (1 / max(T_surface_local, 200) - 1 / 373))
                H2O_atm_loop = min(
                    total_water_inventory,
                    surface_relative_humidity * P_sat_loop * A_surface / g_surface,
                )
                local_species_mass['H2O'] = H2O_atm_loop
                ocean_mass_local = max(total_water_inventory - H2O_atm_loop, 0.0)

            state_iter = _atmosphere_state(local_species_mass)
            greenhouse_terms_local = _greenhouse_response(state_iter, T_surface_local)
            T_new = base_t_eq + greenhouse_terms_local['greenhouse']
            iterations_local = _iter + 1
            if abs(T_new - T_surface_local) < 0.5:
                T_surface_local = T_new
                break
            T_surface_local = 0.5 * (T_surface_local + T_new)

        final_state_local = _atmosphere_state(local_species_mass)
        greenhouse_terms_local = _greenhouse_response(final_state_local, T_surface_local)
        T_surface_local = base_t_eq + greenhouse_terms_local['greenhouse']
        return final_state_local, greenhouse_terms_local, T_surface_local, ocean_mass_local, iterations_local

    def _apply_biogenic_oxygen(local_species_mass, state, target_partial_atm):
        if target_partial_atm <= 0.0 or state['P_surface_atm'] <= 0.0:
            return 0.0
        total_moles = max(sum(state['moles'].values()), 1.0)
        target_o2_moles = total_moles * target_partial_atm / max(state['P_surface_atm'], 1e-8)
        target_o2_mass = target_o2_moles * (MOLECULES['O2']['M'] * 1e-3)
        local_species_mass['O2'] = max(local_species_mass.get('O2', 0.0), target_o2_mass)
        biotic_pal_local = target_partial_atm / 0.2095
        total_o2_pal_local = np.clip(max(abiogenic_o2_pal, biotic_pal_local), 1e-14, 2.5)
        local_species_mass['O3'] = local_species_mass['O2'] * (
            max(total_o2_pal_local, 1e-15) ** 0.4 * 2.5e-6
        )
        return biotic_pal_local

    water_inventory_total = max(species_mass.get('H2O', 0.0) + ocean_mass, 0.0)
    final_state, greenhouse_terms, T_surface, ocean_mass, greenhouse_iterations = _solve_surface_climate(
        T_eq,
        species_mass,
        water_inventory_total,
        ocean_mass,
        initial_temp=max(T_eq + 5.0, 180.0),
        max_iter=10,
    )

    target_biotic_o2_atm = 0.0
    biotic_o2_pal = 0.0
    if (
        biotic_o2_atm is not None
        and biotic_o2_atm > 0.0
        and age_gyr >= 2.0
        and 273.0 <= T_surface <= 330.0
        and final_state['P_surface_atm'] >= 0.3
        and ocean_mass > 0.0
    ):
        insolation_rel = L_star_Lsun / max(semi_major_au, 1e-4) ** 2
        age_factor = np.clip((age_gyr - 2.0) / 2.0, 0.0, 1.0)
        temp_factor = np.exp(-((T_surface - 288.0) / 26.0) ** 2)
        pressure_factor = np.exp(-(abs(np.log10(max(final_state['P_surface_atm'], 1e-4))) / 0.45) ** 2)
        insolation_factor = np.exp(-((insolation_rel - 1.0) / 0.25) ** 2)
        oxygenation_factor = float(np.clip(age_factor * temp_factor * pressure_factor * insolation_factor, 0.0, 1.0))
        target_biotic_o2_atm = float(biotic_o2_atm * oxygenation_factor)
        if target_biotic_o2_atm > 1e-4:
            biotic_o2_pal = _apply_biogenic_oxygen(species_mass, final_state, target_biotic_o2_atm)
            water_inventory_total = max(species_mass.get('H2O', 0.0) + ocean_mass, 0.0)
            final_state, greenhouse_terms, T_surface, ocean_mass, oxygen_iterations = _solve_surface_climate(
                T_eq,
                species_mass,
                water_inventory_total,
                ocean_mass,
                initial_temp=T_surface,
                max_iter=6,
            )
            greenhouse_iterations += oxygen_iterations

    P_CO2_atm = greenhouse_terms['P_CO2_atm']
    P_H2O_atm = greenhouse_terms['P_H2O_atm']
    P_surface_atm = final_state['P_surface_atm']

    # --- Update albedo with atmospheric composition ---
    if ptype not in ('gas_giant', 'mini_neptune'):
        if P_H2O_atm > 0.001:
            f_cloud = min(0.7, (P_H2O_atm / 0.01) ** 0.3)
            cloud_albedo = f_cloud * 0.5
            albedo_bond = min(0.85, albedo_bond + cloud_albedo * 0.4)

        if slow_rotator and P_surface_atm > 0.1 and P_H2O_atm > 0.001:
            locked_cloud_boost = min(0.18 if tidally_locked else 0.08, greenhouse_terms['rotation_cloud_cooling'] / 250.0)
            albedo_bond = min(0.85, albedo_bond + locked_cloud_boost)

        if P_CO2_atm > 10:
            albedo_bond = min(0.85, albedo_bond + 0.15 * np.log10(P_CO2_atm / 10))

        if T_surface < 230:
            albedo_bond = max(albedo_bond, 0.6)
        elif T_surface < 273:
            ice_boost = 0.15 * (273 - T_surface) / 43
            albedo_bond = min(0.7, albedo_bond + ice_boost)

        if 273 < T_surface < 373 and ocean_mass > 0:
            ocean_frac = min(0.7, ocean_mass / 1.4e21)
            albedo_bond = albedo_bond * (1 - ocean_frac * 0.3)

    radiative_albedo = base_albedo_bond + 0.30 * (albedo_bond - base_albedo_bond)
    T_eq_effective = T_eq * (
        max(1.0 - radiative_albedo, 1e-6) / max(1.0 - base_albedo_bond, 1e-6)
    ) ** 0.25
    if abs(T_eq_effective - T_eq) > 0.25:
        water_inventory_total = max(species_mass.get('H2O', 0.0) + ocean_mass, 0.0)
        final_state, greenhouse_terms, T_surface, ocean_mass, albedo_iterations = _solve_surface_climate(
            T_eq_effective,
            species_mass,
            water_inventory_total,
            ocean_mass,
            initial_temp=T_surface,
            max_iter=6,
        )
        greenhouse_iterations += albedo_iterations

    if target_biotic_o2_atm > 1e-4:
        biotic_o2_pal = _apply_biogenic_oxygen(species_mass, final_state, target_biotic_o2_atm)
        final_state = _atmosphere_state(species_mass)
        greenhouse_terms = _greenhouse_response(final_state, T_surface)
        T_surface = T_eq_effective + greenhouse_terms['greenhouse']

    greenhouse = greenhouse_terms['greenhouse']
    P_CO2_atm = greenhouse_terms['P_CO2_atm']
    P_H2O_atm = greenhouse_terms['P_H2O_atm']
    atm_mass = final_state['atm_mass']
    P_surface = final_state['P_surface']
    P_surface_atm = final_state['P_surface_atm']
    P_surface_bar = final_state['P_surface_bar']
    mu = final_state['mu']
    moles = final_state['moles']
    gamma_avg = final_state['gamma_avg']

    composition = {}
    for sp, data in final_state['composition'].items():
        composition[sp] = {
            'vol_frac': round(float(data['vol_frac']), 8),
            'pct': round(float(data['pct']), 4),
            'partial_P_atm': round(float(data['partial_P_atm']), 6),
            'mass_kg': round(float(data['mass_kg']), 2),
        }

    h2o_vol_frac = composition.get('H2O', {}).get('vol_frac', 0.0)
    moist_greenhouse = bool(
        T_surface >= 340.0 or (T_surface >= 320.0 and h2o_vol_frac >= 0.02)
    )
    runaway_greenhouse = bool(
        T_surface >= 410.0 or (greenhouse >= 250.0 and P_H2O_atm >= 1.0 and ocean_mass <= 0.0)
    )
    cold_trap_collapse_risk = bool(
        tidally_locked and P_surface_atm < 0.1 and T_surface < 260.0
    )
    stable_surface_water = bool(
        273.0 <= T_surface <= 393.0
        and P_surface_atm >= 0.01
        and not moist_greenhouse
        and not runaway_greenhouse
        and not cold_trap_collapse_risk
    )

    # --- Atmospheric properties ---
    mu_kg = mu * 1e-3
    scale_height = R_GAS * T_surface / (mu_kg * g_surface)
    rho_air = P_surface * mu_kg / (R_GAS * T_surface)
    speed_sound = np.sqrt(gamma_avg * R_GAS * T_surface / mu_kg)
    tropopause = scale_height * 1.4
    homosphere = scale_height * 12

    return {
        'surface_pressure_atm': round(float(P_surface_atm), 6),
        'surface_pressure_bar': round(float(P_surface_bar), 6),
        'surface_temp_K': round(float(T_surface), 1),
        'T_eq_K': round(float(T_eq), 1),
        'T_eq_effective_K': round(float(T_eq_effective), 1),
        'greenhouse_K': round(float(greenhouse), 1),
        'scale_height_km': round(float(scale_height * 1e-3), 2),
        'tropopause_km': round(float(tropopause * 1e-3), 1),
        'homosphere_km': round(float(homosphere * 1e-3), 1),
        'air_density_kg_m3': round(float(rho_air), 6),
        'sound_speed_m_s': round(float(speed_sound), 1),
        'exosphere_temp_K': round(float(T_exo), 0),
        'mean_mol_mass': round(float(mu), 2),
        'atm_mass_kg': float(f'{atm_mass:.3e}'),
        'atm_relative_mass': float(f'{atm_mass / M:.3e}'),
        'water_ocean_mass_kg': float(f'{ocean_mass:.3e}'),
        'abiogenic_o2_pal': float(f'{abiogenic_o2_pal:.3e}'),
        'biotic_o2_pal': float(f'{biotic_o2_pal:.3e}'),
        'biotic_o2_partial_atm': round(float(target_biotic_o2_atm), 6),
        'volcanic_o2_sink_mol_yr': float(f'{red_flux_mol_yr:.3e}'),
        'albedo_bond_eff': round(float(albedo_bond), 3),
        'surface_relative_humidity': round(float(surface_relative_humidity), 3),
        'composition': composition,
        'feedback_diagnostics': {
            'iterations': greenhouse_iterations,
            'cc_amplification': round(float(greenhouse_terms['cc_amplification']), 3),
            'rotation_cloud_cooling_K': round(float(greenhouse_terms['rotation_cloud_cooling']), 3),
            'redistribution_penalty_K': round(float(greenhouse_terms['redistribution_penalty']), 3),
            'tidal_heating_K': round(float(greenhouse_terms['tidal_heating_K']), 3),
        },
        'climate_regime': {
            'slow_rotator': slow_rotator,
            'tidally_locked': bool(tidally_locked),
            'moist_greenhouse': moist_greenhouse,
            'runaway_greenhouse': runaway_greenhouse,
            'cold_trap_collapse_risk': cold_trap_collapse_risk,
            'stable_surface_water': stable_surface_water,
        },
        'xuv_history': {
            'spec_class': xuv_history['spec_class'],
            't_sat_gyr': round(float(t_sat_gyr), 4),
            'pre_ms_duration_gyr': round(float(xuv_history['pre_ms_duration_gyr']), 4),
            'pre_ms_boost': round(float(xuv_history['pre_ms_boost']), 3),
            'F_xuv_sat_erg_cm2_s': round(float(F_xuv_sat), 4),
            'xuv_fluence_erg_cm2': float(f'{integrated_xuv_fluence:.3e}'),
        },
        'photolysis': {
            'stellar_uv_class': photo_result['stellar_uv_class'],
            'F_UV_total': photo_result['F_UV_total'],
            'flare_boost': round(float(photo_result.get('flare_boost', 1.0)), 3),
            'species': {
                sp: {
                    'tau_upper_yr': round(v.get('tau_upper_yr', 1e12), 6),
                    'tau_mid_yr': round(v.get('tau_mid_yr', 1e12), 6),
                    'tau_column_yr': round(v.get('tau_column_yr', 1e12), 6),
                }
                for sp, v in photo_result['species'].items()
                if sp in ('CH4','CO','HCN','NH3','NO2','SO2','H2S','PH3','H2O','CO2','O3','O2')
            },
        },
    }


# ============================================================
# LATE VENEER — 5-Class System (LV0–LV4)
#
# References:
#   Walker+2009, EPSL 286          — HSE abundances in Earth mantle
#   Day+2016, Rev. Mineral. 81     — Late accretion constraints
#   Bottke+2010, Science 330       — Late heavy bombardment
#   Albarede 2009, Nature 461      — Water delivery by late veneer
#   Fischer-Godde & Kleine 2017    — Ruthenium isotopes → LV origin
# ============================================================

# Per-class delivery properties
# lv_mass_frac : fraction of planet's final mass delivered post-core-closure
# water_frac   : mass fraction of delivered material that is water
# hse_enrichment: siderophile enrichment relative to chondritic
# carbonaceous_frac: fraction of impactors that are C-type (volatile-rich)
# tail_myr     : e-folding bombardment tail timescale (Myr)
LV_CLASSES = {
    'LV0': {'label': '거의 없음',     'lv_mass_frac': 0.0005,
            'water_frac': 1e-5,  'hse_enrichment': 0.1,
            'carbonaceous_frac': 0.01, 'volatile_enrichment': 0.05,
            'tail_myr': 20},
    'LV1': {'label': '미약',         'lv_mass_frac': 0.002,
            'water_frac': 5e-4,  'hse_enrichment': 0.5,
            'carbonaceous_frac': 0.05, 'volatile_enrichment': 0.3,
            'tail_myr': 50},
    'LV2': {'label': '보통',         'lv_mass_frac': 0.005,
            'water_frac': 0.002, 'hse_enrichment': 1.0,
            'carbonaceous_frac': 0.15, 'volatile_enrichment': 1.0,
            'tail_myr': 150},
    'LV3': {'label': '강함',         'lv_mass_frac': 0.015,
            'water_frac': 0.008, 'hse_enrichment': 2.0,
            'carbonaceous_frac': 0.40, 'volatile_enrichment': 2.5,
            'tail_myr': 400},
    'LV4': {'label': '극단적 재공급', 'lv_mass_frac': 0.05,
            'water_frac': 0.03,  'hse_enrichment': 5.0,
            'carbonaceous_frac': 0.70, 'volatile_enrichment': 6.0,
            'tail_myr': 800},
}

# Highly siderophile elements boosted by late veneer (Walker+2009)
# These elements partition strongly into the core during differentiation,
# so their mantle abundance is almost entirely from post-core-closure delivery.
HSE_ELEMENTS = {'Ni', 'Mn', 'Co', 'Cu', 'Mo', 'W'}

# Chalcophile/volatile elements enhanced by carbonaceous impactors
# (Wang & Becker 2013, Braukmüller+2019)
CHALCOPHILE_VOLATILE = {'S': 2.5, 'Se': 3.0, 'Zn': 1.8, 'P': 2.0}


def classify_late_veneer(planet, all_planets, belt, disk, rng=None):
    """Classify a rocky planet's late veneer intensity (LV0–LV4).

    Six scoring variables determine the class:
      1. Giant planet architecture   — close giants excite debris
      2. Asteroid belt mass          — available impactor reservoir
      3. Formation radius            — inner planets receive more
      4. Dynamical excitation        — high e(a) → more delivery
      5. Volatile reservoir           — C-type fraction in belt
      6. Planet mass / escape velocity — bigger planets capture more

    Parameters
    ----------
    planet : dict   — the target planet
    all_planets : list of dict — full system
    belt : dict     — asteroid belt output from build_asteroid_belt
    disk : ProtoplanetaryDisk
    rng : np.random.Generator or None

    Returns
    -------
    dict with 'class', 'label', 'score', 'scores', and all delivery properties
    """
    if rng is None:
        rng = np.random.default_rng()

    scores = {}

    # --- 1. Giant planet architecture (0-1) ---
    giants = [p for p in all_planets if p.get('type') == 'gas_giant']
    if len(giants) > 0:
        closest_au = min(g['semi_major_au'] for g in giants)
        planet_au = planet['semi_major_au']
        # Giants closer to the belt pump more material towards inner planets
        prox_factor = np.clip(5.0 / max(closest_au, 0.5), 0, 2.0)
        scores['giant_arch'] = np.clip(0.15 * len(giants) * prox_factor, 0, 1)
    else:
        scores['giant_arch'] = 0.05  # minimal without giants

    # --- 2. Asteroid belt mass (0-1) ---
    belt_mass = belt.get('final_mass_earth', 0) if belt else 0
    # Solar belt ~4.5e-4 Me → score ~0.5
    scores['belt_mass'] = np.clip(np.log10(max(belt_mass, 1e-8)) / (-3.3) + 0.5, 0, 1)

    # --- 3. Formation radius (0-1) ---
    # Inner planets (< 1 AU) receive more bombardment; outer planets less
    a_planet = planet['semi_major_au']
    scores['formation_radius'] = np.clip(1.0 - 0.3 * np.log10(max(a_planet, 0.05)), 0, 1)

    # --- 4. Dynamical excitation (0-1) ---
    ecc_mean = 0.15  # default
    if belt and 'ecc_a' in belt and len(belt['ecc_a']) > 0:
        ecc_mean = float(np.mean(belt['ecc_a']))
    scores['dyn_excitation'] = np.clip(ecc_mean / 0.3, 0, 1)

    # --- 5. Volatile reservoir (0-1) ---
    c_frac = 0.5  # default
    if belt and 'composition' in belt:
        c_frac = belt['composition'].get('C_type_frac', 0.5)
    scores['volatile_reservoir'] = np.clip(c_frac, 0, 1)

    # --- 6. Planet mass / escape velocity (0-1) ---
    mass_e = planet.get('mass_earth', 1.0)
    # Larger planets have higher gravitational cross-section
    # Earth (1 Me) → ~0.5; Mars (0.1) → ~0.25; Super-Earth (5) → ~0.75
    scores['mass_capture'] = np.clip(0.3 + 0.2 * np.log10(max(mass_e, 0.01)), 0, 1)

    # --- Weighted total score ---
    weights = {
        'giant_arch': 0.25,
        'belt_mass': 0.20,
        'formation_radius': 0.15,
        'dyn_excitation': 0.15,
        'volatile_reservoir': 0.10,
        'mass_capture': 0.15,
    }
    total_score = sum(scores[k] * weights[k] for k in scores)
    # Add stochastic scatter (±0.1)
    total_score += rng.normal(0, 0.08)
    total_score = np.clip(total_score, 0, 1)

    # --- Map score to class ---
    if total_score < 0.15:
        cls = 'LV0'
    elif total_score < 0.35:
        cls = 'LV1'
    elif total_score < 0.55:
        cls = 'LV2'
    elif total_score < 0.75:
        cls = 'LV3'
    else:
        cls = 'LV4'

    props = LV_CLASSES[cls].copy()
    props['class'] = cls
    props['score'] = round(float(total_score), 4)
    props['scores'] = {k: round(float(v), 4) for k, v in scores.items()}

    return props


# ============================================================
# DIFFERENTIATION — Wood+2006 (with LV classification)
# ============================================================
def differentiate_full(planet_mass, bulk_comp, oxidation_iw, diff_progress=1.0,
                       lv_frac_override=None, disprop_scale=None, lv_class_params=None):
    """Full 20-element core/mantle/crust partitioning with Disproportionation & Late Veneer.

    If lv_class_params is provided (from classify_late_veneer), uses class-specific
    delivery properties. Otherwise falls back to legacy lv_frac behavior.
    """
    fe_frac = bulk_comp.get('Fe', SOLAR_X[EL_IDX['Fe']])
    si_frac = bulk_comp.get('Si', SOLAR_X[EL_IDX['Si']])
    fe_si = fe_frac / max(si_frac, 1e-12)
    earth_fe_si = SOLAR_X[EL_IDX['Fe']] / SOLAR_X[EL_IDX['Si']]

    # Bridgmanite Disproportionation: 3Fe2+ -> Fe0 + 2Fe3+
    # Small planets (M < 1 M⊕) have lower mantle pressures → weaker disproportionation
    # Use max(planet_mass, 0.01) to allow continuous variation down to small bodies
    disp_sc = disprop_scale if disprop_scale is not None else 1.5
    ox_shift_disprop = min(disp_sc * np.log10(max(planet_mass, 0.01)), 2.0)
    oxidation_iw += ox_shift_disprop

    cmf_base = 0.325 * (fe_si / earth_fe_si) ** 0.6
    ox_factor = 1.0 - 0.12 * oxidation_iw
    cmf = np.clip(cmf_base * ox_factor * (1 + 0.02 * np.log10(max(planet_mass, 0.1))), 0.05, 0.80)

    al_frac = bulk_comp.get('Al', SOLAR_X[EL_IDX['Al']])
    al_ratio = al_frac / max(SOLAR_X[EL_IDX['Al']], 1e-12)
    crust_frac = np.clip(0.005 * al_ratio * planet_mass ** -0.1, 0.002, 0.06)
    mantle_frac = 1.0 - cmf - crust_frac

    cmf_eff = diff_progress * cmf + (1 - diff_progress) * 0.33
    mantle_eff = diff_progress * mantle_frac + (1 - diff_progress) * 0.64
    crust_eff = diff_progress * crust_frac + (1 - diff_progress) * 0.03

    core_el, mantle_el, crust_el = {}, {}, {}
    tracked = [e for e in ELEMENTS if e != 'He']

    # --- Late Veneer fraction ---
    if lv_class_params is not None:
        lv_frac = np.clip(lv_class_params['lv_mass_frac'], 0.0, 0.10)
        water_frac = lv_class_params.get('water_frac', 0.0)
        hse_enrich = lv_class_params.get('hse_enrichment', 1.0)
        carb_frac = lv_class_params.get('carbonaceous_frac', 0.15)
    elif lv_frac_override is not None:
        lv_frac = np.clip(lv_frac_override, 0.0, 0.10)
        water_frac = 0.001
        hse_enrich = 1.0
        carb_frac = 0.15
    else:
        lv_frac = np.clip(0.005 * planet_mass ** -0.5, 0.001, 0.02)
        water_frac = 0.001
        hse_enrich = 1.0
        carb_frac = 0.15

    for el in tracked:
        bulk = bulk_comp.get(el, 0.0)
        dc, dm, dcr = PARTITION_COEFF.get(el, (0.33, 0.34, 0.33))
        ox_shift = 0.05 * oxidation_iw
        dc_adj = np.clip(dc - ox_shift * dc, 0, 1)
        dm_adj = np.clip(dm + ox_shift * dc * 0.7, 0, 1)
        dcr_adj = np.clip(dcr + ox_shift * dc * 0.3, 0, 1)
        total_d = dc_adj + dm_adj + dcr_adj
        if total_d > 0:
            dc_adj /= total_d; dm_adj /= total_d; dcr_adj /= total_d
        p_shift = 0.02 * np.log10(max(planet_mass, 0.1))
        dc_adj = np.clip(dc_adj + p_shift, 0, 1)
        tot = dc_adj + dm_adj + dcr_adj
        if tot > 0:
            dc_adj /= tot; dm_adj /= tot; dcr_adj /= tot
        
        f_core = diff_progress * dc_adj + (1 - diff_progress) * 0.33
        f_mant = diff_progress * dm_adj + (1 - diff_progress) * 0.34
        f_crus = diff_progress * dcr_adj + (1 - diff_progress) * 0.33

        # Apply primary differentiation to the main body mass (1 - lv_frac)
        c_mass = bulk * (1 - lv_frac) * f_core
        m_mass = bulk * (1 - lv_frac) * f_mant
        cr_mass = bulk * (1 - lv_frac) * f_crus

        # --- Late Veneer delivery (class-dependent) ---
        lv_payload = bulk * lv_frac

        # HSE elements enriched in the mantle (Walker+2009)
        # Core formation strips HSE from mantle; LV re-supplies them
        if el in HSE_ELEMENTS:
            lv_payload *= hse_enrich

        # Water delivery enhances H and O (Albarede 2009)
        if el == 'H':
            lv_payload += lv_frac * water_frac * 0.111  # H mass fraction of H2O
        elif el == 'O':
            lv_payload += lv_frac * water_frac * 0.889  # O mass fraction of H2O

        # Carbonaceous impactors enrich C, N (Fischer-Gödde & Kleine 2017)
        if el == 'C':
            lv_payload *= (1 + 2.0 * carb_frac)
        elif el == 'N':
            lv_payload *= (1 + 1.5 * carb_frac)

        # Chalcophile/volatile elements from carbonaceous impactors
        # (Wang & Becker 2013, Braukmüller+2019)
        # These are enriched in CI/CM chondrites relative to the silicate Earth
        vol_enrich = lv_class_params.get('volatile_enrichment', 1.0) if lv_class_params else 1.0
        if el in CHALCOPHILE_VOLATILE:
            lv_payload *= (1 + CHALCOPHILE_VOLATILE[el] * carb_frac * vol_enrich)

        # Phosphorus: 60-90% of Earth's mantle P is LV-derived (Goldstein+2009)
        if el == 'P':
            lv_payload *= (1 + 3.0 * carb_frac * vol_enrich)

        # LV payload goes to mantle (85%) and crust (15%) — undifferentiated
        m_mass += lv_payload * 0.85
        cr_mass += lv_payload * 0.15

        core_el[el] = float(c_mass)
        mantle_el[el] = float(m_mass)
        crust_el[el] = float(cr_mass)

    def _norm(d):
        total = sum(d.values())
        return {k: round(v / total, 8) for k, v in d.items()} if total > 0 else {k: 0.0 for k in d}

    result = {
        'core_frac': round(float(cmf_eff), 5),
        'mantle_frac': round(float(mantle_eff), 5),
        'crust_frac': round(float(crust_eff), 5),
        'core': _norm(core_el), 'mantle': _norm(mantle_el), 'crust': _norm(crust_el),
        'core_mass': {k: round(v, 10) for k, v in core_el.items()},
        'mantle_mass': {k: round(v, 10) for k, v in mantle_el.items()},
        'crust_mass': {k: round(v, 10) for k, v in crust_el.items()},
        'fe_si': round(float(fe_si), 4),
        'oxidation_iw': round(float(oxidation_iw), 2),
        'diff_progress': round(float(diff_progress), 3),
    }

    # Attach LV class info if available
    if lv_class_params is not None:
        result['late_veneer'] = {
            'class': lv_class_params['class'],
            'label': lv_class_params['label'],
            'score': lv_class_params['score'],
            'lv_mass_frac': round(float(lv_frac), 5),
            'water_frac': round(float(water_frac), 6),
            'hse_enrichment': round(float(hse_enrich), 2),
            'carbonaceous_frac': round(float(carb_frac), 3),
            'tail_myr': lv_class_params.get('tail_myr', 150),
            'scores': lv_class_params.get('scores', {}),
        }

    return result


def diff_timescale(planet_mass):
    """Differentiation timescale in Gyr."""
    return np.clip(0.03 * planet_mass ** -0.5, 0.01, 1.0)


def build_outgassing_reservoir(diff_result):
    """Return the volatile-accessible mantle+crust reservoir as planet-mass fractions."""
    mantle = diff_result.get('mantle_mass', {})
    crust = diff_result.get('crust_mass', {})
    reservoir = {}
    for el in set(mantle) | set(crust):
        accessible = max(mantle.get(el, 0.0), 0.0) + max(crust.get(el, 0.0), 0.0)
        if accessible > 0:
            reservoir[el] = float(accessible)
    return reservoir


def _habitability_assessment(planet, atmosphere=None, radiation=None):
    """Score surface habitability from climate, water retention, and radiation."""
    ptype = planet.get('type')
    hz_bounds = planet.get('hz_bounds_au', {})

    if atmosphere is None:
        return {
            'score': 0.0,
            'label': 'unknown',
            'reasons': ['no_atmosphere_model'],
            'subscores': {},
        }

    surface_candidate = ptype in ('rocky', 'hot_rocky')
    if not surface_candidate:
        return {
            'score': 0.05 if ptype == 'mini_neptune' and atmosphere.get('envelope_regime') == 'secondary_atmosphere' else 0.0,
            'label': 'non_terrestrial',
            'reasons': ['thick_envelope_or_no_surface'],
            'subscores': {
                'hz': 1.0 if planet.get('in_hz_dynamic') else 0.2,
                'surface': 0.0,
            },
        }

    temp = float(atmosphere.get('surface_temp_K', 0.0))
    pressure = max(float(atmosphere.get('surface_pressure_atm', 0.0)), 1e-8)
    ocean_mass = max(float(atmosphere.get('water_ocean_mass_kg', 0.0)), 0.0)
    climate = atmosphere.get('climate_regime', {})
    stable_surface_water = bool(climate.get('stable_surface_water', False))
    moist_greenhouse = bool(climate.get('moist_greenhouse', False))
    runaway_greenhouse = bool(climate.get('runaway_greenhouse', False))
    cold_trap = bool(climate.get('cold_trap_collapse_risk', False))

    if planet.get('in_hz_dynamic'):
        hz_score = 1.0
    elif planet.get('in_hz_optimistic'):
        hz_score = 0.65
    else:
        hz_mid = np.sqrt(
            max(hz_bounds.get('conservative_inner', 0.1), 1e-6)
            * max(hz_bounds.get('conservative_outer', 0.1), 1e-6)
        )
        hz_score = float(np.clip(np.exp(-abs(np.log(max(planet.get('semi_major_au', 1.0), 1e-6) / hz_mid)) / 1.2), 0.0, 0.5))

    temp_score = float(np.clip(np.exp(-((temp - 288.0) / 60.0) ** 2), 0.0, 1.0))
    pressure_score = float(np.clip(np.exp(-(abs(np.log10(pressure)) / 1.15) ** 1.35), 0.0, 1.0))
    water_score = float(np.clip(min(ocean_mass / 1.4e21, 1.0), 0.0, 1.0))
    if water_score == 0.0:
        water_score = float(np.clip(atmosphere.get('composition', {}).get('H2O', {}).get('vol_frac', 0.0) / 0.02, 0.0, 0.35))

    uv_surface = float((radiation or {}).get('surface_W_m2', {}).get('uv', 0.0))
    xuv_surface = float((radiation or {}).get('surface_W_m2', {}).get('x_euv', 0.0))
    radiation_score = float(np.clip(np.exp(-(uv_surface / 12.0 + xuv_surface / 0.2)), 0.0, 1.0))

    climate_modifier = 1.0
    if not stable_surface_water:
        climate_modifier *= 0.6
    if moist_greenhouse:
        climate_modifier *= 0.45
    if runaway_greenhouse:
        climate_modifier *= 0.05
    if cold_trap:
        climate_modifier *= 0.2

    base_score = (
        0.24 * hz_score
        + 0.24 * temp_score
        + 0.16 * pressure_score
        + 0.20 * water_score
        + 0.16 * radiation_score
    )
    score = float(np.clip(base_score * climate_modifier, 0.0, 1.0))

    reasons = []
    if not planet.get('in_hz_dynamic'):
        reasons.append('outside_conservative_hz')
    if moist_greenhouse:
        reasons.append('moist_greenhouse')
    if runaway_greenhouse:
        reasons.append('runaway_greenhouse')
    if cold_trap:
        reasons.append('cold_trap_risk')
    if pressure < 0.01:
        reasons.append('atmosphere_too_thin')
    elif pressure > 200.0:
        reasons.append('surface_pressure_extreme')
    if uv_surface > 10.0 or xuv_surface > 0.2:
        reasons.append('radiation_stress')
    if water_score < 0.2:
        reasons.append('limited_surface_water')

    if score >= 0.72 and stable_surface_water and radiation_score > 0.35 and planet.get('in_hz_dynamic'):
        label = 'candidate'
    elif score >= 0.4:
        label = 'marginal'
    else:
        label = 'hostile'

    return {
        'score': round(score, 4),
        'label': label,
        'reasons': reasons,
        'subscores': {
            'hz': round(hz_score, 4),
            'temperature': round(temp_score, 4),
            'pressure': round(pressure_score, 4),
            'water': round(water_score, 4),
            'radiation': round(radiation_score, 4),
        },
    }

# ============================================================
# BUILD PLANET SYSTEM — Master function
# ============================================================
def build_planet_system(star_id, star_mass, birth_time, r_zone,
                        gce_result, current_time=13.8, rng_seed=None, evo=None,
                        esi_weights=None, lv_frac=None, disprop_scale=None,
                        stellar_model=None):
    """Build full planetary system with all physics.

    Now includes protoplanetary disk model (Hayashi 1981, Andrews+2013)
    and asteroid belt formation (Raymond & Izidoro 2017).
    """
    from .disk import ProtoplanetaryDisk, build_asteroid_belt
    from .stellar import stellar_evolution as _stellar_evolution

    rng = np.random.default_rng(rng_seed if rng_seed is not None else star_id)

    t_arr = np.array(gce_result['time'])
    it = int(np.argmin(np.abs(t_arr - birth_time)))
    ir = min(r_zone, len(gce_result['radius']) - 1)

    comp = {}
    for el in ELEMENTS:
        if el in ('H', 'He'):
            continue
        mf = gce_result['mass_fractions'][el]
        comp[el] = mf[ir][it]

    metallicity = gce_result['metallicity'][ir][it]

    # --- Protoplanetary Disk ---
    r_galactic_kpc = gce_result['radius'][ir] if ir < len(gce_result['radius']) else 8.0
    disk = ProtoplanetaryDisk(star_mass, metallicity, r_galactic_kpc, rng)

    # --- Disk-based planet generation ---
    planets = generate_planets(star_mass, metallicity, rng, disk=disk)
    planet_age = max(current_time - birth_time, 0)

    # --- Asteroid Belt ---
    belt = build_asteroid_belt(disk, planets, planet_age, rng)
    
    L_star = star_mass ** 3.5
    c_mass = star_mass
    max_r_au = 0.0

    if evo:
        L_star = evo.get('luminosity', L_star)
        c_mass = evo.get('current_mass', star_mass)
        max_r_au = evo.get('max_radius_au', sum(p['semi_major_au'] for p in planets)*0.001)

    from .photolysis import estimate_flare_boost

    star_teff_current = evo.get('T_eff', 5778 * star_mass**0.55) if evo else 5778 * star_mass**0.55
    hz_model = estimate_spectral_habitable_zone(L_star, star_teff_current)

    for p in planets:
        formation_a_au = p.get('formation_semi_major_au', p['semi_major_au'])
        p['formation_delay'] = round(0.01 + 0.02 * rng.random(), 4)
        formation_evo = _stellar_evolution(
            star_mass,
            p['formation_delay'],
            metallicity_z=metallicity,
            model=stellar_model,
        )
        formation_luminosity = formation_evo.get('luminosity', star_mass ** 3.5)
        formation_albedo = 0.10 if p['type'] == 'hot_rocky' else 0.20
        p['formation_T_eq_K'] = round(float(estimate_equilibrium_temperature(
            formation_luminosity, formation_a_au, formation_albedo
        )), 1)
        p['formation_luminosity_Lsun'] = round(float(formation_luminosity), 6)
        p['birth_time'] = round(birth_time + p['formation_delay'], 4)
        p['formed'] = bool(current_time >= p['birth_time'])
        p['moon_system'] = None
        p['has_moon_system'] = False
        p['moon_count'] = 0

        if not p['formed']:
            p.update({'differentiation': None, 'bulk_composition': None,
                      'thermal': None, 'magnetic': None, 'atmosphere': None,
                      'physical': None, 'status': 'protoplanetary_disk'})
            continue

        actual_age = max(current_time - p['birth_time'], 0)
        flare_boost = estimate_flare_boost(star_teff_current, actual_age)

        # --- Orbital Expansion via Mass Loss (Adiabatic) ---
        # a_final = a_initial * (M_initial / M_final)
        # Conserves angular momentum as central star loses mass
        p['original_semi_major_au'] = p['semi_major_au']
        if c_mass < star_mass:
            p['semi_major_au'] = round(p['semi_major_au'] * (star_mass / c_mass), 4)

        # --- Dynamic Habitable Zone Calculation ---
        # Recalculate HZ boundaries based on the current stellar luminosity
        dynamic_hz_in = hz_model['conservative_inner_au']
        dynamic_hz_out = hz_model['conservative_outer_au']
        p['hz_bounds_au'] = {
            'conservative_inner': dynamic_hz_in,
            'conservative_outer': dynamic_hz_out,
            'optimistic_inner': hz_model['optimistic_inner_au'],
            'optimistic_outer': hz_model['optimistic_outer_au'],
        }
        p['in_hz_dynamic'] = bool(dynamic_hz_in <= p['semi_major_au'] <= dynamic_hz_out)
        p['in_hz_optimistic'] = bool(
            hz_model['optimistic_inner_au'] <= p['semi_major_au'] <= hz_model['optimistic_outer_au']
        )
        p['is_habitable'] = False # Default

        # --- Stellar Engulfment Check ---
        # If the star's maximum historical radius exceeds the planet's orbit, it is destroyed
        if p['semi_major_au'] < max_r_au:
            p.update({'differentiation': None, 'bulk_composition': None,
                      'thermal': None, 'magnetic': None, 'atmosphere': None,
                      'physical': None, 'status': 'destroyed_by_star'})
            continue

        # --- Physical properties (all planets) ---
        phys = compute_physical_properties(
            p['mass_earth'], p['type'], p['semi_major_au'], c_mass,
            p['rotation_period_hr'], p['axial_tilt_deg'], p['eccentricity'],
            actual_age, L_star, esi_weights=esi_weights
        )
        p['physical'] = phys
        p['T_eq'] = phys['T_eq_K']

        if p['type'] in ('rocky', 'hot_rocky'):
            T_eq = phys['T_eq_K']
            formation_t_eq = p.get('formation_T_eq_K', T_eq)
            depleted = volatile_depletion(comp, formation_t_eq)
            ox = rng.normal(-2.0, 1.5)
            tau = diff_timescale(p['mass_earth'])
            progress = np.clip(actual_age / tau, 0, 1) if tau > 0 else 1.0

            # --- Per-planet Late Veneer classification (LV0–LV4) ---
            lv_params = classify_late_veneer(p, planets, belt, disk, rng)
            if lv_frac is not None:
                lv_params = {
                    **lv_params,
                    'lv_mass_frac': float(np.clip(lv_frac, 0.0, 0.10)),
                }
            p['late_veneer'] = {
                'class': lv_params['class'],
                'label': lv_params['label'],
                'score': lv_params['score'],
                'lv_mass_frac': lv_params['lv_mass_frac'],
                'water_frac': lv_params['water_frac'],
                'hse_enrichment': lv_params['hse_enrichment'],
                'carbonaceous_frac': lv_params['carbonaceous_frac'],
                'tail_myr': lv_params['tail_myr'],
                'scores': lv_params['scores'],
            }

            diff_result = differentiate_full(p['mass_earth'], depleted, ox, progress,
                                             disprop_scale=disprop_scale,
                                             lv_class_params=lv_params)
            p['differentiation'] = diff_result
            outgassing_reservoir = build_outgassing_reservoir(diff_result)
            p['bulk_composition'] = {el: round(v, 8) for el, v in depleted.items() if v > 1e-10}
            p['outgassing_reservoir'] = {el: round(v, 8) for el, v in outgassing_reservoir.items() if v > 1e-10}

            thermal = core_thermal_model(p['mass_earth'], diff_result['core_frac'], depleted, actual_age,
                                         cosmic_time=current_time)
            p['thermal'] = thermal

            P_core_GPa = 340 * p['mass_earth'] ** 0.65
            eta, state = core_viscosity(thermal['T_core'], P_core_GPa)
            p['core_viscosity'] = {'eta_Pa_s': eta, 'state': state}

            spin_for_dynamo = phys.get('rotation_effective_hr', p['rotation_period_hr'])
            mag = magnetic_field(p['mass_earth'], diff_result['core_frac'],
                                 thermal['T_core'], spin_for_dynamo,
                                 thermal['q_cmb'], thermal['core_liquid'])
            p['magnetic'] = mag

            atmo = compute_atmosphere(outgassing_reservoir, p['mass_earth'], p['type'],
                                      T_eq, phys['g_mean_m_s2'], phys['v_esc_km_s'] * 1000,
                                      actual_age, ox, L_star_Lsun=L_star,
                                      star_teff=star_teff_current,
                                      semi_major_au=p['semi_major_au'],
                                      star_radius_rsun=evo.get('radius') if evo else None,
                                      rotation_period_hr=phys.get('rotation_effective_hr'),
                                      orbital_period_days=phys.get('orbital_period_days'),
                                      tidally_locked=phys.get('tidally_locked', False),
                                      tidal_heating_TW=phys.get('tidal_heating_TW', 0.0),
                                      flare_boost=flare_boost)
            p['atmosphere'] = atmo
            
            # --- ESI Recalculation (using actual surface temperature) ---
            w_r   = (esi_weights or {}).get('w_radius', 0.57)
            w_rho = (esi_weights or {}).get('w_density', 1.07)
            w_ve  = (esi_weights or {}).get('w_escape', 0.70)
            w_t   = (esi_weights or {}).get('w_temp', 5.58)
            def _esi(x, x_e, w): return (1 - abs((x - x_e) / (x + x_e))) ** w
            phys['ESI'] = round(float((
                _esi(phys['R_mean_Re'], 1.0, w_r) *
                _esi(phys['density_kg_m3'], 5515, w_rho) *
                _esi(phys['v_esc_km_s'] * 1000, 11186, w_ve) *
                _esi(max(atmo['surface_temp_K'], 100), 288, w_t)
            ) ** 0.25), 4)

            p['radiation'] = compute_radiation_defense(
                p['mass_earth'], phys['R_mean_Re'], mag['B_surface_uT'],
                atmo['atm_mass_kg'], atmo['surface_pressure_bar'], atmo['composition'],
                p['semi_major_au'], star_teff_current, L_star_Lsun=L_star, age_gyr=actual_age,
                flare_boost=flare_boost,
            )
            p['habitability'] = _habitability_assessment(p, atmo, p['radiation'])
            p['habitability_score'] = p['habitability']['score']
            p['is_habitable'] = p['habitability']['label'] == 'candidate'
            p['moon_system'] = build_rocky_moon_system(
                p,
                star_mass=star_mass,
                all_planets=planets,
                late_veneer=lv_params,
                phys=phys,
                current_stellar_mass=c_mass,
                actual_age_gyr=actual_age,
                rng_seed=star_id * 1009 + p['index'] * 131 + 7,
            )
            p['moon_count'] = (
                p['moon_system']['summary']['n_major']
                + p['moon_system']['summary']['n_minor']
            )
            p['has_moon_system'] = bool(p['moon_count'] > 0)

            p['status'] = 'active'

        elif p['type'] == 'gas_giant':
            T_eq_gg = phys['T_eq_K']
            M_jup = 317.8
            m_ratio = p['mass_earth'] / M_jup

            # Interior structure (Thorngren+2016, Wahl+2017)
            # Core mass: ~10-20 M⊕ for Jupiter-mass, scales with Z
            M_core_earth = 10 * (metallicity / 0.0134) ** 0.5 * m_ratio ** 0.1
            M_core_earth = np.clip(M_core_earth, 3, 50)
            core_frac = M_core_earth / p['mass_earth']
            Z_env = 5 * (metallicity / 0.0134) * m_ratio ** (-0.45)
            Z_env = np.clip(Z_env, 1, 100)

            p['differentiation'] = {
                'core_frac': round(float(core_frac), 4),
                'mantle_frac': round(float(1 - core_frac), 4),
                'crust_frac': 0,
                'diff_progress': 1.0,
                'core_type': 'rock+ice',
                'M_core_earth': round(float(M_core_earth), 2),
                'Z_envelope': round(float(Z_env), 1),
            }
            p['bulk_composition'] = {'H': round(0.75 * (1 - core_frac), 4),
                                     'He': round(0.24 * (1 - core_frac), 4)}
            p['thermal'] = {
                'T_core': round(float(20000 * m_ratio ** 0.3), 0),
                'T_1bar': round(float(max(T_eq_gg, 100 * m_ratio ** 0.1)), 0),
                'KH_age_Gyr': round(float(4.5 * m_ratio ** 0.8), 2),
            }
            p['core_viscosity'] = None

            # Magnetic field: B ∝ (ρ_core × ω)^0.5 (Reiners+2010)
            # Fast rotators (Jupiter: 10h) → strong field
            rot_hr = p.get('rotation_period_hr', 10)
            omega_rel = 10.0 / max(rot_hr, 1)  # relative to Jupiter rotation
            B_surface = 430 * m_ratio ** 0.5 * omega_rel ** 0.5
            p['magnetic'] = {'B_surface_uT': round(float(B_surface), 1),
                             'dynamo_active': True,
                             'field_type': '강한 자기장' if B_surface > 100 else '중간 자기장'}
            p['atmosphere'] = _gas_giant_atmo(p['mass_earth'], T_eq_gg,
                                              metallicity, actual_age)
            p['radiation'] = None
            p['habitability'] = _habitability_assessment(p, p['atmosphere'], None)
            p['habitability_score'] = p['habitability']['score']
            p['moon_system'] = build_moon_system(
                p,
                star_mass=star_mass,
                metallicity_z=metallicity,
                disk=disk,
                current_stellar_mass=c_mass,
                actual_age_gyr=actual_age,
                rng_seed=star_id * 1009 + p['index'] * 131 + 17,
            )
            p['moon_count'] = (
                p['moon_system']['summary']['n_regular']
                + p['moon_system']['summary']['n_irregular']
            )
            p['has_moon_system'] = bool(p['moon_count'] > 0)
            p['status'] = 'active'

        else:  # mini_neptune
            T_eq_mn = phys['T_eq_K']
            p['differentiation'] = None
            p['bulk_composition'] = {el: round(v * 0.85, 8) for el, v in comp.items() if v > 1e-10}
            p['thermal'] = None
            p['core_viscosity'] = None

            # Magnetic field: weaker than gas giants
            rot_hr = p.get('rotation_period_hr', 16)
            omega_rel = 16.0 / max(rot_hr, 1)
            B_surface = 100 * p['mass_earth'] ** 0.3 * omega_rel ** 0.4
            p['magnetic'] = {'B_surface_uT': round(float(B_surface), 1),
                             'dynamo_active': True,
                             'field_type': '중간 자기장' if B_surface > 50 else '약한 자기장'}
            p['atmosphere'] = _sub_neptune_atmo(p['mass_earth'], comp,
                                                T_eq_mn, metallicity, actual_age,
                                                semi_major_au=p['semi_major_au'],
                                                star_teff=star_teff_current,
                                                luminosity_lsun=L_star)
            p['radiation'] = compute_radiation_defense(
                p['mass_earth'], phys['R_mean_Re'], p['magnetic']['B_surface_uT'],
                p['atmosphere']['atm_mass_kg'], p['atmosphere']['surface_pressure_bar'],
                p['atmosphere']['composition'], p['semi_major_au'], star_teff_current,
                L_star_Lsun=L_star, age_gyr=actual_age, flare_boost=flare_boost,
            )
            p['habitability'] = _habitability_assessment(p, p['atmosphere'], p['radiation'])
            p['habitability_score'] = p['habitability']['score']
            p['is_habitable'] = p['habitability']['label'] == 'candidate'
            p['moon_system'] = build_moon_system(
                p,
                star_mass=star_mass,
                metallicity_z=metallicity,
                disk=disk,
                current_stellar_mass=c_mass,
                actual_age_gyr=actual_age,
                rng_seed=star_id * 1009 + p['index'] * 131 + 53,
            )
            p['moon_count'] = (
                p['moon_system']['summary']['n_regular']
                + p['moon_system']['summary']['n_irregular']
            )
            p['has_moon_system'] = bool(p['moon_count'] > 0)
            p['status'] = 'active'

    return {
        'star_id': star_id, 'star_mass': round(star_mass, 4),
        'birth_time': round(birth_time, 3),
        'radius_kpc': round(gce_result['radius'][ir], 1),
        'metallicity': round(metallicity, 6),
        'planet_age_gyr': round(planet_age, 3),
        'n_planets': len(planets), 'planets': planets,
        'stellar_composition': {el: round(v, 8) for el, v in comp.items() if v > 1e-10},
        'disk': disk.to_dict(),
        'asteroid_belt': belt,
    }
