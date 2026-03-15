"""
UV Photolysis Model — Stellar-type-dependent photodissociation rates.

Computes species-specific photolysis lifetimes (τ) at three atmospheric
levels (upper / mid / column-integrated) using:
  1. Stellar UV template spectra parameterized by T_eff, age, activity
  2. Photoabsorption cross-sections for 12 molecular species
  3. Column-density shielding with self-shielding for N₂, CO, H₂
  4. J-rate integration over 30 wavelength bins (10–400 nm)

References:
  Ribas+2005, ApJ 622      — XUV evolution L_XUV ∝ t^(-1.23)
  Loyd+2018, ApJ 867       — M-dwarf UV and flare activity
  France+2013, ApJ 763     — UV spectra of M/K dwarfs
  Visser+2009, A&A 503     — CO self-shielding functions
  Li+2013, J.Quant.Spec.   — N₂ self-shielding
  Draine & Bertoldi 1996   — H₂ self-shielding
  Yoshino+1996             — H₂O UV cross-section
  Huestis+2010             — CO₂ UV cross-section
  Mount+1977               — CH₄ UV cross-section
  Chen+1999                — NH₃ UV cross-section
  Rufus+2003               — SO₂ UV cross-section
  Molina & Molina 1986     — O₃ Hartley band
  Vandaele+1998            — NO₂ visible absorption
  Bains+2020               — PH₃ UV cross-section
"""

import numpy as np

# ============================================================
# WAVELENGTH GRID — 30 bins from 10 to 400 nm
# ============================================================
_BIN_EDGES = np.array([
    10, 20, 30, 50, 70, 91.2,
    100, 110, 121.6, 130, 140,
    150, 160, 170, 180, 190, 200,
    210, 230, 250, 270, 290, 310,
    320, 340, 360, 380, 400,
    420, 450, 500
])
NBIN = len(_BIN_EDGES) - 1
BIN_CENTERS = 0.5 * (_BIN_EDGES[:-1] + _BIN_EDGES[1:])
BIN_WIDTHS = np.diff(_BIN_EDGES)


# ============================================================
# STELLAR UV TEMPLATES
# ============================================================

def _planck_photon_flux(T_eff, lam_nm, luminosity_lsun=None, radius_rsun=None):
    """Planck photon flux at 1 AU (photons/cm^2/s/nm)."""
    h = 6.626e-34; c = 2.998e8; k = 1.381e-23
    lam_m = lam_nm * 1e-9
    x = np.clip(h * c / (lam_m * k * T_eff), 0, 500)
    B_lam = 2 * h * c**2 / (lam_m**5) / (np.expm1(x) + 1e-30)
    if radius_rsun is not None:
        R_star = float(radius_rsun)
    elif luminosity_lsun is not None and T_eff > 0:
        T_rel = max(T_eff / 5778.0, 0.05)
        R_star = np.sqrt(max(float(luminosity_lsun), 1e-8)) / (T_rel**2)
    elif T_eff > 7000:
        R_star = 1.5
    elif T_eff > 5200:
        R_star = 1.0
    elif T_eff > 3700:
        R_star = 0.7
    else:
        R_star = 0.3
    R_star = float(np.clip(R_star, 0.05, 2000.0))
    R_m = R_star * 6.957e8
    AU_m = 1.496e11
    F_lam = np.pi * B_lam * (R_m / AU_m)**2
    E_ph = h * c / lam_m
    return np.maximum(F_lam / E_ph * 1e-4 * 1e-9, 0)


_UV_ACTIVITY_MODELS = {
    # sat_age_gyr, EUV_sat, FUV_sat, MUV_sat, decay_euv, decay_fuv, decay_muv
    'F': (0.08, 10.0, 3.0, 1.2, 1.15, 0.80, 0.35),
    'G': (0.10, 22.0, 5.0, 1.5, 1.23, 0.90, 0.40),
    'K': (0.18, 55.0, 15.0, 2.8, 1.15, 0.85, 0.35),
    'M': (0.60, 320.0, 90.0, 6.0, 0.90, 0.65, 0.22),
}

_XUV_HISTORY_MODELS = {
    # sat_age_gyr, F_xuv_sat@1AU(erg/cm2/s), decay_alpha, pre_ms_duration_gyr, pre_ms_boost
    'F': (0.07, 300.0, 1.20, 0.03, 1.2),
    'G': (0.10, 500.0, 1.23, 0.05, 1.5),
    'K': (0.35, 700.0, 1.05, 0.18, 2.5),
    'M': (0.80, 900.0, 0.90, 0.80, 6.0),
}


def _smooth_band_mask(lam_nm, lo, hi, edge):
    left = 1.0 / (1.0 + np.exp(-(lam_nm - lo) / edge))
    right = 1.0 / (1.0 + np.exp((lam_nm - hi) / edge))
    return left * right


def _activity_boost_profile(lam_nm, spec_class, age_gyr):
    """Continuous UV excess from chromosphere/corona.

    Uses a saturation + power-law decay model inspired by Ribas+2005 and
    extended qualitatively for cooler stars where magnetic activity persists
    longer. This avoids the old step-function age classes.
    """
    sat_age, euv_sat, fuv_sat, muv_sat, alpha_euv, alpha_fuv, alpha_muv = _UV_ACTIVITY_MODELS.get(
        spec_class, _UV_ACTIVITY_MODELS['G']
    )
    age = max(float(age_gyr), 1e-3)

    def _band_boost(sat_boost, alpha):
        if age <= sat_age:
            return sat_boost
        return 1.0 + (sat_boost - 1.0) * (age / sat_age) ** (-alpha)

    euv_boost = _band_boost(euv_sat, alpha_euv)
    fuv_boost = _band_boost(fuv_sat, alpha_fuv)
    muv_boost = _band_boost(muv_sat, alpha_muv)

    euv_mask = _smooth_band_mask(lam_nm, 10.0, 91.2, 4.0)
    fuv_mask = _smooth_band_mask(lam_nm, 91.2, 200.0, 10.0)
    muv_mask = _smooth_band_mask(lam_nm, 200.0, 320.0, 12.0)

    boost = (
        1.0
        + (euv_boost - 1.0) * euv_mask
        + (fuv_boost - 1.0) * fuv_mask
        + (muv_boost - 1.0) * muv_mask
    )
    return np.maximum(boost, 1.0)


def _activity_class(age_gyr, spec_class='G'):
    sat_age = _UV_ACTIVITY_MODELS.get(spec_class, _UV_ACTIVITY_MODELS['G'])[0]
    age = max(float(age_gyr), 1e-3)
    if age <= sat_age:
        return 'saturated'
    if age <= sat_age * 5.0:
        return 'active'
    if age <= sat_age * 20.0:
        return 'moderate'
    return 'quiet'


def _spectral_class(T_eff):
    if T_eff >= 6000: return 'F'
    elif T_eff >= 5200: return 'G'
    elif T_eff >= 3700: return 'K'
    else: return 'M'


def stellar_activity_params(T_eff):
    """Return spectral-class activity parameters for UV/XUV evolution."""
    spec = _spectral_class(T_eff)
    sat_age, euv_sat, fuv_sat, muv_sat, alpha_euv, alpha_fuv, alpha_muv = _UV_ACTIVITY_MODELS.get(
        spec, _UV_ACTIVITY_MODELS['G']
    )
    xuv_sat_age, f_xuv_sat_1au, xuv_decay_alpha, pre_ms_duration_gyr, pre_ms_boost = _XUV_HISTORY_MODELS.get(
        spec, _XUV_HISTORY_MODELS['G']
    )
    return {
        'spec_class': spec,
        'uv_sat_age_gyr': sat_age,
        'euv_sat_boost': euv_sat,
        'fuv_sat_boost': fuv_sat,
        'muv_sat_boost': muv_sat,
        'euv_decay_alpha': alpha_euv,
        'fuv_decay_alpha': alpha_fuv,
        'muv_decay_alpha': alpha_muv,
        'xuv_sat_age_gyr': xuv_sat_age,
        'xuv_sat_flux_1au': f_xuv_sat_1au,
        'xuv_decay_alpha': xuv_decay_alpha,
        'pre_ms_duration_gyr': pre_ms_duration_gyr,
        'pre_ms_boost': pre_ms_boost,
    }


def integrated_xuv_history(T_eff, age_gyr, semi_major_au, luminosity_lsun=1.0):
    """Return integrated XUV history at the planet orbit."""
    params = stellar_activity_params(T_eff)
    age = max(float(age_gyr), 1e-4)
    sat_age = max(params['xuv_sat_age_gyr'], 1e-4)
    alpha = params['xuv_decay_alpha']
    distance = max(float(semi_major_au), 0.01)
    luminosity_scale = max(float(luminosity_lsun), 1e-5) ** 0.65
    f_xuv_sat = params['xuv_sat_flux_1au'] * luminosity_scale / distance**2

    pre_ms_time = min(age, params['pre_ms_duration_gyr'])
    fluence_gyr = pre_ms_time * params['pre_ms_boost']

    active_time = max(age - pre_ms_time, 0.0)
    sat_time = min(active_time, sat_age)
    fluence_gyr += sat_time

    if active_time > sat_age:
        tail_ratio = active_time / sat_age
        if abs(alpha - 1.0) < 1e-6:
            fluence_gyr += sat_age * np.log(max(tail_ratio, 1.0))
        else:
            fluence_gyr += sat_age * ((tail_ratio ** (1.0 - alpha)) - 1.0) / (1.0 - alpha)

    fluence_erg_cm2 = f_xuv_sat * fluence_gyr * 3.15e16
    return {
        'spec_class': params['spec_class'],
        't_sat_gyr': sat_age,
        'pre_ms_duration_gyr': params['pre_ms_duration_gyr'],
        'pre_ms_boost': params['pre_ms_boost'],
        'F_xuv_sat_erg_cm2_s': f_xuv_sat,
        'xuv_fluence_erg_cm2': fluence_erg_cm2,
    }


def estimate_flare_boost(T_eff, age_gyr):
    """Return an activity-weighted UV flare enhancement factor.

    The value is an average enhancement meant for photochemical exposure,
    not a peak single-flare amplitude. K/M dwarfs retain stronger flare-driven
    UV excess over longer timescales than solar-type stars.
    """
    params = stellar_activity_params(T_eff)
    spec = params['spec_class']
    age = max(float(age_gyr), 1e-4)
    sat_age = max(params['uv_sat_age_gyr'], 1e-4)

    if spec == 'M':
        sat_boost = 4.5
        decay_alpha = 0.65
        floor = 1.15
    elif spec == 'K':
        sat_boost = 2.2
        decay_alpha = 0.8
        floor = 1.05
    elif spec == 'G':
        sat_boost = 1.25
        decay_alpha = 1.0
        floor = 1.0
    else:
        sat_boost = 1.1
        decay_alpha = 1.1
        floor = 1.0

    if age <= sat_age:
        boost = sat_boost
    else:
        boost = floor + (sat_boost - floor) * (age / sat_age) ** (-decay_alpha)

    if age <= params['pre_ms_duration_gyr']:
        boost *= 1.1

    max_boost = 8.0 if spec == 'M' else (3.0 if spec == 'K' else 1.5)
    return float(np.clip(boost, 1.0, max_boost))


def build_uv_template(T_eff, age_gyr, semi_major_au, flare_boost=1.0, luminosity_lsun=None, radius_rsun=None):
    """Build stellar UV flux at the planet (photons/cm²/s/nm)."""
    spec = _spectral_class(T_eff)
    activity = _activity_class(age_gyr, spec)
    F_1au = _planck_photon_flux(T_eff, BIN_CENTERS, luminosity_lsun, radius_rsun)
    F_1au *= _activity_boost_profile(BIN_CENTERS, spec, age_gyr)
    if flare_boost > 1.0 and spec in ('K', 'M'):
        flare_excess = flare_boost - 1.0
        spectral_scale = 1.0 if spec == 'M' else 0.55
        flare_mask = BIN_CENTERS < (200 if spec == 'M' else 170)
        F_1au[flare_mask] *= 1.0 + flare_excess * spectral_scale
    r2 = max(semi_major_au, 0.01) ** 2
    F_planet = F_1au / r2
    h = 6.626e-34; c = 2.998e8
    E_ph = h * c / (BIN_CENTERS * 1e-9)
    F_uv_total = float(np.sum(F_planet * BIN_WIDTHS * E_ph * 1e7))
    return {
        'flux': F_planet, 'spec_class': spec, 'activity': activity,
        'label': f'{spec}-{activity}', 'F_UV_total': round(F_uv_total, 2), 'T_eff': T_eff,
        'flare_boost': round(float(flare_boost), 3),
    }


# ============================================================
# PHOTOABSORPTION CROSS-SECTIONS — band-limited σ(λ)
# ============================================================
# Format: (σ_peak cm², λ_center nm, width nm, λ_min nm, λ_max nm)

def _band_xsec(lam_nm, bands):
    """Sum of band-limited Gaussian cross-sections."""
    sigma = np.zeros_like(lam_nm, dtype=float)
    for s_peak, l0, w, lmin, lmax in bands:
        mask = (lam_nm >= lmin) & (lam_nm <= lmax)
        sigma[mask] += s_peak * np.exp(-((lam_nm[mask] - l0) / w) ** 2)
    return sigma

PHOTO_XSEC = {
    # H₂O: Yoshino+1996 — continuum 140-190 nm + Ly-α channel
    # σ(Ly-α) = 1.4e-17 cm² but narrow to avoid overlapping with O₂ SR continuum
    'H2O': [(1.5e-17, 165, 25, 140, 200), (1.4e-17, 121.6, 3, 118, 126)],
    # CO₂: Huestis+2010 — continuum 140-200 nm
    # σ(Ly-α) ≈ 6e-20 cm² (weak absorption at 121.6 nm)
    'CO2': [(3e-17, 150, 20, 135, 210), (6e-20, 121.6, 3, 118, 126)],
    # CH₄: Mount+1977 — primary photolysis via Ly-α (121.6 nm)
    #   σ(Ly-α) = 1.85e-17 cm²
    #   Also absorbs in 130-160 nm but O₂ SR continuum blocks this
    'CH4': [(1.85e-17, 121.6, 3, 118, 126), (2e-17, 145, 12, 135, 165)],
    # NH₃: Chen+1999 — 170-220 nm (A-band, above SR continuum)
    'NH3': [(1e-17, 195, 18, 165, 225), (5e-18, 170, 10, 155, 185)],
    # N₂: Li+2013 — predissociation 80-100 nm ONLY
    'N2':  [(1.5e-17, 90, 6, 78, 102)],
    # CO: Visser+2009 — 91-115 nm
    'CO':  [(2e-17, 105, 8, 90, 118)],
    # H₂: Draine & Bertoldi 1996 — Lyman-Werner 91-110 nm
    'H2':  [(3e-18, 100, 6, 90, 112)],
    # SO₂: Rufus+2003 — 190-230 nm + Clements 250-320 nm
    'SO2': [(1e-17, 210, 15, 185, 235), (4e-18, 290, 30, 240, 340)],
    # H₂S: Grosch+2015 — 190-260 nm
    'H2S': [(6e-18, 230, 25, 180, 270)],
    # O₃: Molina & Molina 1986 — Hartley 200-310 nm
    'O3':  [(1.1e-17, 255, 30, 200, 320), (5e-20, 330, 20, 310, 360)],
    # NO₂: Vandaele+1998 — 300-420 nm
    'NO2': [(6e-19, 370, 40, 300, 430), (4e-19, 320, 25, 290, 350)],
    # PH₃: Bains+2020 — 160-230 nm
    'PH3': [(1e-17, 185, 20, 155, 240)],
    # HCN: 120-190 nm (includes Ly-α channel)
    'HCN': [(5e-18, 155, 20, 130, 195), (3e-18, 121.6, 3, 118, 126)],
    # O₂: Three distinct absorption regions:
    #   1. SR bands (102-136 nm): STRUCTURED line absorption with deep windows.
    #      At Ly-α (121.6 nm): σ ≈ 1e-24 cm² (Yoshino+1992) — essentially
    #      transparent. Using this as the effective band-average since Ly-α
    #      carries most of the photon flux in this region.
    #   2. SR continuum (136-175 nm): smooth, strong σ ≈ 1.5e-17
    #   3. Herzberg continuum (195-245 nm): extremely weak σ ≈ 1e-23
    'O2':  [(1e-24, 121.6, 15, 102, 136),
            (1.5e-17, 150, 12, 136, 175),
            (1e-23, 220, 20, 195, 245)],
    # He: ionization < 50 nm
    'He':  [(7e-18, 30, 10, 10, 55)],
}

# Major UV shields — only these contribute to optical depth
UV_SHIELDS = {'O2', 'O3', 'N2', 'CO2', 'H2O', 'H2', 'CO'}


# ============================================================
# ALTITUDE-RESOLVED J-RATE INTEGRATION
# ============================================================
# Photolysis occurs at different altitudes. UV-labile species like CH₄
# are destroyed primarily in the stratosphere/mesosphere where less O₂
# is overhead. We integrate J-rates over 10 altitude layers from 0 to 5
# scale heights, computing the overhead absorber column at each level.

N_ALT_LAYERS = 10  # altitude layers from z=0 to z=5H

def _self_shielding_H2(N): return min(1.0, 0.965/(1+N/5e14)**2 + 0.035/np.sqrt(1+N/5e14)*np.exp(-8.5e-4*np.sqrt(1+N/5e14))) if N > 1e14 else 1.0
def _self_shielding_CO(N): return float(np.clip(10**(-(np.log10(max(N,1e15))-15)*0.5), 1e-4, 1.0)) if N > 1e15 else 1.0
def _self_shielding_N2(N): return float(np.clip(10**(-(np.log10(max(N,1e16))-16)*0.4), 1e-4, 1.0)) if N > 1e16 else 1.0


def compute_photolysis_rates(uv_template, species_surface_col, species_list=None):
    """Compute J-rates and lifetimes τ at three atmospheric levels.

    Integrates J(z) over 10 altitude layers from surface to 5 scale heights.

    Parameters
    ----------
    uv_template : dict from build_uv_template()
    species_surface_col : dict of species → surface column density N_0 (cm⁻²)
    species_list : optional list of species to compute

    Returns
    -------
    dict: species → {tau_upper_yr, tau_mid_yr, tau_column_yr, ...}
    """
    if species_list is None:
        species_list = list(PHOTO_XSEC.keys())

    F_lam = uv_template['flux']

    # Pre-compute cross-section arrays
    xsec_arrays = {}
    for sp in PHOTO_XSEC:
        xsec_arrays[sp] = _band_xsec(BIN_CENTERS, PHOTO_XSEC[sp])

    # Altitude grid: z/H = 0, 0.5, 1.0, ..., 4.5 (10 layers)
    z_over_H = np.linspace(0, 4.5, N_ALT_LAYERS)

    # Species-specific vertical distribution (relative scale height H_sp / H_atm)
    # H₂O: trapped below tropopause (cold trap), effective H ~ 0.25 * H_atm
    # O₃: concentrated in stratosphere, effective H ~ 3 * H_atm (peaks at ~25 km)
    # Well-mixed gases (N₂, O₂, CO₂, Ar): H_sp = H_atm
    SCALE_HEIGHT_RATIO = {
        'H2O': 0.25,  # tropospheric cold trap (Earth: H_eff ~ 2 km vs H_atm ~ 8 km)
        'O3':  3.0,   # stratospheric ozone layer
        'H2':  1.5,   # lighter, extends higher
        'He':  2.0,   # very light, diffusive separation
    }
    # Default: 1.0 (well-mixed)

    result = {}
    for sp in species_list:
        if sp not in PHOTO_XSEC:
            continue

        sigma_i = xsec_arrays[sp]
        J_profile = np.zeros(N_ALT_LAYERS)

        for iz, z_h in enumerate(z_over_H):
            # Overhead column for each shield species at altitude z
            tau_total = np.zeros(NBIN)
            for absorber in UV_SHIELDS:
                if absorber not in species_surface_col:
                    continue
                N_0 = species_surface_col[absorber]
                if N_0 <= 0:
                    continue
                if absorber not in xsec_arrays:
                    continue

                # Species-specific scale height
                h_ratio = SCALE_HEIGHT_RATIO.get(absorber, 1.0)
                # Column above z: N_0 * exp(-z/H_sp) = N_0 * exp(-z_h / h_ratio)
                N_above = N_0 * np.exp(-z_h / h_ratio)

                shield = 1.0
                if absorber == 'H2':   shield = _self_shielding_H2(N_above)
                elif absorber == 'CO': shield = _self_shielding_CO(N_above)
                elif absorber == 'N2': shield = _self_shielding_N2(N_above)

                tau_total += xsec_arrays[absorber] * N_above * shield

            tau_total = np.clip(tau_total, 0, 300)
            F_atten = F_lam * np.exp(-tau_total)
            J_profile[iz] = float(np.sum(sigma_i * F_atten * BIN_WIDTHS))

        # Upper (top of atmosphere, z/H=4.5): essentially unshielded
        J_upper = J_profile[-1]
        # Mid (z/H ~ 2): partial shielding
        J_mid = J_profile[N_ALT_LAYERS // 2]
        # Column-integrated: density-weighted average
        # Weight by density at each level: ρ(z) ∝ exp(-z/H)
        weights = np.exp(-z_over_H)
        J_column = np.sum(J_profile * weights) / np.sum(weights)

        def _j_to_tau(J):
            return round(1.0 / (J * 3.156e7), 6) if J > 0 else 1e12

        result[sp] = {
            'J_upper': J_upper, 'J_mid': J_mid, 'J_column': J_column,
            'tau_upper_yr': _j_to_tau(J_upper),
            'tau_mid_yr': _j_to_tau(J_mid),
            'tau_column_yr': _j_to_tau(J_column),
        }

    return result


# ============================================================
# HIGH-LEVEL API
# ============================================================

def compute_photolysis(T_eff, age_gyr, semi_major_au, species_mass_kg,
                       R_planet_m, g_surface, scale_height_m, flare_boost=1.0,
                       luminosity_lsun=None, radius_rsun=None):
    """Full photolysis calculation for a rocky planet.

    Computes surface column densities from species masses and passes
    them to the altitude-resolved J-rate integrator.
    """
    uv = build_uv_template(T_eff, age_gyr, semi_major_au, flare_boost, luminosity_lsun, radius_rsun)

    # Compute surface column densities (cm⁻²)
    A_surface = 4 * np.pi * R_planet_m**2
    from gce.planets import MOLECULES

    surface_cols = {}
    for sp, mass_kg in species_mass_kg.items():
        mol = MOLECULES.get(sp)
        if mol is None or mass_kg <= 0:
            continue
        m_mol_kg = mol['M'] * 1.66e-27
        N_total = mass_kg / m_mol_kg
        N_col = N_total / (A_surface * 1e4)  # cm⁻²
        surface_cols[sp] = float(N_col)

    rates = compute_photolysis_rates(uv, surface_cols)

    return {
        'stellar_uv_class': uv['label'],
        'F_UV_total': uv['F_UV_total'],
        'flare_boost': uv.get('flare_boost', 1.0),
        'species': rates,
    }
