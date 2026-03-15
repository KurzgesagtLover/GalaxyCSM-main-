"""
Stellar property helpers shared by evolution tracks and galaxy synthesis.

This module contains only stateless property / scaling functions so that
track builders and visualization code can depend on a small, reusable core.

References:
  - Raiteri et al. 1996, A&A 315, 105      : metallicity-dependent lifetime fit
  - Tout et al. 1996, MNRAS 281, 257       : analytic stellar luminosity/radius scalings
  - Vink et al. 2001, A&A 369, 574         : metallicity scaling of line-driven winds
"""

import numpy as np

from .physics import stellar_lifetime as _stellar_lifetime_model

SPECTRAL = [
    ('O', 16.0, 100.0, 40000, '#92b5ff'),
    ('B', 2.1, 16.0, 20000, '#a2c0ff'),
    ('A', 1.4, 2.1, 8500, '#d5e0ff'),
    ('F', 1.04, 1.4, 6500, '#f8f7ff'),
    ('G', 0.8, 1.04, 5500, '#fff4e8'),
    ('K', 0.45, 0.8, 4000, '#ffd2a1'),
    ('M', 0.08, 0.45, 3200, '#ffcc6f'),
]

PHASE_KR = {
    'molecular_cloud': '분자구름',
    'disk': '원시원반',
    'pre-MS': '전주계열',
    'MS': '주계열',
    'subgiant': '준거성',
    'RGB': '적색거성가지',
    'HB': '수평가지',
    'AGB': '점근거성가지',
    'post-AGB': '후기 AGB',
    'PN': '행성상 성운',
    'WD': '백색왜성',
    'Black Dwarf': '흑색왜성',
    'blue_dwarf': '청색왜성',
    'BSG': '청색초거성',
    'RSG': '적색초거성',
    'YSG': '황색초거성',
    'LBV': '밝은 청색 변광성',
    'WR': '볼프-레이에',
    'hypergiant': '극대거성',
    'SN': '초신성',
    'NS': '중성자별',
    'BH': '블랙홀',
}


def _spectral(mass):
    for sp, lo, hi, temp, color in SPECTRAL:
        if lo <= mass < hi:
            return sp, temp, color
    return ('M', 3200, '#ffcc6f') if mass < 0.45 else ('O', 40000, '#92b5ff')


def _ms_lifetime(mass, metallicity_z=0.02):
    """Main-sequence lifetime wrapper around the Raiteri-style fit."""
    if mass <= 0:
        return 1e4
    tau = float(np.atleast_1d(_stellar_lifetime_model(np.array([mass]), Z=metallicity_z))[0])
    return min(tau, 1e4)


def _ms_lum(mass):
    """Piecewise main-sequence luminosity anchor.

    The exponents are simulator-level approximations informed by canonical
    mass-luminosity relations and kept deliberately simple for fast sampling.
    """
    if mass < 0.43:
        return 0.23 * mass ** 2.3
    if mass < 2:
        return mass ** 4.0
    if mass > 100:
        return 1.5 * mass ** 3.0
    return 1.5 * mass ** 3.5


def _ms_radius(mass):
    """Piecewise main-sequence radius anchor based on standard scaling laws."""
    return mass ** 0.8 if mass < 1 else mass ** 0.57


def _ms_state(mass, metallicity_z=0.02):
    """Approximate Z-dependent main-sequence anchor point for tracks.

    Uses simplified luminosity/radius scalings plus a mild metallicity trend so
    metal-poor stars stay hotter and more compact at fixed mass, consistent with
    the direction seen in standard stellar structure models.
    """
    z = float(np.clip(metallicity_z, 1e-6, 0.06))
    z_ratio = z / 0.02
    lum_exp = -0.08 if mass < 1.2 else -0.15
    rad_exp = 0.04 if mass < 1.2 else 0.08
    lum_scale = float(np.clip(z_ratio ** lum_exp, 0.75, 1.6))
    radius_scale = float(np.clip(z_ratio ** rad_exp, 0.75, 1.35))

    lum_ms = _ms_lum(mass) * lum_scale
    radius_ms = _ms_radius(mass) * radius_scale
    t_anchor = _spectral(mass)[1]
    temp_ms = t_anchor * np.clip(
        (lum_ms / max(radius_ms ** 2, 1e-9)) ** 0.25 / max(t_anchor / 5778.0, 1e-9),
        0.85,
        1.25,
    )
    return lum_ms, temp_ms, radius_ms


def _giant_track_scales(metallicity_z=0.02):
    """Metallicity scaling used for RGB/HB/AGB heuristic track amplitudes."""
    z = float(np.clip(metallicity_z, 1e-6, 0.06))
    z_ratio = z / 0.02
    lum_scale = float(np.clip(z_ratio ** -0.05, 0.85, 1.25))
    radius_scale = float(np.clip(z_ratio ** 0.07, 0.75, 1.35))
    temp_scale = float(np.clip((lum_scale / max(radius_scale ** 2, 1e-9)) ** 0.25, 0.9, 1.15))
    return lum_scale, temp_scale, radius_scale


def _massive_track_scales(metallicity_z=0.02):
    """Simple metallicity scaling for line-driven winds in massive stars.

    The wind exponent is chosen to track the qualitative Vink et al. (2001)
    result that metal-rich massive stars lose mass more efficiently.
    """
    z = float(np.clip(metallicity_z, 1e-6, 0.06))
    z_ratio = z / 0.02
    wind_scale = float(np.clip(z_ratio ** 0.6, 0.25, 2.5))
    radius_scale = float(np.clip(z_ratio ** 0.12, 0.7, 1.5))
    temp_scale = float(np.clip(z_ratio ** -0.05, 0.9, 1.15))
    wr_temp_scale = float(np.clip(z_ratio ** -0.08, 0.9, 1.2))
    return wind_scale, temp_scale, radius_scale, wr_temp_scale


def _temp_to_color(temp_eff):
    if temp_eff > 30000:
        return '#92b5ff'
    if temp_eff > 10000:
        return '#aabfff'
    if temp_eff > 7500:
        return '#d5e0ff'
    if temp_eff > 6000:
        return '#f8f7ff'
    if temp_eff > 5200:
        return '#fff4e8'
    if temp_eff > 3700:
        return '#ffd2a1'
    if temp_eff > 2400:
        return '#ffcc6f'
    return '#ff6030'


def _spectral_class(temp_eff, luminosity, phase):
    """Compute full spectral classification e.g. G2V, K5III.

    This is a presentation-layer mapping from effective temperature and phase to
    MK-like labels; it is intended for UI readability rather than catalog-grade
    stellar classification.
    """
    types = [('O', 30000), ('B', 10000), ('A', 7500), ('F', 6000), ('G', 5200), ('K', 3700), ('M', 2400)]
    sp = 'M'
    for idx, (symbol, temp_min) in enumerate(types):
        if temp_eff >= temp_min:
            sp = symbol
            if idx < len(types) - 1 or idx == 0:
                t_hi = 60000 if idx == 0 else types[idx - 1][1]
                t_lo = temp_min
                subtype = int(9 * (1 - (temp_eff - t_lo) / max(t_hi - t_lo, 1)))
                subtype = max(0, min(9, subtype))
                sp = f"{symbol}{subtype}"
            break

    if phase in ('MS', 'pre-MS'):
        lc = 'V'
    elif phase == 'subgiant':
        lc = 'IV'
    elif phase in ('RGB', 'HB', 'AGB'):
        lc = 'III' if luminosity < 1000 else 'II'
    elif phase in ('BSG', 'RSG', 'YSG', 'LBV', 'hypergiant'):
        lc = 'Ia' if luminosity > 100000 else 'Ib'
    elif phase == 'WR':
        return 'WR'
    elif phase == 'WD':
        return 'D' + sp[0] if len(sp) > 0 else 'DA'
    elif phase in ('NS', 'BH'):
        return phase
    else:
        lc = 'V'
    return sp + lc


__all__ = [
    'PHASE_KR',
    'SPECTRAL',
    '_giant_track_scales',
    '_massive_track_scales',
    '_ms_lifetime',
    '_ms_lum',
    '_ms_radius',
    '_ms_state',
    '_spectral',
    '_spectral_class',
    '_temp_to_color',
]
