"""
Nucleosynthesis yield tables for all tracked channels.

Each function returns a 1-D array of length N_ELEMENTS giving the
*net* yield (mass of newly synthesised element ejected) per unit mass
of stars formed (CCSNe / AGB) or per event (Ia / NSM).

Values are approximate, based on Kobayashi+20, Iwamoto+99,
Karakas & Lugaro 18, and Rosswog+14.  All yields are tuneable.
"""
import numpy as np
from .config import ELEMENTS, N_ELEMENTS, EL_IDX, Z_SUN

# ============================================================
# Helper: build array from dict
# ============================================================
def _arr(d):
    a = np.zeros(N_ELEMENTS)
    for el, val in d.items():
        if el in EL_IDX:
            a[EL_IDX[el]] = val
    return a

# ============================================================
# 1. Core-Collapse SNe  (IMF-averaged, per M☉ formed)
#    Z-dependent: linear interpolation between Z=0 and Z=Z☉
# ============================================================

_CCSNE_Z0 = {
    'C': 1.4e-3, 'N': 6.0e-5, 'O': 1.2e-2, 'Mg': 1.0e-3,
    'Si': 8.0e-4, 'S': 3.5e-4, 'Fe': 6.0e-4, 'Al': 4.0e-5,
    'P': 4.5e-6, 'Zn': 4.0e-7, 'Mn': 3.5e-6, 'Co': 2.8e-6,
    'Cu': 1.4e-7, 'Ni': 2.2e-5, 'V': 2.8e-7, 'Se': 5.0e-8,
    'Mo': 1.4e-9, 'W': 1.4e-11, 'Ba': 1.4e-9, 'Eu': 0.0,
}
_CCSNE_ZSUN = {
    'C': 2.5e-3, 'N': 6.0e-4, 'O': 8.5e-3, 'Mg': 7.5e-4,
    'Si': 6.5e-4, 'S': 3.2e-4, 'Fe': 7.0e-4, 'Al': 5.5e-5,
    'P': 7.0e-6, 'Zn': 2.0e-6, 'Mn': 8.0e-6, 'Co': 3.5e-6,
    'Cu': 6.0e-7, 'Ni': 2.8e-5, 'V': 4.0e-7, 'Se': 1.5e-7,
    'Mo': 1.0e-8, 'W': 1.0e-9, 'Ba': 2.0e-8, 'Eu': 0.0,
}
_CC0 = _arr(_CCSNE_Z0)
_CCS = _arr(_CCSNE_ZSUN)

def ccsne_yields(Z):
    """IMF-averaged CCSNe net yields. Linear interpolation in Z."""
    f = np.clip(Z / Z_SUN, 0.0, 2.0)
    return _CC0 + (_CCS - _CC0) * f

# Returned fraction from massive stars (>8 M☉): ~0.18 of total mass formed
CCSNE_RETURN_FRAC = 0.18

# He yield from CCSNe: ~0.006 M☉ He produced per M☉ of stars formed
# (Kobayashi+2020 — core He burning + shell He burning in massive stars)
CCSNE_HE_YIELD = 0.006

# ============================================================
# 2. Type Ia SN — per event (~1.4 M☉ ejecta)
#    Metallicity-independent (Iwamoto+99 W7 model)
# ============================================================

TYPE_IA_YIELDS = _arr({
    'C': 4.83e-2, 'O': 1.43e-1, 'Si': 1.54e-1, 'S': 8.6e-2,
    'Fe': 7.44e-1, 'Ni': 1.21e-1, 'Mn': 8.5e-3, 'Co': 6.4e-4,
    'Zn': 1.5e-5, 'Cu': 3.0e-6, 'V': 9.0e-5, 'Al': 1.0e-3,
    'Mg': 8.6e-3, 'P': 3.0e-4, 'N': 0.0,
})

# ============================================================
# 3. AGB stars — per dying star  (mass-dependent, Z-dependent)
#    Bilinear interpolation across representative mass / Z anchors
# ============================================================
_AGB_MASS_GRID = np.array([1.0, 1.5, 2.5, 4.0, 6.0, 8.0], dtype=float)
_AGB_Z_GRID = Z_SUN * np.array([0.05, 0.30, 1.00, 2.00], dtype=float)

# These anchors encode broad literature trends: low/intermediate-mass AGB
# stars enrich C and heavy-s elements, while higher-mass AGB / super-AGB
# stars favour N and Al through hot-bottom burning.
_AGB_ELEMENT_TABLES = {
    'C': np.array([
        [1.5e-3, 4.5e-3, 1.30e-2, 8.0e-3, 3.0e-3, 1.2e-3],
        [2.0e-3, 5.5e-3, 1.40e-2, 7.5e-3, 2.8e-3, 1.1e-3],
        [2.4e-3, 6.0e-3, 1.20e-2, 6.0e-3, 2.2e-3, 9.0e-4],
        [2.1e-3, 5.2e-3, 9.0e-3, 4.5e-3, 1.8e-3, 8.0e-4],
    ], dtype=float),
    'N': np.array([
        [3.0e-4, 5.0e-4, 1.0e-3, 4.0e-3, 7.0e-3, 9.0e-3],
        [5.0e-4, 8.0e-4, 1.4e-3, 5.5e-3, 9.0e-3, 1.1e-2],
        [7.0e-4, 1.0e-3, 2.0e-3, 7.0e-3, 1.1e-2, 1.3e-2],
        [1.0e-3, 1.4e-3, 2.8e-3, 8.5e-3, 1.3e-2, 1.6e-2],
    ], dtype=float),
    'Ba': np.array([
        [1.0e-9, 3.0e-9, 9.0e-9, 4.0e-9, 1.2e-9, 5.0e-10],
        [2.0e-9, 6.0e-9, 2.2e-8, 1.1e-8, 3.5e-9, 1.2e-9],
        [1.5e-9, 4.5e-9, 1.4e-8, 7.0e-9, 2.5e-9, 9.0e-10],
        [8.0e-10, 2.4e-9, 7.0e-9, 3.5e-9, 1.2e-9, 4.0e-10],
    ], dtype=float),
    'Mo': np.array([
        [6.0e-11, 1.8e-10, 7.0e-10, 3.0e-10, 1.1e-10, 4.0e-11],
        [1.5e-10, 4.0e-10, 1.6e-9, 7.0e-10, 2.4e-10, 8.0e-11],
        [1.1e-10, 3.0e-10, 1.0e-9, 4.5e-10, 1.6e-10, 6.0e-11],
        [6.0e-11, 1.7e-10, 5.0e-10, 2.4e-10, 8.0e-11, 3.0e-11],
    ], dtype=float),
    'Se': np.array([
        [2.0e-10, 6.0e-10, 1.2e-9, 1.0e-9, 4.0e-10, 1.2e-10],
        [4.0e-10, 1.0e-9, 2.4e-9, 2.0e-9, 8.0e-10, 2.5e-10],
        [3.0e-10, 8.0e-10, 1.8e-9, 1.4e-9, 6.0e-10, 2.0e-10],
        [1.6e-10, 4.0e-10, 9.0e-10, 7.0e-10, 3.0e-10, 1.0e-10],
    ], dtype=float),
    'W': np.array([
        [2.0e-11, 5.0e-11, 1.0e-10, 6.0e-11, 2.0e-11, 8.0e-12],
        [5.0e-11, 1.0e-10, 2.3e-10, 1.2e-10, 4.0e-11, 1.5e-11],
        [4.0e-11, 8.0e-11, 1.6e-10, 8.0e-11, 3.0e-11, 1.1e-11],
        [2.0e-11, 4.0e-11, 8.0e-11, 4.0e-11, 1.5e-11, 6.0e-12],
    ], dtype=float),
    'Al': np.array([
        [0.0, 0.0, 5.0e-7, 2.0e-6, 4.0e-6, 6.0e-6],
        [0.0, 0.0, 8.0e-7, 3.0e-6, 6.0e-6, 9.0e-6],
        [0.0, 0.0, 1.0e-6, 4.0e-6, 8.0e-6, 1.2e-5],
        [0.0, 0.0, 1.5e-6, 5.0e-6, 1.0e-5, 1.5e-5],
    ], dtype=float),
}


def _interp_agb_table(table, m_star, Z):
    m_val = float(np.clip(m_star, _AGB_MASS_GRID[0], _AGB_MASS_GRID[-1]))
    z_val = float(np.clip(Z, _AGB_Z_GRID[0], _AGB_Z_GRID[-1]))
    mass_interp = np.array([
        np.interp(m_val, _AGB_MASS_GRID, z_row) for z_row in table
    ], dtype=float)
    return float(np.interp(z_val, _AGB_Z_GRID, mass_interp))


def agb_yields(m_star, Z, yield_s_multiplier=1.0):
    """Net yields from a single AGB star of initial mass m_star.

    The previous step-function model introduced artificial discontinuities at
    1.5 and 4 Msun. This version interpolates across representative mass and
    metallicity anchors so the solver samples smoother AGB behaviour.
    """
    y = np.zeros(N_ELEMENTS)
    for el in ('C', 'N', 'Al'):
        y[EL_IDX[el]] = _interp_agb_table(_AGB_ELEMENT_TABLES[el], m_star, Z)
    for el in ('Ba', 'Mo', 'Se', 'W'):
        y[EL_IDX[el]] = _interp_agb_table(_AGB_ELEMENT_TABLES[el], m_star, Z) * yield_s_multiplier
    return y

# ============================================================
# 4. Neutron-star mergers (r-process) — per event
#    Based on Rosswog+14, GW170817 kilonova
# ============================================================

def nsm_yields(ejecta_mass=0.03, yield_r_multiplier=1.0):
    """r-process yields per merger event, scaled to total ejecta_mass."""
    # Relative fractions within r-process ejecta (Rosswog+14)
    fracs = {
        'Se': 1.5e-3, 'Mo': 4.0e-4, 'Ba': 5.0e-4, 'Eu': 7.0e-4,
        'W':  5.0e-4, 'Zn': 1.0e-3, 'Co': 4.0e-4, 'Ni': 3.0e-3,
        'Fe': 5.0e-3,
    }
    y = np.zeros(N_ELEMENTS)
    for el, frac in fracs.items():
        if el in EL_IDX:
            y[EL_IDX[el]] = ejecta_mass * frac * yield_r_multiplier
    return y


# ============================================================
# 5. Collapsar / Jet-driven SNe (r-process) — per event
#    Lighter r-process elements dominant (Siegel+2019)
# ============================================================

def collapsar_yields(ejecta_mass=0.05, yield_r_multiplier=1.0):
    """r-process yields from collapsar / jet-SNe.

    Collapsars produce lighter r-process elements (1st peak)
    more efficiently than NSM, with less actinide production.
    Ref: Siegel, Barnes & Metzger 2019, Nature 569, 241.
    """
    fracs = {
        'Se': 3.0e-3, 'Mo': 1.0e-3, 'Zn': 2.0e-3,   # lighter r-process
        'Ba': 2.0e-4, 'Eu': 1.5e-4,                   # less heavy r than NSM
        'W':  1.0e-4, 'Co': 5.0e-4, 'Ni': 2.0e-3,
        'Fe': 8.0e-3,
    }
    y = np.zeros(N_ELEMENTS)
    for el, frac in fracs.items():
        if el in EL_IDX:
            y[EL_IDX[el]] = ejecta_mass * frac * yield_r_multiplier
    return y
