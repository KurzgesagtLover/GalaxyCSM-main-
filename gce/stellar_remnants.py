"""Remnant-state helpers for stellar evolution tracks.

References:
  - Mestel 1952, MNRAS 112, 583            : white-dwarf cooling baseline
  - van Horn 1968, ApJ 151, 227            : Debye cooling regime
  - Winget et al. 2009, ApJ 693, L6        : crystallization / latent-heat signature
  - Yakovlev & Pethick 2004, ARA&A 42, 169 : neutron-star cooling review
  - Page et al. 2006, Nucl. Phys. A 777    : minimal cooling paradigm
"""

import numpy as np


def wd_cooling(cool_age_gyr, wd_mass=0.6, include_crystallization=True, luminosity_cap=0.05):
    """White-dwarf cooling track in solar units.

    Returns a dict with phase, luminosity, temperature, radius, and remnant mass.
    The current implementation keeps a fixed WD radius to preserve the behavior
    of the original simulator while isolating the cooling physics for extension.

    The luminosity law is a simulator-oriented Mestel-style cooling sequence with
    optional crystallization bump and late Debye-like steepening.
    """
    cool_age = max(1e-4, float(cool_age_gyr))
    lum_mestel = 0.01 * (cool_age / 0.01) ** (-1.4)
    luminosity = lum_mestel

    if include_crystallization and 2.0 < cool_age < 8.0:
        crystallization_bump = 1.0 + 0.5 * np.sin((cool_age - 2.0) / 6.0 * np.pi)
        luminosity *= crystallization_bump

    if cool_age > 8.0:
        lum_at_8gyr = 0.01 * (8.0 / 0.01) ** (-1.4)
        luminosity = lum_at_8gyr * (cool_age / 8.0) ** (-4.0)

    luminosity = min(float(luminosity), float(luminosity_cap))
    radius_rsun = 0.012
    temperature = 5778.0 * (luminosity / (radius_rsun ** 2)) ** 0.25
    phase = 'Black Dwarf' if temperature <= 500 else 'WD'
    return {
        'phase': phase,
        'luminosity': luminosity,
        'temperature': temperature,
        'radius_rsun': radius_rsun,
        'current_mass': float(wd_mass),
    }


def ns_cooling(cool_age_gyr):
    """Neutron-star cooling track in solar units.

    Uses a compact three-regime approximation: neutrino-dominated cooling,
    transition to superfluid/photon cooling, then late photon cooling.
    """
    t_yr = max(1e-9, float(cool_age_gyr)) * 1e9
    if t_yr < 1e5:
        temperature = 2e6 * (t_yr / 1000.0) ** (-0.1666)
    elif t_yr < 1e7:
        temp_1e5 = 2e6 * (1e5 / 1000.0) ** (-0.1666)
        temperature = temp_1e5 * (t_yr / 1e5) ** (-0.5)
    else:
        temp_1e5 = 2e6 * (1e5 / 1000.0) ** (-0.1666)
        temp_1e7 = temp_1e5 * (1e7 / 1e5) ** (-0.5)
        temperature = temp_1e7 * (t_yr / 1e7) ** (-1.0)

    radius_rsun = 1.4e-5
    luminosity = (radius_rsun ** 2) * (temperature / 5778.0) ** 4
    return {
        'phase': 'NS',
        'luminosity': luminosity,
        'temperature': temperature,
        'radius_rsun': radius_rsun,
        'current_mass': 1.4,
    }


def bh_state(bh_mass):
    """Black-hole endpoint state placeholder for future expansion.

    No radiative evolution is modeled here yet; this function exists so future
    accretion-disk or Hawking-evaporation extensions have a dedicated home.
    """
    return {
        'phase': 'BH',
        'luminosity': 0.0,
        'temperature': 0.0,
        'radius_rsun': 0.0,
        'current_mass': float(bh_mass),
    }


__all__ = ['bh_state', 'ns_cooling', 'wd_cooling']
