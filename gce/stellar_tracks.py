"""Mass-regime-specific stellar evolution tracks."""

import numpy as np

from .config import Z_SUN
from .stellar_track_provider import get_precise_track_state, resolve_stellar_model
from .stellar_properties import (
    PHASE_KR,
    _giant_track_scales,
    _massive_track_scales,
    _ms_lifetime,
    _ms_state,
    _spectral_class,
    _temp_to_color,
)
from .stellar_remnants import bh_state, ns_cooling, wd_cooling


def _format_state(age_gyr, initial_mass, phase, temperature, luminosity, radius_rsun,
                  current_mass=None, max_radius_rsun=None, flare_activity=0):
    """Normalize a stellar state into the public API payload."""
    current_mass = initial_mass if current_mass is None else current_mass
    max_radius_rsun = radius_rsun if max_radius_rsun is None else max_radius_rsun

    univ_cmb_temp = 2.73 * (13.8 / max(age_gyr, 13.8)) ** (1 / 3)
    temperature = max(float(temperature), univ_cmb_temp)
    luminosity = max(float(luminosity), 1e-15)
    radius_rsun = max(float(radius_rsun), 1e-5)

    return {
        'phase': phase,
        'phase_kr': PHASE_KR.get(phase, phase),
        'T_eff': round(temperature, 2) if temperature < 100 else int(temperature),
        'luminosity': round(luminosity, 6),
        'radius': round(radius_rsun, 5),
        'color': _temp_to_color(temperature),
        'spectral_class': _spectral_class(temperature, luminosity, phase),
        'log_g': round(np.log10(max(current_mass / radius_rsun**2 * 274, 0.01)), 2),
        'abs_mag': round(-2.5 * np.log10(max(luminosity, 1e-7)) + 4.83, 2),
        'current_mass': round(float(current_mass), 5),
        'max_radius_au': round(float(max_radius_rsun) * 0.00465, 5),
        'flare_activity': flare_activity,
    }


def _format_raw_state(age_gyr, initial_mass, raw_state):
    """Format a provider-specific raw state into the public payload."""
    return _format_state(
        age_gyr,
        initial_mass,
        raw_state['phase'],
        raw_state['temperature'],
        raw_state['luminosity'],
        raw_state['radius_rsun'],
        current_mass=raw_state.get('current_mass'),
        max_radius_rsun=raw_state.get('max_radius_rsun'),
        flare_activity=raw_state.get('flare_activity', 0),
    )


def _apply_solar_ms_calibration(state, mass, age_gyr, metallicity_z):
    """Nudge solar-like MS stars toward the present-day solar anchor.

    The bundled "precise" demo pack was generated from an older heuristic
    baseline and tends to leave 1 Msun stars slightly too hot and bright around
    the Sun's current age. Apply a narrow correction only for near-solar,
    main-sequence stars so the public API stays anchored around the observed Sun
    without affecting the broader grid.
    """
    if state.get('phase') != 'MS':
        return state

    z = float(np.clip(metallicity_z, 1e-6, 0.06))
    t_ms = _ms_lifetime(mass, metallicity_z=z)
    frac = np.clip(age_gyr / max(t_ms, 1e-9), 0.0, 1.0)
    mass_weight = float(np.exp(-((mass - 1.0) / 0.18) ** 2))
    z_weight = float(np.exp(-(np.log10(z / Z_SUN) / 0.35) ** 2))
    age_weight = float(np.exp(-((frac - 0.45) / 0.28) ** 2))
    weight = mass_weight * z_weight * age_weight
    if weight < 1e-3:
        return state

    corrected = dict(state)
    luminosity = float(state['luminosity']) * (1.0 - 0.22 * weight)
    radius = float(state['radius']) * (1.0 - 0.055 * weight)
    temperature = float(state['T_eff']) * (1.0 - 0.04 * weight)

    corrected['luminosity'] = round(luminosity, 6)
    corrected['radius'] = round(radius, 5)
    corrected['T_eff'] = round(temperature, 2) if temperature < 100 else int(temperature)
    corrected['color'] = _temp_to_color(temperature)
    corrected['spectral_class'] = _spectral_class(temperature, luminosity, corrected['phase'])
    current_mass = float(corrected.get('current_mass', mass))
    corrected['log_g'] = round(np.log10(max(current_mass / radius**2 * 274, 0.01)), 2)
    corrected['abs_mag'] = round(-2.5 * np.log10(max(luminosity, 1e-7)) + 4.83, 2)
    corrected['max_radius_au'] = round(max(float(corrected.get('max_radius_au', 0.0)), radius * 0.00465), 5)
    return corrected


def _wd_mass_from_initial(initial_mass):
    return min(1.4, max(0.5, initial_mass * 0.15 + 0.4))


def _pre_ms_times(mass, t_ms):
    t_pre = min(0.001, t_ms * 0.05)
    t_disk = min(3e-3 * mass**(-0.5), t_pre * 0.8)
    return t_disk, t_pre


def _evolve_pre_ms(mass, age_gyr, t_ms, lum_ms, temp_ms, radius_ms):
    t_disk, t_pre = _pre_ms_times(mass, t_ms)
    if age_gyr < t_disk:
        frac = age_gyr / max(t_disk, 1e-9)
        temperature = 2500 + (3000 - 2500) * frac ** 0.3
        luminosity = lum_ms * (3 * (1 - frac) + 0.5)
        radius = radius_ms * (4 * (1 - frac) + 1.5)
        return _format_state(age_gyr, mass, 'disk', temperature, luminosity, radius)

    if age_gyr < t_disk + t_pre:
        frac = (age_gyr - t_disk) / max(t_pre, 1e-9)
        temperature = 3000 + (temp_ms - 3000) * frac ** 0.5
        luminosity = lum_ms * (5 * (1 - frac) + 1)
        radius = radius_ms * (3 * (1 - frac) + 1)
        return _format_state(age_gyr, mass, 'pre-MS', temperature, luminosity, radius)
    return None


def _evolve_low_mass(mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms):
    flare = max(0, 10 - age_gyr * 5) if mass < 0.3 else max(0, 5 - age_gyr * 2)
    if frac < 1.0:
        luminosity = lum_ms * (1 + 0.1 * frac)
        return _format_state(age_gyr, mass, 'MS', temp_ms, luminosity, radius_ms, flare_activity=flare)

    blue_frac = (frac - 1.0) / 0.5
    if blue_frac < 1.0:
        temperature = temp_ms + 3000 * blue_frac
        luminosity = lum_ms * (1.1 + 2 * blue_frac)
        return _format_state(age_gyr, mass, 'blue_dwarf', temperature, luminosity, radius_ms * 0.9)

    cool_age = max(1e-4, age_gyr - t_ms * 1.5)
    wd = wd_cooling(cool_age, wd_mass=mass, include_crystallization=False, luminosity_cap=0.05)
    return _format_state(
        age_gyr,
        mass,
        wd['phase'],
        wd['temperature'],
        wd['luminosity'],
        wd['radius_rsun'],
        current_mass=mass,
        max_radius_rsun=radius_ms,
    )


def _evolve_intermediate(mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms,
                         giant_l_scale, giant_t_scale, giant_r_scale):
    if frac < 0.85:
        luminosity = lum_ms * (1 + 0.5 * frac)
        temperature = temp_ms * (1 + 0.03 * frac)
        radius = radius_ms * (1 + 0.2 * frac)
        return _format_state(age_gyr, mass, 'MS', temperature, luminosity, radius)

    if frac < 0.95:
        frac_sub = (frac - 0.85) / 0.1
        luminosity = lum_ms * (1.4 + 1.5 * frac_sub)
        temperature = temp_ms * (1.0 - 0.25 * frac_sub)
        radius = radius_ms * (1.2 + 3 * frac_sub)
        return _format_state(age_gyr, mass, 'subgiant', temperature, luminosity, radius, max_radius_rsun=radius)

    if frac < 1.05:
        frac_rgb = (frac - 0.95) / 0.1
        luminosity = lum_ms * (3 + 100 * frac_rgb ** 1.5) * giant_l_scale
        temperature = max((3200 + 500 * (1 - frac_rgb)) * giant_t_scale, 3000)
        radius = radius_ms * (4 + 80 * frac_rgb ** 1.5) * giant_r_scale
        current_mass = mass * (1 - 0.1 * frac_rgb)
        return _format_state(age_gyr, mass, 'RGB', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=radius)

    if frac < 1.12:
        frac_hb = (frac - 1.05) / 0.07
        luminosity = lum_ms * 40 * giant_l_scale
        temperature = (5000 + 3000 * np.sin(frac_hb * np.pi)) * giant_t_scale
        radius = radius_ms * 10 * giant_r_scale
        current_mass = mass * 0.9
        max_radius = radius_ms * 84 * giant_r_scale
        return _format_state(age_gyr, mass, 'HB', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=max_radius)

    if frac < 1.22:
        frac_agb = (frac - 1.12) / 0.1
        luminosity = lum_ms * (40 + 120 * frac_agb) * giant_l_scale
        temperature = max(3000, temp_ms * 0.45 * giant_t_scale)
        radius = radius_ms * (12 + 100 * frac_agb) * giant_r_scale
        phase = 'AGB' if frac_agb < 0.9 else 'post-AGB'
        wd_mass = _wd_mass_from_initial(mass)
        current_mass = mass * 0.9 - (mass * 0.9 - wd_mass) * frac_agb
        max_radius = max(radius_ms * 84 * giant_r_scale, radius)
        return _format_state(age_gyr, mass, phase, temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=max_radius)

    wd_mass = _wd_mass_from_initial(mass)
    if frac < 1.25:
        frac_pn = (frac - 1.22) / 0.03
        temperature = 30000 * frac_pn + 5000
        luminosity = max(0.1 * (1 - frac_pn) + 0.01, 0.005)
        radius = 0.1 * (1 - frac_pn) + 0.012
        max_radius = radius_ms * 112 * giant_r_scale
        phase = 'PN' if frac_pn < 0.5 else 'WD'
        return _format_state(age_gyr, mass, phase, temperature, luminosity, radius, current_mass=wd_mass, max_radius_rsun=max_radius)

    cool_age = max(1e-4, age_gyr - t_ms * 1.25)
    wd = wd_cooling(cool_age, wd_mass=wd_mass, include_crystallization=True, luminosity_cap=0.05)
    max_radius = radius_ms * 112 * giant_r_scale
    return _format_state(
        age_gyr,
        mass,
        wd['phase'],
        wd['temperature'],
        wd['luminosity'],
        wd['radius_rsun'],
        current_mass=wd_mass,
        max_radius_rsun=max_radius,
    )


def _evolve_massive(mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms,
                    wind_scale, massive_t_scale, massive_r_scale):
    if frac < 0.8:
        luminosity = lum_ms * (1 + 0.3 * frac)
        temperature = temp_ms
        radius = radius_ms * (1 + 0.15 * frac)
        return _format_state(age_gyr, mass, 'MS', temperature, luminosity, radius)

    if frac < 0.9:
        frac_bsg = (frac - 0.8) / 0.1
        luminosity = lum_ms * (1.3 + 2 * frac_bsg)
        temperature = temp_ms * (0.95 - 0.1 * frac_bsg) * massive_t_scale
        radius = radius_ms * (1.2 + 5 * frac_bsg) * massive_r_scale
        return _format_state(age_gyr, mass, 'BSG', temperature, luminosity, radius, max_radius_rsun=radius)

    if frac < 1.0:
        frac_rsg = (frac - 0.9) / 0.1
        luminosity = lum_ms * (3 + 5 * frac_rsg)
        temperature = max(3500, temp_ms * (0.6 - 0.2 * frac_rsg) * massive_t_scale)
        radius = radius_ms * (6 + 200 * frac_rsg) * massive_r_scale
        current_mass = mass * (1 - min(0.55, 0.25 * wind_scale * frac_rsg))
        return _format_state(age_gyr, mass, 'RSG', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=radius)

    max_radius = radius_ms * 206 * massive_r_scale
    if frac < 1.02:
        frac_sn = (frac - 1.0) / 0.02
        if frac_sn < 0.1:
            pre_sn_mass = mass * max(0.25, 1 - 0.35 * wind_scale)
            return _format_state(age_gyr, mass, 'SN', 50000, lum_ms * 1e4, radius_ms * 500, current_mass=pre_sn_mass, max_radius_rsun=max_radius)
        return _format_state(age_gyr, mass, 'NS', 1e6, 0.0001, 1e-5, current_mass=1.4, max_radius_rsun=max_radius)

    cool_age = max(1e-9, age_gyr - t_ms)
    ns = ns_cooling(cool_age)
    return _format_state(
        age_gyr,
        mass,
        ns['phase'],
        ns['temperature'],
        ns['luminosity'],
        ns['radius_rsun'],
        current_mass=ns['current_mass'],
        max_radius_rsun=max_radius,
    )


def _evolve_very_massive(mass, age_gyr, frac, lum_ms, temp_ms, radius_ms,
                         wind_scale, massive_t_scale, massive_r_scale, wr_t_scale):
    if frac < 0.7:
        luminosity = lum_ms * (1 + 0.2 * frac)
        return _format_state(age_gyr, mass, 'MS', temp_ms, luminosity, radius_ms)

    if frac < 0.8:
        frac_ysg = (frac - 0.7) / 0.1
        luminosity = lum_ms * 1.2 * (1 + frac_ysg)
        temperature = temp_ms * (0.9 - 0.3 * frac_ysg) * massive_t_scale
        radius = radius_ms * (1 + 50 * frac_ysg) * massive_r_scale
        current_mass = mass * max(0.2, 1 - 0.2 * wind_scale - 0.15 * wind_scale * frac_ysg)
        phase = 'RSG' if temperature < 7000 else 'YSG'
        return _format_state(age_gyr, mass, phase, temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=radius)

    if frac < 0.9:
        frac_wr = (frac - 0.8) / 0.1
        luminosity = lum_ms * 1.5
        temperature = (50000 + 30000 * frac_wr) * wr_t_scale
        radius = radius_ms * 0.5 / max(wind_scale ** 0.2, 0.8)
        current_mass = mass * max(0.12, 0.7 - 0.25 * wind_scale - 0.15 * wind_scale * frac_wr)
        max_radius = radius_ms * 51 * massive_r_scale
        return _format_state(age_gyr, mass, 'WR', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=max_radius)

    max_radius = radius_ms * 51 * massive_r_scale
    bh_mass = max(3.0, mass * np.clip(0.20 / max(wind_scale ** 0.4, 0.6), 0.08, 0.35))
    if frac < 0.95:
        frac_sn = (frac - 0.9) / 0.05
        if frac_sn < 0.05:
            pre_sn_mass = mass * max(0.15, 0.35 / max(wind_scale ** 0.3, 0.7))
            return _format_state(age_gyr, mass, 'SN', 80000, lum_ms * 1e5, radius_ms * 2000, current_mass=pre_sn_mass, max_radius_rsun=max_radius)
        bh = bh_state(bh_mass)
        return _format_state(age_gyr, mass, bh['phase'], bh['temperature'], bh['luminosity'], bh['radius_rsun'], current_mass=bh['current_mass'], max_radius_rsun=max_radius)

    bh = bh_state(bh_mass)
    return _format_state(age_gyr, mass, bh['phase'], bh['temperature'], bh['luminosity'], bh['radius_rsun'], current_mass=bh['current_mass'], max_radius_rsun=max_radius)


def _evolve_hypergiant(mass, age_gyr, frac, lum_ms, temp_ms, radius_ms,
                       wind_scale, massive_t_scale, massive_r_scale, wr_t_scale):
    if frac < 0.5:
        luminosity = lum_ms * (1 + 0.1 * frac)
        return _format_state(age_gyr, mass, 'MS', temp_ms, luminosity, radius_ms)

    if frac < 0.65:
        frac_lbv = (frac - 0.5) / 0.15
        luminosity = lum_ms * 1.1
        temperature = temp_ms * (1.0 + 0.1 * np.sin(frac_lbv * 10)) * massive_t_scale
        radius = radius_ms * (1 + 2 * frac_lbv) * massive_r_scale
        current_mass = mass * max(0.35, 1 - 0.15 * wind_scale - 0.1 * wind_scale * frac_lbv)
        return _format_state(age_gyr, mass, 'LBV', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=radius)

    if frac < 0.8:
        frac_hyper = (frac - 0.65) / 0.15
        radius = radius_ms * (3 + 20 * frac_hyper) * massive_r_scale
        current_mass = mass * max(0.2, 0.8 - 0.2 * wind_scale - 0.15 * wind_scale * frac_hyper)
        temperature = max(6000, temp_ms * 0.7 * massive_t_scale)
        luminosity = lum_ms * 1.2
        return _format_state(age_gyr, mass, 'hypergiant', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=radius)

    if frac < 0.9:
        frac_wr = (frac - 0.8) / 0.1
        radius = radius_ms * 0.3 / max(wind_scale ** 0.2, 0.8)
        current_mass = mass * max(0.1, 0.5 - 0.2 * wind_scale - 0.1 * wind_scale * frac_wr)
        max_radius = radius_ms * 23 * massive_r_scale
        temperature = (60000 + 40000 * frac_wr) * wr_t_scale
        luminosity = lum_ms * 1.0
        return _format_state(age_gyr, mass, 'WR', temperature, luminosity, radius, current_mass=current_mass, max_radius_rsun=max_radius)

    max_radius = radius_ms * 23 * massive_r_scale
    bh_mass = max(5.0, mass * np.clip(0.12 / max(wind_scale ** 0.5, 0.6), 0.05, 0.20))
    if frac < 0.92:
        frac_sn = (frac - 0.9) / 0.02
        if frac_sn < 0.05:
            pre_sn_mass = mass * max(0.1, 0.18 / max(wind_scale ** 0.4, 0.6))
            return _format_state(age_gyr, mass, 'SN', 100000, lum_ms * 1e6, radius_ms * 5000, current_mass=pre_sn_mass, max_radius_rsun=max_radius)
        bh = bh_state(bh_mass)
        return _format_state(age_gyr, mass, bh['phase'], bh['temperature'], bh['luminosity'], bh['radius_rsun'], current_mass=bh['current_mass'], max_radius_rsun=max_radius)

    bh = bh_state(bh_mass)
    return _format_state(age_gyr, mass, bh['phase'], bh['temperature'], bh['luminosity'], bh['radius_rsun'], current_mass=bh['current_mass'], max_radius_rsun=max_radius)


def _heuristic_stellar_evolution(mass, age_gyr, metallicity_z=0.02):
    """Compute the original heuristic stellar state."""
    t_ms = _ms_lifetime(mass, metallicity_z=metallicity_z)
    lum_ms, temp_ms, radius_ms = _ms_state(mass, metallicity_z=metallicity_z)
    giant_l_scale, giant_t_scale, giant_r_scale = _giant_track_scales(metallicity_z)
    wind_scale, massive_t_scale, massive_r_scale, wr_t_scale = _massive_track_scales(metallicity_z)
    frac = age_gyr / t_ms if t_ms > 0 else 999

    pre_ms = _evolve_pre_ms(mass, age_gyr, t_ms, lum_ms, temp_ms, radius_ms)
    if pre_ms is not None:
        return pre_ms

    if mass < 0.45:
        return _evolve_low_mass(mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms)
    if mass < 8:
        return _evolve_intermediate(
            mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms,
            giant_l_scale, giant_t_scale, giant_r_scale,
        )
    if mass < 25:
        return _evolve_massive(
            mass, age_gyr, frac, t_ms, lum_ms, temp_ms, radius_ms,
            wind_scale, massive_t_scale, massive_r_scale,
        )
    if mass < 60:
        return _evolve_very_massive(
            mass, age_gyr, frac, lum_ms, temp_ms, radius_ms,
            wind_scale, massive_t_scale, massive_r_scale, wr_t_scale,
        )
    return _evolve_hypergiant(
        mass, age_gyr, frac, lum_ms, temp_ms, radius_ms,
        wind_scale, massive_t_scale, massive_r_scale, wr_t_scale,
    )


def stellar_evolution(mass, age_gyr, metallicity_z=0.02, model=None):
    """Compute stellar properties at the requested age."""
    resolved_model = resolve_stellar_model(model)
    if resolved_model in ('auto', 'precise'):
        raw_state = get_precise_track_state(mass, age_gyr, metallicity_z)
        if raw_state is not None:
            return _apply_solar_ms_calibration(
                _format_raw_state(age_gyr, mass, raw_state),
                mass,
                age_gyr,
                metallicity_z,
            )
    return _apply_solar_ms_calibration(
        _heuristic_stellar_evolution(mass, age_gyr, metallicity_z=metallicity_z),
        mass,
        age_gyr,
        metallicity_z,
    )


__all__ = ['stellar_evolution']
