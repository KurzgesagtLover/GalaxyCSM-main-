"""Semi-analytic rocky planet moon formation model.

This module models giant-impact-generated rocky moons without running a full
N-body collision history. It is intended for population synthesis and uses
surrogate quantities such as late-veneer intensity, giant-planet stirring,
and Hill-sphere survival limits.

References:
  - Canup 2004, Icarus 168, 433
  - Ida et al. 1997, Nature 389, 353
  - Elser et al. 2011, A&A 529, A128
  - Barr 2016, JGR Planets 121, 1573
  - Quarles et al. 2020, AJ 159, 80
"""

import numpy as np

AU_M = 1.496e11
M_EARTH = 5.972e24
R_EARTH = 6.371e6
M_SUN = 1.989e30
G_CONST = 6.674e-11


def _clip(value, lo, hi):
    return float(np.clip(float(value), lo, hi))


def _planet_radius_re(mass_earth):
    return max(float(mass_earth), 0.01) ** 0.27


def _moon_radius_re(mass_earth, iron_frac, silicate_frac, volatile_frac):
    density = 7200.0 * iron_frac + 3300.0 * silicate_frac + 1400.0 * volatile_frac
    density = max(float(density), 1100.0)
    radius_re = max(float(mass_earth), 1e-10) ** (1.0 / 3.0) * (5515.0 / density) ** (1.0 / 3.0)
    return _clip(radius_re, 0.005, 1.2)


def _orbital_period_days(planet_mass_earth, semi_major_rp, planet_radius_re):
    a_m = semi_major_rp * planet_radius_re * R_EARTH
    m_kg = planet_mass_earth * M_EARTH
    period_s = 2.0 * np.pi * np.sqrt(a_m ** 3 / max(G_CONST * m_kg, 1e-20))
    return period_s / 86400.0


def _giant_neighbor_influence(planet, all_planets):
    if not all_planets:
        return 0.0
    a = max(float(planet.get('semi_major_au', 1.0)), 0.02)
    score = 0.0
    for other in all_planets:
        if other is planet:
            continue
        if other.get('type') not in ('gas_giant', 'mini_neptune'):
            continue
        other_a = max(float(other.get('semi_major_au', 1.0)), 0.02)
        mass_term = _clip(float(other.get('mass_earth', 1.0)) / 100.0, 0.05, 10.0) ** 0.45
        spacing = abs(np.log(a / other_a))
        score = max(score, mass_term * np.exp(-spacing / 0.8))
    return _clip(score, 0.0, 1.5)


def _proxy_late_veneer_score(planet, all_planets):
    a = max(float(planet.get('semi_major_au', 1.0)), 0.03)
    ecc = _clip(planet.get('eccentricity', 0.02), 0.0, 0.8)
    mass = max(float(planet.get('mass_earth', 1.0)), 0.03)
    giant_influence = _giant_neighbor_influence(planet, all_planets)
    score = (
        0.18
        + 0.18 * np.log10(1.0 + mass)
        + 0.22 * giant_influence
        + 0.20 * ecc
        + 0.10 * np.clip(1.0 - 0.35 * np.log10(a / 0.8 + 1e-6), 0.0, 1.0)
    )
    return _clip(score, 0.05, 0.95)


def estimate_impact_state(planet, star_mass, all_planets=None, late_veneer=None,
                          phys=None, current_stellar_mass=None, rng=None):
    """Estimate a last-major-impact state for a rocky planet."""
    if rng is None:
        rng = np.random.default_rng()

    mass_earth = float(planet['mass_earth'])
    semi_major_au = max(float(planet.get('semi_major_au', 1.0)), 0.02)
    ecc = _clip(planet.get('eccentricity', 0.02), 0.0, 0.8)
    planet_radius_re = float((phys or {}).get('R_mean_Re', _planet_radius_re(mass_earth)))
    current_star_mass = max(float(current_stellar_mass or star_mass), 0.08)
    hill_radius_m = semi_major_au * AU_M * ((mass_earth * M_EARTH) / (3.0 * current_star_mass * M_SUN)) ** (1.0 / 3.0)
    hill_radius_rp = hill_radius_m / max(planet_radius_re * R_EARTH, 1.0)
    stable_outer_rp = 0.36 * hill_radius_rp
    rho_planet = 5515.0 * mass_earth / max(planet_radius_re ** 3, 1e-6)
    roche_limit_rp = max(2.7, 2.44 * (rho_planet / 3000.0) ** (1.0 / 3.0))

    giant_influence = _giant_neighbor_influence(planet, all_planets)
    lv_score = float((late_veneer or {}).get('score', _proxy_late_veneer_score(planet, all_planets)))
    stellar_tide_penalty = _clip((semi_major_au / 0.12) ** 0.7, 0.02, 1.0)
    if planet.get('type') == 'hot_rocky':
        stellar_tide_penalty *= 0.55
    stability_factor = _clip((stable_outer_rp - 1.5 * roche_limit_rp) / 55.0, 0.0, 1.0)
    survival_factor = _clip(stability_factor * stellar_tide_penalty, 0.0, 1.0)

    impact_score = _clip(
        0.15
        + 0.55 * lv_score
        + 0.10 * giant_influence
        + 0.08 * np.log10(1.0 + mass_earth)
        + 0.10 * ecc,
        0.0,
        1.0,
    )
    giant_impact_prob = _clip(0.05 + 0.85 * impact_score, 0.0, 0.98)
    impactor_mass_ratio = _clip(
        0.015 + 0.22 * impact_score * rng.lognormal(0.0, 0.35),
        0.01,
        0.45,
    )
    impact_velocity_over_escape = _clip(
        0.95 + 0.45 * giant_influence + 0.25 * ecc + rng.normal(0.0, 0.10),
        0.8,
        2.8,
    )
    impact_angle_deg = _clip(rng.normal(45.0, 18.0), 10.0, 80.0)
    angular_momentum_proxy = _clip(
        impactor_mass_ratio * np.sin(np.radians(impact_angle_deg)) * impact_velocity_over_escape,
        0.0,
        1.0,
    )

    return {
        'late_veneer_score': round(float(lv_score), 4),
        'giant_neighbor_influence': round(float(giant_influence), 4),
        'giant_impact_prob': round(float(giant_impact_prob), 4),
        'impactor_mass_ratio': round(float(impactor_mass_ratio), 4),
        'impact_velocity_over_escape': round(float(impact_velocity_over_escape), 4),
        'impact_angle_deg': round(float(impact_angle_deg), 3),
        'angular_momentum_proxy': round(float(angular_momentum_proxy), 4),
        'hill_radius_rp': round(float(hill_radius_rp), 3),
        'stable_outer_rp': round(float(stable_outer_rp), 3),
        'roche_limit_rp': round(float(roche_limit_rp), 3),
        'survival_factor': round(float(survival_factor), 4),
        'stability_factor': round(float(stability_factor), 4),
        'stellar_tide_penalty': round(float(stellar_tide_penalty), 4),
    }


def _empty_rocky_moon_system(formation_channel='none', impact_state=None):
    return {
        'formation_channel': formation_channel,
        'impact_state': impact_state or {},
        'debris_disk': {
            'disk_mass_fraction': 0.0,
            'disk_mass_earth': 0.0,
            'silicate_fraction': 0.0,
            'iron_fraction': 0.0,
            'volatile_fraction': 0.0,
            'retention_factor': 0.0,
        },
        'summary': {
            'n_major': 0,
            'n_minor': 0,
            'largest_moon_mass_earth': 0.0,
            'total_moon_mass_earth': 0.0,
            'moon_to_planet_mass_ratio': 0.0,
            'binary_like': False,
        },
        'major_moons': [],
        'minor_moons': [],
    }


def _major_family(moon_to_planet_ratio, volatile_frac):
    if moon_to_planet_ratio > 0.05:
        return 'charon_like'
    if volatile_frac > 0.08:
        return 'volatile_impact_moon'
    return 'moon_like'


def build_rocky_moon_system(planet, star_mass, all_planets=None, late_veneer=None,
                            phys=None, current_stellar_mass=None, actual_age_gyr=4.5,
                            rng_seed=None):
    """Build a rocky moon system from a giant-impact surrogate model."""
    rng = np.random.default_rng(rng_seed)
    impact = estimate_impact_state(
        planet,
        star_mass=star_mass,
        all_planets=all_planets,
        late_veneer=late_veneer,
        phys=phys,
        current_stellar_mass=current_stellar_mass,
        rng=rng,
    )
    if impact['stable_outer_rp'] <= impact['roche_limit_rp'] * 1.2 or impact['survival_factor'] <= 0.01:
        return _empty_rocky_moon_system('none', impact_state=impact)

    force_formation = (
        impact['late_veneer_score'] >= 0.75
        and impact['giant_impact_prob'] >= 0.55
        and impact['survival_factor'] >= 0.35
    )
    if not force_formation and rng.random() > impact['giant_impact_prob']:
        return _empty_rocky_moon_system('none', impact_state=impact)

    theta = np.radians(impact['impact_angle_deg'])
    mass_earth = float(planet['mass_earth'])
    disk_mass_fraction = (
        0.055
        * impact['impactor_mass_ratio'] ** 0.85
        * np.sin(theta) ** 1.8
        * np.exp(-((impact['impact_velocity_over_escape'] - 1.1) / 0.75) ** 2)
        * (0.35 + 0.75 * impact['late_veneer_score'])
        * (0.20 + 0.80 * impact['survival_factor'])
    )
    disk_mass_fraction = _clip(disk_mass_fraction, 1e-6, 0.08)
    disk_mass_earth = mass_earth * disk_mass_fraction
    if disk_mass_earth < 1e-5:
        return _empty_rocky_moon_system('none', impact_state=impact)

    water_hint = float((late_veneer or {}).get('water_frac', 5e-4))
    volatile_frac = _clip(0.01 + 3.0 * water_hint, 0.0, 0.18)
    iron_frac = _clip(
        0.03 + 0.08 * (1.0 - impact['impact_angle_deg'] / 80.0) + 0.03 * max(impact['impact_velocity_over_escape'] - 1.0, 0.0),
        0.01,
        0.25,
    )
    silicate_frac = max(0.0, 1.0 - iron_frac - volatile_frac)

    accretion_efficiency = _clip(
        0.22 + 0.38 * impact['late_veneer_score'] + 0.10 * np.exp(-((impact['impact_velocity_over_escape'] - 1.1) / 0.6) ** 2),
        0.15,
        0.72,
    )
    total_moon_mass_earth = disk_mass_earth * accretion_efficiency

    formation_channel = 'giant_impact'
    binary_like = False
    if total_moon_mass_earth / max(mass_earth, 1e-9) > 0.05 and mass_earth < 0.6 and planet.get('semi_major_au', 1.0) > 0.3:
        formation_channel = 'binary_terrestrial'
        binary_like = True
    elif total_moon_mass_earth < 3e-4:
        formation_channel = 'debris_chain'

    major_moons = []
    minor_moons = []

    stable_outer_rp = float(impact['stable_outer_rp'])
    roche_limit_rp = float(impact['roche_limit_rp'])
    planet_radius_re = float((phys or {}).get('R_mean_Re', _planet_radius_re(mass_earth)))
    rotation_period_hr = max(float(planet.get('rotation_period_hr', 24.0)), 1.0)

    major_mass_earth = 0.0
    if formation_channel != 'debris_chain':
        major_mass_earth = total_moon_mass_earth * (0.80 if binary_like else 0.68)
    if major_mass_earth >= 5e-5:
        initial_rp = max(roche_limit_rp * 1.35, 3.2 + 1.6 * rng.random())
        tidal_strength = _clip(
            0.22 * mass_earth ** 0.25 * (24.0 / rotation_period_hr) ** 0.35 * (0.012 / max(major_mass_earth / mass_earth, 5e-5)) ** 0.15,
            0.03,
            2.0,
        )
        final_rp = initial_rp * (1.0 + 40.0 * actual_age_gyr * tidal_strength) ** 0.55
        final_rp = _clip(final_rp, initial_rp, stable_outer_rp * 0.72)
        eccentricity = _clip(rng.lognormal(np.log(0.006), 0.45), 0.0005, 0.06)
        tidal_heating_index = _clip(1200.0 * major_mass_earth * (eccentricity + 0.002) ** 2 / max(final_rp ** 4.8, 1.0), 0.0, 12.0)
        mass_ratio = major_mass_earth / max(mass_earth, 1e-9)
        major_moons.append({
            'index': 0,
            'name': 'M-1',
            'mass_earth': round(float(major_mass_earth), 6),
            'radius_re': round(float(_moon_radius_re(major_mass_earth, iron_frac, silicate_frac, volatile_frac)), 4),
            'semi_major_rp': round(float(final_rp), 4),
            'formation_radius_rp': round(float(initial_rp), 4),
            'orbital_period_days': round(float(_orbital_period_days(mass_earth, final_rp, planet_radius_re)), 4),
            'eccentricity': round(float(eccentricity), 5),
            'retrograde': False,
            'tidally_locked': True,
            'tidal_heating_index': round(float(tidal_heating_index), 4),
            'iron_frac': round(float(iron_frac), 4),
            'silicate_frac': round(float(silicate_frac), 4),
            'volatile_frac': round(float(volatile_frac), 4),
            'moon_to_planet_mass_ratio': round(float(mass_ratio), 6),
            'family': _major_family(mass_ratio, volatile_frac),
        })

    leftover_mass = max(total_moon_mass_earth - sum(moon['mass_earth'] for moon in major_moons), 0.0)
    max_minor = 2 if formation_channel == 'debris_chain' else 3
    if leftover_mass > 1e-6:
        n_minor = int(np.clip(
            rng.poisson(0.35 + 2.2 * min(leftover_mass / max(total_moon_mass_earth, 1e-12), 1.0)),
            0,
            max_minor,
        ))
        if formation_channel == 'debris_chain' and n_minor == 0:
            n_minor = 1
        if n_minor > 0:
            weights = rng.lognormal(0.0, 0.6, n_minor)
            minor_masses = leftover_mass * weights / max(float(weights.sum()), 1e-12)
            base_r = max(roche_limit_rp * 1.25, 2.8)
            radii = np.geomspace(base_r, min(stable_outer_rp * 0.35, base_r * (1.8 ** max(n_minor - 1, 0))), n_minor)
            for idx, mass_minor in enumerate(minor_masses):
                a_rp = _clip(radii[idx], base_r, stable_outer_rp * 0.55)
                eccentricity = _clip(rng.lognormal(np.log(0.01), 0.55), 0.001, 0.12)
                family = 'phobos_like' if idx == 0 else 'deimos_like'
                minor_moons.append({
                    'index': idx,
                    'name': f'm-{idx + 1}',
                    'mass_earth': round(float(mass_minor), 8),
                    'radius_re': round(float(_moon_radius_re(mass_minor, iron_frac * 0.8, silicate_frac * 1.05, volatile_frac * 0.5)), 4),
                    'semi_major_rp': round(float(a_rp), 4),
                    'formation_radius_rp': round(float(a_rp), 4),
                    'orbital_period_days': round(float(_orbital_period_days(mass_earth, a_rp, planet_radius_re)), 4),
                    'eccentricity': round(float(eccentricity), 5),
                    'retrograde': False,
                    'tidally_locked': bool(a_rp < 40.0),
                    'tidal_heating_index': round(float(_clip(400.0 * mass_minor * (eccentricity + 0.002) ** 2 / max(a_rp ** 4.5, 1.0), 0.0, 4.0)), 4),
                    'iron_frac': round(float(_clip(iron_frac * 0.8, 0.01, 0.2)), 4),
                    'silicate_frac': round(float(_clip(silicate_frac, 0.6, 0.97)), 4),
                    'volatile_frac': round(float(_clip(volatile_frac * 0.5, 0.0, 0.08)), 4),
                    'moon_to_planet_mass_ratio': round(float(mass_minor / max(mass_earth, 1e-9)), 8),
                    'family': family,
                })

    all_masses = [moon['mass_earth'] for moon in major_moons + minor_moons]
    total_mass = float(sum(all_masses))
    largest_mass = max(all_masses) if all_masses else 0.0
    return {
        'formation_channel': formation_channel,
        'impact_state': impact,
        'debris_disk': {
            'disk_mass_fraction': round(float(disk_mass_fraction), 6),
            'disk_mass_earth': round(float(disk_mass_earth), 6),
            'silicate_fraction': round(float(silicate_frac), 4),
            'iron_fraction': round(float(iron_frac), 4),
            'volatile_fraction': round(float(volatile_frac), 4),
            'retention_factor': round(float(impact['survival_factor']), 4),
        },
        'summary': {
            'n_major': int(len(major_moons)),
            'n_minor': int(len(minor_moons)),
            'largest_moon_mass_earth': round(float(largest_mass), 6),
            'total_moon_mass_earth': round(float(total_mass), 6),
            'moon_to_planet_mass_ratio': round(float(total_mass / max(mass_earth, 1e-9)), 6),
            'binary_like': bool(binary_like),
        },
        'major_moons': major_moons,
        'minor_moons': minor_moons,
    }


def estimate_rocky_moon_summary(planet, star_mass, all_planets=None, late_veneer=None,
                                phys=None, current_stellar_mass=None, actual_age_gyr=4.5,
                                rng_seed=None):
    system = build_rocky_moon_system(
        planet,
        star_mass=star_mass,
        all_planets=all_planets,
        late_veneer=late_veneer,
        phys=phys,
        current_stellar_mass=current_stellar_mass,
        actual_age_gyr=actual_age_gyr,
        rng_seed=rng_seed,
    )
    summary = system['summary']
    return {
        'moon_count_estimate': int(summary['n_major'] + summary['n_minor']),
        'has_moon_system': bool(summary['n_major'] + summary['n_minor'] > 0),
        'has_large_moon': bool(summary['largest_moon_mass_earth'] >= 0.002),
        'has_binary_terrestrial': bool(summary['binary_like']),
        'largest_moon_mass_earth': summary['largest_moon_mass_earth'],
        'formation_channel': system['formation_channel'],
    }


__all__ = [
    'build_rocky_moon_system',
    'estimate_impact_state',
    'estimate_rocky_moon_summary',
]
