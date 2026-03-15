"""Semi-analytic giant / Neptune-class moon system model.

This module adds regular / irregular satellite systems for giant and
Neptune-class planets while
staying lightweight enough for overview generation.

References:
  - Canup & Ward 2006, Nature 441, 834
  - Sasaki et al. 2010, ApJ 714, 1052
  - Ogihara & Ida 2012, ApJ 753, 60
  - Cilibrasi et al. 2018, MNRAS 480, 4355
  - Heller & Barnes 2013, Astrobiology 13, 18
"""

import numpy as np

AU_M = 1.496e11
M_EARTH = 5.972e24
R_EARTH = 6.371e6
M_SUN = 1.989e30
R_JUP_M = 7.1492e7
R_JUP_RE = R_JUP_M / R_EARTH
M_JUP_EARTH = 317.8
Z_SUN = 0.0134


def _clip(value, lo, hi):
    return float(np.clip(float(value), lo, hi))


def _moon_host_regime(planet):
    return 'neptunian' if planet.get('type') == 'mini_neptune' else 'jovian'


def _planet_radius_rj(mass_earth, host_type='gas_giant', t_eq_k=120.0):
    """Approximate host-planet radius in Jupiter radii."""
    mass = max(float(mass_earth), 15.0)
    if host_type == 'mini_neptune':
        base = 0.35 * (mass / 17.0) ** 0.22
        inflation = 1.0 + 0.04 * np.clip((float(t_eq_k) - 800.0) / 1000.0, 0.0, 1.5)
        return _clip(base * inflation, 0.18, 0.75)

    if mass < 40.0:
        base = 0.62 * (mass / 20.0) ** 0.42
    elif mass < 300.0:
        base = 0.90 + 0.12 * np.tanh((mass - 80.0) / 120.0)
    else:
        base = 1.03 - 0.05 * np.log10(mass / 300.0 + 1.0)
    inflation = 1.0 + 0.10 * np.clip((float(t_eq_k) - 900.0) / 1200.0, 0.0, 1.5)
    return _clip(base * inflation, 0.35, 1.8)


def _equilibrium_temperature(luminosity_lsun, semi_major_au, albedo_bond=0.2):
    a_au = max(float(semi_major_au), 0.01)
    alb = _clip(albedo_bond, 0.0, 0.95)
    return 278.5 * max(float(luminosity_lsun), 1e-8) ** 0.25 * (1.0 - alb) ** 0.25 / np.sqrt(a_au)


def _normalize_composition(rock_frac, ice_frac, volatile_frac):
    parts = np.array([
        max(float(rock_frac), 0.0),
        max(float(ice_frac), 0.0),
        max(float(volatile_frac), 0.0),
    ], dtype=float)
    total = max(float(parts.sum()), 1e-12)
    parts /= total
    return tuple(float(x) for x in parts)


def _moon_radius_re(mass_earth, rock_frac, ice_frac, volatile_frac):
    density = 3300.0 * rock_frac + 1450.0 * ice_frac + 900.0 * volatile_frac
    density = max(float(density), 700.0)
    radius_re = max(float(mass_earth), 1e-8) ** (1.0 / 3.0) * (5515.0 / density) ** (1.0 / 3.0)
    return _clip(radius_re, 0.01, 1.5)


def _regular_family(mass_earth, rock_frac, ice_frac, volatile_frac, tidal_heating_index):
    if volatile_frac > 0.16 and mass_earth > 0.004:
        return 'titan_like'
    if rock_frac > 0.68 and ice_frac < 0.20 and tidal_heating_index > 1.2:
        return 'io_like'
    if 0.18 <= ice_frac <= 0.48 and rock_frac > 0.40:
        return 'europa_like'
    if ice_frac > 0.50 and mass_earth > 0.01 and tidal_heating_index > 0.2:
        return 'ganymede_like'
    if ice_frac > 0.45:
        return 'callisto_like'
    return 'io_like' if rock_frac > 0.7 else 'europa_like'


def _count_resonant_pairs(regular_moons):
    return sum(1 for moon in regular_moons if moon.get('resonance'))


def _summarize_moons(regular_moons, irregular_moons):
    masses = [moon['mass_earth'] for moon in regular_moons + irregular_moons]
    family_mass = {}
    for moon in regular_moons + irregular_moons:
        family_mass[moon['family']] = family_mass.get(moon['family'], 0.0) + moon['mass_earth']
    dominant_family = max(family_mass, key=family_mass.get) if family_mass else 'none'
    return {
        'n_regular': int(len(regular_moons)),
        'n_irregular': int(len(irregular_moons)),
        'total_regular_mass_earth': round(float(sum(moon['mass_earth'] for moon in regular_moons)), 6),
        'largest_moon_mass_earth': round(float(max(masses) if masses else 0.0), 6),
        'resonant_chain_count': int(_count_resonant_pairs(regular_moons)),
        'tidal_heating_candidates': int(sum(moon['tidal_heating_index'] > 1.0 for moon in regular_moons)),
        'dominant_family': dominant_family,
    }


def derive_cpd_properties(planet, star_mass, metallicity_z=0.0134, disk=None,
                          current_stellar_mass=None, actual_age_gyr=4.5):
    """Derive circumplanetary disk properties from giant-planet formation state."""
    planet_mass_earth = float(planet['mass_earth'])
    host_type = planet.get('type', 'gas_giant')
    host_regime = _moon_host_regime(planet)
    semi_major_au = max(float(planet['semi_major_au']), 0.01)
    formation_a_au = max(float(planet.get('formation_semi_major_au', semi_major_au)), 0.02)
    migration_delta_au = abs(float(planet.get('migration_delta_au', 0.0)))
    migration_efficiency = _clip(planet.get('migration_efficiency', 0.25), 0.0, 1.0)
    current_star_mass = max(float(current_stellar_mass or star_mass), 0.08)
    metallicity_rel = _clip(float(metallicity_z) / max(Z_SUN, 1e-8), 0.1, 5.0)
    disk_mass_ratio = float(disk.disk_mass / max(star_mass, 0.08)) if disk is not None else _clip(0.07 * star_mass ** 0.15, 0.01, 0.15)
    dust_to_gas = float(disk.dust_to_gas) if disk is not None else _clip(0.01 * metallicity_rel, 1e-4, 0.05)
    disk_lifetime_myr = float(disk.lifetime_myr) if disk is not None else _clip(3.0 * star_mass ** (-0.5), 0.5, 20.0)
    disk_snow_line_au = float(getattr(disk, 'snow_line_au', planet.get('disk_snow_line_au_at_formation', max(0.7, 2.7 * star_mass ** 0.9))))

    hot_jupiter = bool(planet.get('is_hot_jupiter', False) or semi_major_au < 0.15)
    hot_compact = bool(host_regime == 'neptunian' and semi_major_au < 0.12)
    formation_t_eq = float(planet.get('formation_T_eq_K', _equilibrium_temperature(max(star_mass, 0.08) ** 3.5, formation_a_au)))
    giant_core_mass = float(planet.get('giant_core_mass_earth', np.clip(0.12 * planet_mass_earth ** 0.7, 5.0, 25.0)))
    radius_rj = _planet_radius_rj(planet_mass_earth, host_type=host_type, t_eq_k=planet.get('T_eq', formation_t_eq))
    planet_radius_m = radius_rj * R_JUP_M
    planet_radius_re = radius_rj * R_JUP_RE
    planet_density = 5515.0 * planet_mass_earth / max(planet_radius_re ** 3, 1e-6)

    hill_radius_m = semi_major_au * AU_M * ((planet_mass_earth * M_EARTH) / (3.0 * current_star_mass * M_SUN)) ** (1.0 / 3.0)
    hill_radius_rp = hill_radius_m / max(planet_radius_m, 1.0)
    stable_outer_rp = 0.48 * hill_radius_rp
    if host_regime == 'neptunian':
        cpd_outer_rp = min(75.0, 0.08 * hill_radius_rp)
    else:
        cpd_outer_rp = min(120.0, 0.12 * hill_radius_rp)
    roche_limit_rp = max(2.4, 2.44 * (planet_density / 1800.0) ** (1.0 / 3.0))
    regular_zone_outer_rp = max(0.0, min(cpd_outer_rp, stable_outer_rp * 0.75))

    gas_supply_factor = _clip((disk_mass_ratio / 0.08) ** 0.6, 0.25, 3.0)
    migration_survival = np.exp(-0.9 * migration_delta_au / max(formation_a_au, 0.2))
    migration_survival = 0.35 + 0.65 * migration_survival
    if hot_jupiter:
        migration_survival *= 0.22
    if hot_compact:
        migration_survival *= 0.35
    if host_regime == 'neptunian':
        cpd_mass_fraction = _clip(
            0.0035 * (planet_mass_earth / 17.0) ** 0.20 * gas_supply_factor * migration_survival,
            8e-5,
            0.012,
        )
    else:
        cpd_mass_fraction = _clip(
            0.015 * (planet_mass_earth / M_JUP_EARTH) ** 0.22 * gas_supply_factor * migration_survival,
            8e-4,
            0.05,
        )
    cpd_mass_earth = planet_mass_earth * cpd_mass_fraction
    if host_regime == 'neptunian':
        cpd_dust_to_gas = _clip(
            dust_to_gas * (1.20 + 0.12 * gas_supply_factor) * (1.08 if formation_a_au >= disk_snow_line_au else 0.95),
            0.002,
            0.12,
        )
    else:
        cpd_dust_to_gas = _clip(
            dust_to_gas * (1.10 + 0.18 * gas_supply_factor) * (1.05 if formation_a_au >= disk_snow_line_au else 0.9),
            0.002,
            0.08,
        )

    q_index = _clip(0.55 + 0.12 * migration_efficiency, 0.5, 0.75)
    if host_regime == 'neptunian':
        temperature_100rp_k = _clip(
            28.0
            + 26.0 * (planet_mass_earth / 17.0) ** 0.32
            + 0.12 * formation_t_eq
            + 8.0 * (1.0 - migration_survival)
            + 4.0 * (giant_core_mass / 10.0),
            25.0,
            180.0,
        )
    else:
        temperature_100rp_k = _clip(
            35.0
            + 45.0 * (planet_mass_earth / M_JUP_EARTH) ** 0.40
            + 0.18 * formation_t_eq
            + 12.0 * (1.0 - migration_survival)
            + 5.0 * (giant_core_mass / 10.0),
            35.0,
            220.0,
        )
    snow_line_rp = 100.0 * (temperature_100rp_k / 170.0) ** (1.0 / q_index)
    cpd_dissipation_myr = _clip(disk_lifetime_myr * (0.25 + 0.55 * migration_survival), 0.05, 10.0)
    if host_regime == 'neptunian':
        accessible_solid_mass_earth = max(0.0, min(
            cpd_mass_earth * cpd_dust_to_gas * 0.75,
            planet_mass_earth * 1.8e-4 * (0.8 + 0.35 * metallicity_rel),
        ))
    else:
        accessible_solid_mass_earth = max(0.0, min(
            cpd_mass_earth * cpd_dust_to_gas * 0.65,
            planet_mass_earth * 4.0e-4 * (0.8 + 0.35 * metallicity_rel),
        ))
    accessible_solid_mass_earth *= migration_survival
    tidal_evolution_strength = _clip(
        (planet_mass_earth / M_JUP_EARTH) ** 0.45
        * (10.0 / max(float(planet.get('rotation_period_hr', 10.0)), 1.0)) ** 0.35
        * (1.0 + 0.08 * min(actual_age_gyr, 10.0))
        * (1.0 / max(semi_major_au, 0.08)) ** 0.1,
        0.1,
        8.0,
    )

    return {
        'host_type': host_type,
        'host_regime': host_regime,
        'mass_fraction': round(float(cpd_mass_fraction), 6),
        'mass_earth': round(float(cpd_mass_earth), 6),
        'dust_to_gas': round(float(cpd_dust_to_gas), 5),
        'outer_radius_rp': round(float(max(cpd_outer_rp, 0.0)), 3),
        'outer_radius_km': round(float(max(cpd_outer_rp, 0.0) * planet_radius_m / 1000.0), 1),
        'dissipation_myr': round(float(cpd_dissipation_myr), 4),
        'snow_line_rp': round(float(max(snow_line_rp, 0.0)), 3),
        'temperature_100rp_K': round(float(temperature_100rp_k), 2),
        'migration_efficiency': round(float(migration_efficiency), 4),
        'tidal_evolution_strength': round(float(tidal_evolution_strength), 4),
        'temperature_profile': {
            'reference_radius_rp': 100.0,
            'reference_temperature_K': round(float(temperature_100rp_k), 2),
            'power_law_index': round(float(q_index), 3),
        },
        'hill_radius_rp': round(float(max(hill_radius_rp, 0.0)), 3),
        'stable_outer_rp': round(float(max(stable_outer_rp, 0.0)), 3),
        'regular_zone_outer_rp': round(float(max(regular_zone_outer_rp, 0.0)), 3),
        'roche_limit_rp': round(float(roche_limit_rp), 3),
        'accessible_solid_mass_earth': round(float(accessible_solid_mass_earth), 6),
        'disk_snow_line_au': round(float(disk_snow_line_au), 3),
        'formation_equilibrium_temp_K': round(float(formation_t_eq), 2),
        'giant_core_mass_earth': round(float(giant_core_mass), 3),
        'hot_jupiter_penalty': round(float(0.22 if hot_jupiter else (0.35 if hot_compact else 1.0)), 3),
        'inner_orbit_penalty': round(float(0.22 if hot_jupiter else (0.35 if hot_compact else 1.0)), 3),
    }


def _cpd_temperature(cpd, radius_rp):
    profile = cpd['temperature_profile']
    ref_r = max(float(profile['reference_radius_rp']), 1.0)
    ref_t = float(profile['reference_temperature_K'])
    q_idx = float(profile['power_law_index'])
    return ref_t * (max(float(radius_rp), 1.0) / ref_r) ** (-q_idx)


def generate_regular_moons(planet, cpd, rng):
    """Generate regular moons that form in the circumplanetary disk."""
    host_regime = cpd.get('host_regime', 'jovian')
    inner_edge_rp = max(float(cpd['roche_limit_rp']) * 1.15, 3.0)
    outer_edge_rp = float(cpd['regular_zone_outer_rp'])
    mass_budget = float(cpd['accessible_solid_mass_earth'])
    if outer_edge_rp <= inner_edge_rp * 1.2 or mass_budget < 5e-4:
        return []

    packing_ratio = 1.65 + 0.10 * rng.random()
    max_slots = max(int(np.floor(np.log(outer_edge_rp / inner_edge_rp) / np.log(packing_ratio))) + 1, 0)
    max_regular = 4 if host_regime == 'neptunian' else 7
    desired_count = int(np.clip(
        round(
            (0.7 if host_regime == 'neptunian' else 1.2)
            + (2.2 if host_regime == 'neptunian' else 4.2) * np.tanh(mass_budget / (0.01 if host_regime == 'neptunian' else 0.03))
            + 0.8 * np.tanh((outer_edge_rp - inner_edge_rp) / 25.0)
            + rng.normal(0, 0.6)
        ),
        0,
        max_regular,
    ))
    n_regular = min(max_slots, desired_count)
    if n_regular <= 0:
        return []

    form_hi = max(inner_edge_rp * 1.3, outer_edge_rp * 0.86)
    formation_radii = np.geomspace(inner_edge_rp * 1.2, form_hi, n_regular)
    formation_radii *= rng.lognormal(0.0, 0.06, n_regular)
    formation_radii = np.clip(formation_radii, inner_edge_rp * 1.05, outer_edge_rp * 0.98)
    formation_radii.sort()
    for i in range(1, len(formation_radii)):
        formation_radii[i] = max(formation_radii[i], formation_radii[i - 1] * 1.32)
        formation_radii[i] = min(formation_radii[i], outer_edge_rp * (0.985 - 0.02 * max(0, n_regular - i - 1)))

    weights = rng.lognormal(0.0, 0.7, n_regular)
    if n_regular > 1:
        weights *= np.linspace(1.15, 0.85, n_regular)
    moon_masses = mass_budget * weights / max(float(weights.sum()), 1e-12)

    regular_moons = []
    snow_line_rp = float(cpd['snow_line_rp'])
    migration_efficiency = float(cpd['migration_efficiency'])
    tidal_strength = float(cpd['tidal_evolution_strength'])
    planet_radius_km = float(cpd['outer_radius_km']) / max(float(cpd['outer_radius_rp']), 1e-6)

    final_radii = []
    resonances = [None] * n_regular
    for i, formation_rp in enumerate(formation_radii):
        drift_mass_scale = 0.004 if host_regime == 'neptunian' else 0.01
        drift_strength = migration_efficiency * (moon_masses[i] / drift_mass_scale + 0.2) ** 0.28 * (formation_rp / max(inner_edge_rp + 1.0, 5.0)) ** 0.18
        migration_delta_rp = np.clip(formation_rp * (0.04 + 0.18 * drift_strength), 0.1, formation_rp * 0.55)
        final_rp = max(inner_edge_rp * (1.03 + 0.10 * i), formation_rp - migration_delta_rp)
        if final_radii:
            final_rp = max(final_rp, final_radii[-1] * 1.28)
        final_radii.append(min(final_rp, outer_edge_rp * 0.99))

    for i in range(1, n_regular):
        ratio_now = final_radii[i] / max(final_radii[i - 1], 1e-6)
        capture_prob = (0.12 if host_regime == 'neptunian' else 0.20) + 0.55 * migration_efficiency
        if 1.35 <= ratio_now <= 2.45 and rng.random() < capture_prob:
            target_ratio = 2.0 if ratio_now > 1.7 else 1.5
            target_r = final_radii[i - 1] * target_ratio
            if target_r <= outer_edge_rp * 0.99:
                final_radii[i] = target_r
                resonances[i] = {'type': '2:1' if target_ratio > 1.7 else '3:2', 'with_inner': i - 1}

    for i in range(1, n_regular):
        final_radii[i] = max(final_radii[i], final_radii[i - 1] * 1.22)
        final_radii[i] = min(final_radii[i], outer_edge_rp * (0.99 - 0.01 * max(0, n_regular - i - 1)))

    for i, mass_earth in enumerate(moon_masses):
        formation_rp = float(formation_radii[i])
        final_rp = float(final_radii[i])
        formed_beyond_snow = bool(formation_rp >= snow_line_rp)
        formation_temp = _cpd_temperature(cpd, formation_rp)

        if formed_beyond_snow:
            ice_frac = _clip(0.48 + 0.20 * np.tanh((formation_rp - snow_line_rp) / max(snow_line_rp, 5.0)) + rng.normal(0.0, 0.04), 0.35, 0.85)
            volatile_frac = _clip(0.03 + 0.18 * np.clip((150.0 - formation_temp) / 120.0, 0.0, 1.0) * (mass_earth / 0.01 + 0.4) ** 0.2, 0.0, 0.35)
            rock_frac = max(0.02, 1.0 - ice_frac - volatile_frac)
        else:
            rock_frac = _clip(0.70 + 0.15 * np.clip((snow_line_rp - formation_rp) / max(snow_line_rp, 5.0), 0.0, 1.0) + rng.normal(0.0, 0.04), 0.55, 0.96)
            ice_frac = _clip(0.20 - 0.10 * np.clip((snow_line_rp - formation_rp) / max(snow_line_rp, 5.0), 0.0, 1.0) + rng.normal(0.0, 0.03), 0.02, 0.35)
            volatile_frac = max(0.0, 1.0 - rock_frac - ice_frac)
        rock_frac, ice_frac, volatile_frac = _normalize_composition(rock_frac, ice_frac, volatile_frac)

        is_resonant = resonances[i] is not None
        eccentricity = _clip(rng.lognormal(np.log(0.003 if is_resonant else 0.0015), 0.55), 0.0002, 0.08 if is_resonant else 0.03)
        inclination_deg = _clip(rng.lognormal(np.log(0.4), 0.6), 0.02, 3.0)
        resonance_boost = 1.8 if is_resonant else 1.0
        tidal_heating_index = _clip(
            1500.0 * tidal_strength * mass_earth * (eccentricity + 0.001) ** 2 * resonance_boost / max(final_rp ** 5, 1.0),
            0.0,
            25.0,
        )
        family = _regular_family(mass_earth, rock_frac, ice_frac, volatile_frac, tidal_heating_index)
        radius_re = _moon_radius_re(mass_earth, rock_frac, ice_frac, volatile_frac)

        regular_moons.append({
            'index': i,
            'name': f'R-{i + 1}',
            'mass_earth': round(float(mass_earth), 6),
            'radius_re': round(float(radius_re), 4),
            'semi_major_rp': round(float(final_rp), 4),
            'semi_major_km': round(float(final_rp * planet_radius_km), 1),
            'eccentricity': round(float(eccentricity), 5),
            'inclination_deg': round(float(inclination_deg), 3),
            'retrograde': False,
            'formation_rp': round(float(formation_rp), 4),
            'formation_temp_K': round(float(formation_temp), 2),
            'formed_beyond_cpd_snowline': formed_beyond_snow,
            'rock_frac': round(float(rock_frac), 4),
            'ice_frac': round(float(ice_frac), 4),
            'volatile_frac': round(float(volatile_frac), 4),
            'migration_delta_rp': round(float(max(formation_rp - final_rp, 0.0)), 4),
            'resonance': resonances[i],
            'tidal_heating_index': round(float(tidal_heating_index), 4),
            'family': family,
        })

    return regular_moons


def generate_irregular_moons(planet, cpd, rng):
    """Generate captured irregular moons."""
    host_regime = cpd.get('host_regime', 'jovian')
    hill_radius_rp = float(cpd['hill_radius_rp'])
    stable_outer_rp = float(cpd['stable_outer_rp'])
    if stable_outer_rp < 25.0:
        return []

    semi_major_au = max(float(planet['semi_major_au']), 0.01)
    hot_penalty = float(cpd['hot_jupiter_penalty'])
    mean_count = (
        (0.9 if host_regime == 'neptunian' else 1.2)
        * (hill_radius_rp / 120.0) ** 0.55
        * (semi_major_au / 1.0) ** 0.20
        * hot_penalty
    )
    mean_count = _clip(mean_count, 0.0, 6.0)
    n_irregular = int(np.clip(rng.poisson(mean_count), 0, 10))
    triton_capture = bool(
        host_regime == 'neptunian'
        and semi_major_au >= max(4.0, float(cpd['disk_snow_line_au']) * 1.1)
        and hot_penalty > 0.3
        and rng.random() < 0.45
    )
    if n_irregular <= 0 and not triton_capture:
        return []

    disk_snow_line_au = float(cpd['disk_snow_line_au'])
    irregular_moons = []
    if triton_capture:
        irregular_moons.append({
            'index': 0,
            'name': 'I-1',
            'mass_earth': round(float(_clip(10 ** rng.uniform(-3.8, -2.2), 1.5e-4, 6e-3)), 8),
            'semi_major_rhill': round(float(_clip(10 ** rng.uniform(np.log10(0.08), np.log10(0.22)), 0.08, 0.22)), 5),
            'eccentricity': round(float(_clip(rng.lognormal(np.log(0.08), 0.45), 0.01, 0.35)), 5),
            'inclination_deg': round(float(_clip(rng.uniform(145.0, 179.0), 145.0, 179.0)), 3),
            'retrograde': True,
            'capture_channel': 'binary_exchange',
            'stability_score': round(float(_clip(rng.uniform(0.45, 0.95), 0.45, 0.95)), 4),
            'family': 'irregular_icy',
        })
    for i in range(n_irregular):
        retrograde = bool(rng.random() < 0.65)
        ecc = _clip(rng.beta(1.2, 1.8), 0.08, 0.85)
        inc = _clip(rng.uniform(25.0, 175.0), 25.0, 175.0)
        if retrograde and inc < 90.0:
            inc = 180.0 - inc
        outer_limit = 0.70 if retrograde else 0.45
        semi_major_rhill = _clip(10 ** rng.uniform(np.log10(0.06), np.log10(outer_limit)), 0.06, outer_limit)
        mass_earth = _clip(10 ** rng.uniform(-7.2, -2.4), 5e-8, 5e-3)
        stability_score = _clip((outer_limit / max(semi_major_rhill * (1.0 + ecc), 1e-4)), 0.05, 1.0)
        icy = bool(planet.get('formation_semi_major_au', semi_major_au) >= disk_snow_line_au)
        family = 'irregular_icy' if icy else 'irregular_rocky'
        capture_channel = rng.choice(['gas_drag_capture', 'pull_down_capture', 'binary_exchange'])
        irregular_moons.append({
            'index': len(irregular_moons),
            'name': f'I-{len(irregular_moons) + 1}',
            'mass_earth': round(float(mass_earth), 8),
            'semi_major_rhill': round(float(semi_major_rhill), 5),
            'eccentricity': round(float(ecc), 5),
            'inclination_deg': round(float(inc), 3),
            'retrograde': retrograde,
            'capture_channel': capture_channel,
            'stability_score': round(float(stability_score), 4),
            'family': family,
        })

    irregular_moons.sort(key=lambda moon: moon['semi_major_rhill'])
    for i, moon in enumerate(irregular_moons):
        moon['index'] = i
        moon['name'] = f'I-{i + 1}'
    return irregular_moons


def build_moon_system(planet, star_mass, metallicity_z=0.0134, disk=None,
                      current_stellar_mass=None, actual_age_gyr=4.5, rng_seed=None):
    """Build the full moon system for a giant or Neptune-class planet."""
    rng = np.random.default_rng(rng_seed)
    cpd = derive_cpd_properties(
        planet,
        star_mass=star_mass,
        metallicity_z=metallicity_z,
        disk=disk,
        current_stellar_mass=current_stellar_mass,
        actual_age_gyr=actual_age_gyr,
    )
    regular_moons = generate_regular_moons(planet, cpd, rng)
    irregular_moons = generate_irregular_moons(planet, cpd, rng)
    summary = _summarize_moons(regular_moons, irregular_moons)
    return {
        'cpd': cpd,
        'summary': summary,
        'regular_moons': regular_moons,
        'irregular_moons': irregular_moons,
    }


def estimate_moon_summary(planet, star_mass, metallicity_z=0.0134, disk=None,
                          current_stellar_mass=None, actual_age_gyr=4.5, rng_seed=None):
    """Return a lightweight overview summary for one moon-hosting giant."""
    system = build_moon_system(
        planet,
        star_mass=star_mass,
        metallicity_z=metallicity_z,
        disk=disk,
        current_stellar_mass=current_stellar_mass,
        actual_age_gyr=actual_age_gyr,
        rng_seed=rng_seed,
    )
    summary = system['summary']
    large_moon_threshold = 0.0025 if planet.get('type') == 'mini_neptune' else 0.008
    return {
        'moon_count_estimate': int(summary['n_regular'] + summary['n_irregular']),
        'has_moon_system': bool(summary['n_regular'] + summary['n_irregular'] > 0),
        'has_regular_moons': bool(summary['n_regular'] > 0),
        'has_irregular_moons': bool(summary['n_irregular'] > 0),
        'has_large_moon': bool(summary['largest_moon_mass_earth'] >= large_moon_threshold),
        'has_resonant_moon_chain': bool(summary['resonant_chain_count'] > 0),
        'largest_moon_mass_earth': summary['largest_moon_mass_earth'],
        'dominant_family': summary['dominant_family'],
    }


__all__ = [
    'build_moon_system',
    'derive_cpd_properties',
    'estimate_moon_summary',
    'generate_irregular_moons',
    'generate_regular_moons',
]
