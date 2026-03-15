"""Planetary system architecture generation helpers."""

import numpy as np

INNER_EDGE_AU_MIN = 0.03


def _infer_snow_line_au(star_mass):
    """Approximate snow line when a full disk model is unavailable."""
    l_star = star_mass ** 3.5 if star_mass < 2.0 else 1.5 * star_mass ** 3.5
    return float(np.clip(2.7 * l_star ** 0.5, 0.5, 50.0))


def _disk_inner_edge_au(star_mass):
    """Magnetospheric cavity / inner migration trap scale."""
    return float(np.clip(0.035 * max(star_mass, 0.1) ** 0.4, INNER_EDGE_AU_MIN, 0.12))


def _apply_disk_migration(a_form, mass_earth, ptype, star_mass, rng, disk=None):
    """Apply semi-analytic Type I/II migration and return final orbit."""
    a_form = float(max(a_form, _disk_inner_edge_au(star_mass) * 1.05))
    inner_edge = _disk_inner_edge_au(star_mass)

    if disk is None:
        disk_mass_ratio = np.clip(0.08 * max(star_mass, 0.2) ** 0.2, 0.02, 0.18)
        disk_lifetime_myr = np.clip(2.8 * max(star_mass, 0.2) ** -0.4, 0.8, 8.0)
        snow_line_au = _infer_snow_line_au(star_mass)
    else:
        disk_mass_ratio = np.clip(disk.disk_mass / max(star_mass, 0.1), 1e-3, 0.5)
        disk_lifetime_myr = float(np.clip(disk.lifetime_myr, 0.3, 20.0))
        snow_line_au = float(disk.snow_line_au)

    aspect_ratio = np.clip(0.033 * max(a_form, 0.1) ** 0.25 * max(star_mass, 0.1) ** -0.15, 0.02, 0.12)
    alpha_visc = 3e-3
    q_planet = mass_earth * 3.003e-6 / max(star_mass, 0.1)
    q_gap = max(3.0 * aspect_ratio**3, 40.0 * alpha_visc * aspect_ratio**2)
    gap_opened = q_planet >= q_gap

    torque_scale = (disk_mass_ratio / 0.04) ** 0.7 * (disk_lifetime_myr / 3.0) ** 0.5
    torque_scale = np.clip(torque_scale * rng.lognormal(0.0, 0.2), 0.12, 6.0)
    snow_trap = snow_line_au * rng.uniform(0.8, 1.05)
    hot_j_prob = np.clip(
        0.01 + 2.0 * max(disk_mass_ratio - 0.08, 0.0) + 0.06 * max(disk_lifetime_myr - 3.0, 0.0),
        0.0, 0.35
    )
    hot_jupiter_channel = bool(
        ptype == 'gas_giant'
        and a_form > 1.2 * snow_line_au
        and (
            (disk_mass_ratio > 0.12 and disk_lifetime_myr > 5.0 and mass_earth > 150.0)
            or rng.random() < hot_j_prob
        )
    )

    if gap_opened or ptype == 'gas_giant':
        migration_mode = 'type_ii'
        tau_myr = 0.8 * (a_form / 5.0) * (0.05 / max(disk_mass_ratio, 1e-3)) ** 0.6
        tau_myr *= (0.04 / aspect_ratio) ** 2 * (300.0 / max(mass_earth, 30.0)) ** 0.15
        if hot_jupiter_channel:
            stopping_radius = inner_edge * rng.uniform(1.2, 2.0)
        else:
            stopping_radius = max(snow_trap, inner_edge * 2.5)
        formation_fraction = rng.uniform(0.2, 0.6)
    elif ptype == 'mini_neptune':
        migration_mode = 'type_i'
        tau_myr = 3.5 * (5.0 / max(mass_earth, 0.5)) * max(a_form, 0.2) ** 1.1
        tau_myr *= (0.04 / max(disk_mass_ratio, 0.005)) ** 0.55 * (aspect_ratio / 0.035) ** 2
        stopping_radius = max(snow_trap, inner_edge * 2.2)
        formation_fraction = rng.uniform(0.5, 0.9)
    else:
        migration_mode = 'type_i'
        tau_myr = 8.0 * (1.0 / max(mass_earth, 0.1)) * max(a_form, 0.1) ** 1.15
        tau_myr *= (0.04 / max(disk_mass_ratio, 0.005)) ** 0.5 * (aspect_ratio / 0.035) ** 2
        stopping_radius = max(inner_edge * rng.uniform(1.8, 3.0), 0.05)
        if 0.5 * snow_line_au <= a_form <= 1.3 * snow_line_au and mass_earth > 0.5:
            stopping_radius = max(stopping_radius, snow_trap)
            tau_myr *= 1.8
        formation_fraction = rng.uniform(0.7, 0.98)

    available_migration_time = disk_lifetime_myr * max(1.0 - formation_fraction, 0.02)
    tau_myr = max(tau_myr / max(torque_scale, 1e-3), 0.08)
    migration_time_myr = available_migration_time
    decay = np.exp(-migration_time_myr / tau_myr)
    a_final = stopping_radius + (a_form - stopping_radius) * decay
    a_final = float(np.clip(a_final, inner_edge, a_form))

    return {
        'a_final_au': a_final,
        'formation_semi_major_au': round(a_form, 4),
        'migration_mode': migration_mode,
        'migration_delta_au': round(a_form - a_final, 4),
        'migration_timescale_myr': round(tau_myr, 4),
        'migration_efficiency': round(float(1.0 - decay), 4),
        'migrated_inward': bool(a_final < a_form - 0.02),
        'gap_opened': bool(gap_opened),
    }


def generate_planets(star_mass, metallicity_z, rng=None, disk=None):
    """Generate planetary system architecture.

    If a ProtoplanetaryDisk object is provided, planets are drawn from
    the disk solid mass budget.  Otherwise a lightweight probabilistic
    fallback is used (for the fast galaxy-overview path).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Fast fallback (no disk object) for galaxy overview ---
    if disk is None:
        return _generate_planets_fast(star_mass, metallicity_z, rng)

    # --- Disk-based generation ---
    return _generate_planets_from_disk(disk, star_mass, rng)


def _generate_planets_fast(star_mass, metallicity_z, rng):
    """Lightweight probabilistic planet generation (galaxy overview)."""
    f_planet = np.clip(0.3 + 8.0 * metallicity_z, 0.05, 0.95)
    n_max = int(np.clip(2 + 6 * metallicity_z / 0.014, 1, 8))
    n_planets = min(rng.binomial(n_max, f_planet), 8)
    l_star = star_mass ** 3.5
    hz_in = 0.95 * np.sqrt(l_star)
    hz_out = 1.37 * np.sqrt(l_star)
    snow_line_au = _infer_snow_line_au(star_mass)
    giant_prob = np.clip(0.015 + 0.18 * metallicity_z / 0.014, 0.005, 0.22)
    mini_neptune_prob = np.clip(0.15 + 0.35 * metallicity_z / 0.014, 0.08, 0.65)
    planets = []
    for i in range(n_planets):
        draw = rng.random()
        if draw < giant_prob / (1.0 + 0.35 * i):
            ptype = 'gas_giant'
            mass = np.clip(rng.lognormal(np.log(140.0), 0.7), 30.0, 3000.0)
            a_form = snow_line_au * (1.2 + rng.exponential(0.8))
            giant_core_mass = np.clip(0.12 * mass ** 0.72, 5.0, 25.0)
        elif draw < giant_prob + mini_neptune_prob / (1.0 + 0.15 * i):
            ptype = 'mini_neptune'
            mass = np.clip(rng.lognormal(np.log(6.0), 0.45), 2.5, 20.0)
            a_form = snow_line_au * rng.lognormal(-0.1, 0.35)
            giant_core_mass = np.clip(0.80 * mass, 2.0, 16.0)
        else:
            ptype = 'rocky'
            log_mass = rng.normal(-0.2, 0.55)
            mass = np.clip(10 ** log_mass, 0.05, 5.0)
            a_form = min(0.25 * (1.6 ** i) * rng.lognormal(0, 0.2), snow_line_au * 0.9)
            giant_core_mass = None

        mig = _apply_disk_migration(a_form, mass, ptype, star_mass, rng)
        a = mig['a_final_au']
        if ptype == 'rocky' and a < 0.1:
            ptype = 'hot_rocky'

        ecc = rng.beta(0.867, 3.03)
        ecc = np.clip(ecc * (0.3 + 0.7 * a / max(a + 0.1, 0.2)), 0, 0.6)
        rot_period = np.clip(24.0 * mass ** -0.3 * rng.lognormal(0, 0.3), 2.0, 10000.0)
        axial_tilt = np.clip(rng.rayleigh(23.0), 0, 170)
        planets.append({
            'index': i, 'mass_earth': round(mass, 3),
            'semi_major_au': round(a, 3), 'type': ptype,
            'in_hz': bool(hz_in <= a <= hz_out),
            'eccentricity': round(float(ecc), 4),
            'rotation_period_hr': round(float(rot_period), 1),
            'axial_tilt_deg': round(float(axial_tilt), 1),
            'formation_semi_major_au': mig['formation_semi_major_au'],
            'migration_mode': mig['migration_mode'],
            'migration_delta_au': mig['migration_delta_au'],
            'migration_timescale_myr': mig['migration_timescale_myr'],
            'migration_efficiency': mig['migration_efficiency'],
            'migrated_inward': mig['migrated_inward'],
            'gap_opened': mig['gap_opened'],
            'is_hot_jupiter': bool(ptype == 'gas_giant' and a < 0.15),
            'formation_epoch_myr': round(float(np.clip(rng.lognormal(np.log(1.5), 0.35), 0.2, 8.0)), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
            'giant_core_mass_earth': round(float(giant_core_mass), 3) if ptype in ('gas_giant', 'mini_neptune') and giant_core_mass is not None else None,
            'disk_snow_line_au_at_formation': round(float(snow_line_au), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
            'disk_dust_to_gas': round(float(np.clip(0.01 * metallicity_z / max(0.0134, 1e-8), 1e-4, 0.05)), 6) if ptype in ('gas_giant', 'mini_neptune') else None,
            'disk_lifetime_myr': round(float(np.clip(3.0 * star_mass ** (-0.5), 0.5, 20.0)), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
        })
    return planets


def _generate_planets_from_disk(disk, star_mass, rng):
    """Disk-budget-based planet generation.

    Planets are grown from the disk solid mass budget:
      1. Giant planet core(s) beyond snow line accrete gas → gas giants
      2. Remaining solid mass inside snow line → rocky / mini-Neptune planets
      3. Orbital spacings follow mutual-Hill-radii packing.

    References:
        Ida & Lin (2004), Mordasini+2012 (population synthesis),
        Kipping 2013 (eccentricity).
    """
    l_star = star_mass ** 3.5
    hz_in = 0.95 * np.sqrt(l_star)
    hz_out = 1.37 * np.sqrt(l_star)

    total_solid = disk.total_solid_mass_earth  # M⊕
    remaining_solid = total_solid
    planets = []
    idx = 0

    # --- Phase 1: Giant planet(s) beyond snow line ---
    # Core-accretion probability scales with solid mass beyond snow line
    solid_beyond_snow = disk._integrate_solid_mass(
        disk.snow_line_au, disk.r_disk_au) / 5.972e27
    # Critical core mass ~ 10 M⊕ for runaway gas accretion (Pollack+96)
    n_giant_max = int(np.clip(solid_beyond_snow / 10.0, 0, 3))
    p_giant = np.clip(0.1 * solid_beyond_snow / 10.0, 0, 0.7)

    a_giant_start = disk.snow_line_au * (1.5 + rng.exponential(0.5))
    for _ in range(n_giant_max):
        if rng.random() > p_giant or remaining_solid < 5.0:
            break
        # Core mass: 5–20 M⊕
        core_mass = np.clip(rng.lognormal(np.log(10), 0.4), 5, 20)
        if core_mass > remaining_solid * 0.5:
            break
        # Gas accretion: core → gas giant (100-3000 M⊕)
        gas_mult = rng.lognormal(np.log(30), 0.6)
        total_mass = np.clip(core_mass * gas_mult, 15, 3000)
        ptype = 'gas_giant' if total_mass > 10 else 'mini_neptune'
        a_form = a_giant_start * rng.lognormal(0, 0.15)
        a_form = np.clip(a_form, disk.snow_line_au * 1.0, disk.r_disk_au * 0.8)
        mig = _apply_disk_migration(a_form, total_mass, ptype, star_mass, rng, disk=disk)
        a = mig['a_final_au']
        ecc = rng.beta(0.867, 3.03)
        ecc = np.clip(ecc * (0.3 + 0.7 * a / max(a + 0.1, 0.2)), 0, 0.6)
        rot_period = np.clip(10 * total_mass**-0.15 * rng.lognormal(0, 0.3), 2, 300)
        axial_tilt = np.clip(rng.rayleigh(15.0), 0, 170)
        planets.append({
            'index': idx, 'mass_earth': round(float(total_mass), 3),
            'semi_major_au': round(float(a), 3), 'type': ptype,
            'in_hz': bool(hz_in <= a <= hz_out),
            'eccentricity': round(float(ecc), 4),
            'rotation_period_hr': round(float(rot_period), 1),
            'axial_tilt_deg': round(float(axial_tilt), 1),
            'formation_semi_major_au': mig['formation_semi_major_au'],
            'migration_mode': mig['migration_mode'],
            'migration_delta_au': mig['migration_delta_au'],
            'migration_timescale_myr': mig['migration_timescale_myr'],
            'migration_efficiency': mig['migration_efficiency'],
            'migrated_inward': mig['migrated_inward'],
            'gap_opened': mig['gap_opened'],
            'is_hot_jupiter': bool(ptype == 'gas_giant' and a < 0.15),
            'formation_epoch_myr': round(float(np.clip(rng.uniform(0.15, disk.lifetime_myr * 0.8), 0.1, disk.lifetime_myr)), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
            'giant_core_mass_earth': round(float(core_mass), 3) if ptype in ('gas_giant', 'mini_neptune') else None,
            'disk_snow_line_au_at_formation': round(float(disk.snow_line_au), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
            'disk_dust_to_gas': round(float(disk.dust_to_gas), 6) if ptype in ('gas_giant', 'mini_neptune') else None,
            'disk_lifetime_myr': round(float(disk.lifetime_myr), 4) if ptype in ('gas_giant', 'mini_neptune') else None,
        })
        remaining_solid -= core_mass
        a_giant_start = a_form * (2.0 + rng.exponential(0.5))
        idx += 1

    # --- Phase 2: Ice giants / mini-Neptunes near snow line ---
    solid_near_snow = disk._integrate_solid_mass(
        disk.snow_line_au * 0.7, disk.snow_line_au * 2.0) / 5.972e27
    n_ice_max = int(np.clip(solid_near_snow / 5.0, 0, 2))
    a_ice = disk.snow_line_au * rng.lognormal(0, 0.2)
    for _ in range(n_ice_max):
        if remaining_solid < 2.0:
            break
        mass = np.clip(rng.lognormal(np.log(5), 0.5), 2.5, 20)
        if mass > remaining_solid * 0.4:
            mass = remaining_solid * 0.3
        if mass < 2.0:
            break
        a_form = np.clip(a_ice, 0.5, disk.r_disk_au * 0.7)
        mig = _apply_disk_migration(a_form, mass, 'mini_neptune', star_mass, rng, disk=disk)
        a_final = mig['a_final_au']
        ecc = rng.beta(0.867, 3.03)
        ecc = np.clip(ecc * (0.3 + 0.7 * a_final / max(a_final + 0.1, 0.2)), 0, 0.4)
        rot_period = np.clip(16 * mass**-0.2 * rng.lognormal(0, 0.3), 3, 500)
        axial_tilt = np.clip(rng.rayleigh(25.0), 0, 170)
        planets.append({
            'index': idx, 'mass_earth': round(float(mass), 3),
            'semi_major_au': round(float(a_final), 3), 'type': 'mini_neptune',
            'in_hz': bool(hz_in <= a_final <= hz_out),
            'eccentricity': round(float(ecc), 4),
            'rotation_period_hr': round(float(rot_period), 1),
            'axial_tilt_deg': round(float(axial_tilt), 1),
            'formation_semi_major_au': mig['formation_semi_major_au'],
            'migration_mode': mig['migration_mode'],
            'migration_delta_au': mig['migration_delta_au'],
            'migration_timescale_myr': mig['migration_timescale_myr'],
            'migration_efficiency': mig['migration_efficiency'],
            'migrated_inward': mig['migrated_inward'],
            'gap_opened': mig['gap_opened'],
            'is_hot_jupiter': False,
            'formation_epoch_myr': round(float(np.clip(rng.uniform(0.2, disk.lifetime_myr * 0.9), 0.1, disk.lifetime_myr)), 4),
            'giant_core_mass_earth': round(float(np.clip(0.80 * mass, 2.0, 16.0)), 3),
            'disk_snow_line_au_at_formation': round(float(disk.snow_line_au), 4),
            'disk_dust_to_gas': round(float(disk.dust_to_gas), 6),
            'disk_lifetime_myr': round(float(disk.lifetime_myr), 4),
        })
        remaining_solid -= mass * 0.5  # only core uses solids
        a_ice = a_form * (2.0 + rng.exponential(0.3))
        idx += 1

    # --- Phase 3: Rocky planets inside snow line ---
    solid_inner = disk._integrate_solid_mass(0.1, disk.snow_line_au) / 5.972e27
    solid_inner = min(solid_inner, remaining_solid)
    n_rocky_max = int(np.clip(solid_inner / 0.3, 0, 5))
    a_rocky = 0.3 + rng.exponential(0.2)
    for _ in range(n_rocky_max):
        if remaining_solid < 0.05:
            break
        mass = np.clip(rng.lognormal(np.log(0.8), 0.6), 0.05, 5.0)
        if mass > remaining_solid * 0.5:
            mass = remaining_solid * 0.4
        if mass < 0.05:
            break
        a_form = np.clip(a_rocky, 0.05, disk.snow_line_au * 0.95)
        mig = _apply_disk_migration(a_form, mass, 'rocky', star_mass, rng, disk=disk)
        a_final = mig['a_final_au']
        ptype = 'hot_rocky' if a_final < 0.1 else 'rocky'
        ecc = rng.beta(0.867, 3.03)
        ecc = np.clip(ecc * (0.3 + 0.7 * a_final / max(a_final + 0.1, 0.2)), 0, 0.4)
        rot_period = np.clip(24 * mass**-0.3 * rng.lognormal(0, 0.3), 2, 10000)
        axial_tilt = np.clip(rng.rayleigh(23.0), 0, 170)
        planets.append({
            'index': idx, 'mass_earth': round(float(mass), 3),
            'semi_major_au': round(float(a_final), 3), 'type': ptype,
            'in_hz': bool(hz_in <= a_final <= hz_out),
            'eccentricity': round(float(ecc), 4),
            'rotation_period_hr': round(float(rot_period), 1),
            'axial_tilt_deg': round(float(axial_tilt), 1),
            'formation_semi_major_au': mig['formation_semi_major_au'],
            'migration_mode': mig['migration_mode'],
            'migration_delta_au': mig['migration_delta_au'],
            'migration_timescale_myr': mig['migration_timescale_myr'],
            'migration_efficiency': mig['migration_efficiency'],
            'migrated_inward': mig['migrated_inward'],
            'gap_opened': mig['gap_opened'],
            'is_hot_jupiter': False,
        })
        remaining_solid -= mass
        a_rocky = a_form * (1.4 + rng.exponential(0.3))
        idx += 1

    # Sort planets by semi-major axis and re-index
    planets.sort(key=lambda p: p['semi_major_au'])
    for i, p in enumerate(planets):
        p['index'] = i

    return planets
