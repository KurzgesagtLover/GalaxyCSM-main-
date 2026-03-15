"""
Galaxy synthesis and H-R utilities built on modular stellar evolution pieces.

Public import paths remain stable for the API and tests, while the underlying
physics now lives in dedicated submodules.
"""

import numpy as np

from .config import DEFAULT_VIEW_T_MAX, coerce_solver_params
from .physics import (
    sample_kroupa_imf,
    sample_powerlaw_imf,
    stellar_lifetime as _stellar_lifetime_model,
)
from .solver import GCESolver
from .stellar_properties import PHASE_KR, SPECTRAL, _ms_lifetime, _ms_state, _spectral
from .stellar_tracks import stellar_evolution


def hr_track(mass, age_max=None, n_points=300, metallicity_z=0.02):
    """Generate detailed H-R evolution track."""
    t_ms = _ms_lifetime(mass, metallicity_z=metallicity_z)
    total = age_max if age_max is not None else t_ms * 1.5 + 1
    total = max(total, t_ms * 1.4)
    total = min(total, DEFAULT_VIEW_T_MAX)

    ages = set()
    pre_end = min(0.002, t_ms * 0.05)
    ages.update(np.linspace(0, pre_end, 15))
    ages.update(np.linspace(pre_end, t_ms * 0.5, 25))
    ages.update(np.linspace(t_ms * 0.5, t_ms * 0.85, 25))
    ages.update(np.linspace(t_ms * 0.85, t_ms * 0.95, 20))
    ages.update(np.linspace(t_ms * 0.95, t_ms * 1.05, 30))
    ages.update(np.linspace(t_ms * 1.05, t_ms * 1.15, 30))
    ages.update(np.linspace(t_ms * 1.15, t_ms * 1.25, 25))
    ages.update(np.linspace(t_ms * 1.25, t_ms * 1.35, 20))
    if total > t_ms * 1.35:
        ages.update(np.linspace(t_ms * 1.35, total, 40))
    ages = sorted(a for a in ages if 0 <= a <= total)
    if len(ages) > n_points:
        keep = np.linspace(0, len(ages) - 1, n_points, dtype=int)
        ages = [ages[i] for i in sorted(set(keep.tolist()))]

    track = []
    for age in ages:
        evo = stellar_evolution(mass, age, metallicity_z=metallicity_z)
        track.append({
            'age': round(age, 6),
            'T_eff': evo['T_eff'],
            'luminosity': evo['luminosity'],
            'radius': evo['radius'],
            'phase': evo['phase'],
            'phase_kr': evo['phase_kr'],
            'color': evo['color'],
            'spectral_class': evo['spectral_class'],
        })
    return track


def _sample_imf(rng, n):
    return sample_kroupa_imf(rng, n, m_lo=0.08, m_hi=100.0)


def _sample_popiii_imf(rng, n):
    """Use a bounded top-heavy power law instead of a uniform 50-300 Msun draw."""
    return sample_powerlaw_imf(rng, n, m_lo=10.0, m_hi=150.0, alpha=1.7)


def _spiral_pos(r, rng, n, n_arms=4, pitch_deg=12.0):
    pitch = np.radians(pitch_deg)
    arm = rng.integers(0, n_arms, n)
    arm_off = 2 * np.pi * arm / n_arms
    theta = (1.0 / np.tan(pitch)) * np.log(np.clip(r, 0.1, None)) + arm_off
    theta += rng.normal(0, 0.35 + 0.02 * r, n)
    r_sc = np.abs(r + rng.normal(0, 0.25, n))
    bulge_frac = np.exp(-r / 1.5)
    iso = rng.uniform(0, 2 * np.pi, n)
    theta = np.where(rng.uniform(0, 1, n) < bulge_frac, iso, theta)
    return r_sc * np.cos(theta), r_sc * np.sin(theta), rng.normal(0, 0.08 + 0.01 * r, n)


from .moons import estimate_moon_summary
from .planets import estimate_spectral_habitable_zone, generate_planets


_GIANT_PHASES = {
    'subgiant', 'RGB', 'HB', 'AGB', 'post-AGB', 'PN',
    'BSG', 'RSG', 'YSG', 'LBV', 'WR', 'hypergiant', 'SN',
}
_DEAD_PHASES = {'WD', 'Black Dwarf', 'NS', 'BH'}


def _overview_phase_bucket(phase):
    if phase in ('disk', 'pre-MS'):
        return 'pre-MS'
    if phase in ('MS', 'blue_dwarf'):
        return 'MS'
    if phase in _GIANT_PHASES:
        return 'giant'
    if phase in _DEAD_PHASES:
        return 'dead'
    return 'MS'


def _overview_alive_until(mass, metallicity_z=0.02):
    """Approximate age when the star leaves luminous stellar phases."""
    t_ms = _ms_lifetime(mass, metallicity_z=metallicity_z)
    if mass < 0.45:
        return t_ms * 1.50
    if mass < 8.0:
        return t_ms * 1.25
    if mass < 25.0:
        return t_ms * 1.02
    if mass < 60.0:
        return t_ms * 0.95
    return t_ms * 0.90


def generate_galaxy(n_stars=25000, params=None, seed=42):
    rng = np.random.default_rng(seed)
    solver_params = coerce_solver_params(params)
    solver = GCESolver(solver_params)
    result = solver.solve()
    t_grid = np.array(result['time'])
    r_grid = np.array(result['radius'])
    sfr = np.array(result['sfr'])
    nt, nr = len(t_grid), len(r_grid)
    dt = np.zeros(nt)
    dt[1:] = np.diff(t_grid)
    dt[0] = dt[1]
    dr = r_grid[1] - r_grid[0] if nr > 1 else 1.0

    weights = np.zeros((nr, nt))
    for ir in range(nr):
        weights[ir] = sfr[ir] * dt * 2 * np.pi * max(r_grid[ir], 0.1) * dr
    weights = np.clip(weights, 0, None)
    total_w = max(weights.sum(), 1e-10)
    probs = (weights / total_w).ravel()

    bins = rng.choice(nr * nt, size=n_stars, p=probs)
    ir_arr, it_arr = bins // nt, bins % nt
    births = t_grid[it_arr]
    metallicity_grid = np.array(result['metallicity'])
    birth_metallicity = metallicity_grid[ir_arr, it_arr]
    current_time = float(t_grid[-1])

    masses = _sample_imf(rng, n_stars)
    popiii_mask = (births <= max(dt[0], 0.02)) | (birth_metallicity < 1e-4)
    if np.any(popiii_mask):
        masses[popiii_mask] = _sample_popiii_imf(rng, int(np.sum(popiii_mask)))

    xs, ys, zs = np.empty(n_stars), np.empty(n_stars), np.empty(n_stars)
    for ir in range(nr):
        mask = ir_arr == ir
        if mask.sum() == 0:
            continue
        px, py, pz = _spiral_pos(r_grid[ir], rng, mask.sum())
        xs[mask], ys[mask], zs[mask] = px, py, pz

    ms_life = np.asarray(_stellar_lifetime_model(masses, np.clip(birth_metallicity, 1e-6, 0.06)), dtype=float)
    deaths = np.full_like(births, 99999.0)
    types, colors, temps, sizes, lums = [], [], [], [], []
    phase_current, phase_bucket, phase_kr_current = [], [], []
    current_mass_arr, spectral_current = [], []
    has_p, has_rocky, has_gas, has_hz = [], [], [], []
    has_moon_system, has_regular_moons = [], []
    has_irregular_moons, has_large_moon = [], []
    has_resonant_moon_chain, moon_count_estimate = [], []
    met_arr = []

    for idx, mass in enumerate(masses):
        ir, it = ir_arr[idx], it_arr[idx]
        metallicity_z = float(birth_metallicity[idx])
        star_age = max(current_time - births[idx], 0.0)
        evo = stellar_evolution(mass, star_age, metallicity_z=metallicity_z)
        sp, _, _ = _spectral(mass)
        types.append(sp)
        colors.append(evo['color'])
        temps.append(round(float(evo['T_eff']), 1))
        lums.append(round(float(evo['luminosity']), 6))
        radius_now = max(float(evo['radius']), 1e-5)
        sizes.append(round(min(0.12 + 0.18 * np.log10(1.0 + radius_now * 10.0), 1.2), 3))
        phase_current.append(evo['phase'])
        phase_kr_current.append(evo['phase_kr'])
        phase_bucket.append(_overview_phase_bucket(evo['phase']))
        current_mass_arr.append(float(evo['current_mass']))
        spectral_current.append(evo['spectral_class'])
        deaths[idx] = births[idx] + _overview_alive_until(mass, metallicity_z=metallicity_z)

        fe_h = result['XH']['Fe'][ir][it]
        met_arr.append(round(float(fe_h), 4))

        p_rng = np.random.default_rng(idx)
        system = generate_planets(mass, metallicity_z, p_rng)
        hp = len(system) > 0
        h_rocky = any(p['type'] in ('rocky', 'hot_rocky', 'mini_neptune') for p in system)
        h_gas = any(p['type'] == 'gas_giant' for p in system)
        moon_stats = {
            'moon_count_estimate': 0,
            'has_moon_system': False,
            'has_regular_moons': False,
            'has_irregular_moons': False,
            'has_large_moon': False,
            'has_resonant_moon_chain': False,
        }
        for planet in system:
            if planet['type'] not in ('gas_giant', 'mini_neptune'):
                continue
            moon_summary = estimate_moon_summary(
                planet,
                star_mass=mass,
                metallicity_z=metallicity_z,
                current_stellar_mass=evo['current_mass'],
                actual_age_gyr=star_age,
                rng_seed=idx * 4099 + planet['index'] * 131 + 17,
            )
            moon_stats['moon_count_estimate'] += moon_summary['moon_count_estimate']
            moon_stats['has_moon_system'] = moon_stats['has_moon_system'] or moon_summary['has_moon_system']
            moon_stats['has_regular_moons'] = moon_stats['has_regular_moons'] or moon_summary['has_regular_moons']
            moon_stats['has_irregular_moons'] = moon_stats['has_irregular_moons'] or moon_summary['has_irregular_moons']
            moon_stats['has_large_moon'] = moon_stats['has_large_moon'] or moon_summary['has_large_moon']
            moon_stats['has_resonant_moon_chain'] = (
                moon_stats['has_resonant_moon_chain'] or moon_summary['has_resonant_moon_chain']
            )
        if phase_current[-1] in _DEAD_PHASES:
            hhz = False
        else:
            hz_now = estimate_spectral_habitable_zone(evo['luminosity'], evo['T_eff'])
            hhz = any(
                hz_now['conservative_inner_au'] <= p['semi_major_au'] <= hz_now['conservative_outer_au']
                for p in system
            )

        has_p.append(1 if hp else 0)
        has_rocky.append(1 if h_rocky else 0)
        has_gas.append(1 if h_gas else 0)
        has_hz.append(1 if hhz else 0)
        has_moon_system.append(1 if moon_stats['has_moon_system'] else 0)
        has_regular_moons.append(1 if moon_stats['has_regular_moons'] else 0)
        has_irregular_moons.append(1 if moon_stats['has_irregular_moons'] else 0)
        has_large_moon.append(1 if moon_stats['has_large_moon'] else 0)
        has_resonant_moon_chain.append(1 if moon_stats['has_resonant_moon_chain'] else 0)
        moon_count_estimate.append(int(moon_stats['moon_count_estimate']))

    return {
        'stars': {
            'x': xs.tolist(),
            'y': ys.tolist(),
            'z': zs.tolist(),
            'mass': masses.tolist(),
            'type': types,
            'color': colors,
            'temp': temps,
            'size': sizes,
            'luminosity': lums,
            'birth': births.tolist(),
            'death': deaths.tolist(),
            'birth_z': birth_metallicity.tolist(),
            'ms_lifetime': ms_life.tolist(),
            'phase_current': phase_current,
            'phase_bucket': phase_bucket,
            'phase_kr_current': phase_kr_current,
            'current_mass': current_mass_arr,
            'spectral_current': spectral_current,
            'r_zone': ir_arr.tolist(),
            't_zone': it_arr.tolist(),
            'met': met_arr,
            'has_planets': has_p,
            'has_rocky': has_rocky,
            'has_gas': has_gas,
            'has_hz': has_hz,
            'has_moon_system': has_moon_system,
            'has_regular_moons': has_regular_moons,
            'has_irregular_moons': has_irregular_moons,
            'has_large_moon': has_large_moon,
            'has_resonant_moon_chain': has_resonant_moon_chain,
            'moon_count_estimate': moon_count_estimate,
        },
        'n_stars': n_stars,
        'time': result['time'],
        'radius': result['radius'],
        't_max': solver_params.get('t_max', 13.8),
        'gce': result,
    }


__all__ = [
    'PHASE_KR',
    'SPECTRAL',
    '_ms_lifetime',
    '_ms_state',
    '_overview_alive_until',
    '_overview_phase_bucket',
    '_sample_popiii_imf',
    '_spectral',
    'generate_galaxy',
    'hr_track',
    'stellar_evolution',
]
