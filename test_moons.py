# -*- coding: utf-8 -*-
"""Physical consistency checks for giant and Neptune-class moon systems."""

import sys, io


def _configure_stdout():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


import numpy as np

from gce.disk import ProtoplanetaryDisk
from gce.moons import build_moon_system

SEP = "=" * 60


def _giant_planet(
    mass_earth,
    semi_major_au,
    formation_semi_major_au=None,
    migration_efficiency=0.35,
    rotation_period_hr=10.0,
    is_hot_jupiter=False,
    giant_core_mass_earth=12.0,
):
    formation_a = formation_semi_major_au if formation_semi_major_au is not None else semi_major_au
    migration_delta = max(float(formation_a) - float(semi_major_au), 0.0)
    return {
        'index': 0,
        'type': 'gas_giant',
        'mass_earth': float(mass_earth),
        'semi_major_au': float(semi_major_au),
        'formation_semi_major_au': float(formation_a),
        'migration_delta_au': float(migration_delta),
        'migration_efficiency': float(migration_efficiency),
        'migration_mode': 'type_ii',
        'rotation_period_hr': float(rotation_period_hr),
        'T_eq': 120.0 if semi_major_au > 0.3 else 900.0,
        'formation_T_eq_K': 130.0,
        'giant_core_mass_earth': float(giant_core_mass_earth),
        'is_hot_jupiter': bool(is_hot_jupiter),
    }


def _neptune_planet(
    mass_earth=17.0,
    semi_major_au=24.0,
    formation_semi_major_au=None,
    migration_efficiency=0.22,
    rotation_period_hr=16.0,
    giant_core_mass_earth=10.0,
):
    formation_a = formation_semi_major_au if formation_semi_major_au is not None else semi_major_au
    migration_delta = max(float(formation_a) - float(semi_major_au), 0.0)
    return {
        'index': 1,
        'type': 'mini_neptune',
        'mass_earth': float(mass_earth),
        'semi_major_au': float(semi_major_au),
        'formation_semi_major_au': float(formation_a),
        'migration_delta_au': float(migration_delta),
        'migration_efficiency': float(migration_efficiency),
        'migration_mode': 'type_i',
        'rotation_period_hr': float(rotation_period_hr),
        'T_eq': 85.0 if semi_major_au > 1.0 else 700.0,
        'formation_T_eq_K': 110.0,
        'giant_core_mass_earth': float(giant_core_mass_earth),
        'is_hot_jupiter': False,
    }


def test_more_massive_giants_build_more_massive_regular_systems():
    print(SEP)
    print("TEST 1: More Massive Giants Build More Massive Moon Systems")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, np.random.default_rng(12))
    light = _giant_planet(95.0, 5.2, giant_core_mass_earth=8.0)
    heavy = _giant_planet(650.0, 5.2, giant_core_mass_earth=18.0)

    moons_light = build_moon_system(light, 1.0, 0.0134, disk=disk, rng_seed=101)
    moons_heavy = build_moon_system(heavy, 1.0, 0.0134, disk=disk, rng_seed=101)

    print(f"  light CPD={moons_light['cpd']['mass_earth']:.4f} Me  regular={moons_light['summary']['total_regular_mass_earth']:.4f} Me")
    print(f"  heavy CPD={moons_heavy['cpd']['mass_earth']:.4f} Me  regular={moons_heavy['summary']['total_regular_mass_earth']:.4f} Me")

    assert moons_heavy['cpd']['mass_earth'] > moons_light['cpd']['mass_earth']
    assert moons_heavy['summary']['total_regular_mass_earth'] >= moons_light['summary']['total_regular_mass_earth']
    assert moons_heavy['summary']['largest_moon_mass_earth'] >= moons_light['summary']['largest_moon_mass_earth']
    print("  [OK] More massive giants support more massive regular satellite systems")


def test_hot_jupiters_retain_fewer_regular_moons():
    print("\n" + SEP)
    print("TEST 2: Hot Jupiters Retain Fewer Regular Moons")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, np.random.default_rng(21))
    cold = _giant_planet(318.0, 5.2, formation_semi_major_au=5.2, migration_efficiency=0.20)
    hot = _giant_planet(318.0, 0.05, formation_semi_major_au=5.2, migration_efficiency=0.85, is_hot_jupiter=True)

    moons_cold = build_moon_system(cold, 1.0, 0.0134, disk=disk, rng_seed=22)
    moons_hot = build_moon_system(hot, 1.0, 0.0134, disk=disk, rng_seed=22)

    print(f"  cold regular={moons_cold['summary']['n_regular']}  mass={moons_cold['summary']['total_regular_mass_earth']:.4f} Me")
    print(f"  hot  regular={moons_hot['summary']['n_regular']}  mass={moons_hot['summary']['total_regular_mass_earth']:.4f} Me")
    print(f"  cold outer={moons_cold['cpd']['regular_zone_outer_rp']:.2f} Rp  hot outer={moons_hot['cpd']['regular_zone_outer_rp']:.2f} Rp")

    assert moons_hot['cpd']['regular_zone_outer_rp'] < moons_cold['cpd']['regular_zone_outer_rp']
    assert moons_hot['summary']['n_regular'] <= moons_cold['summary']['n_regular']
    assert moons_hot['summary']['total_regular_mass_earth'] <= moons_cold['summary']['total_regular_mass_earth']
    print("  [OK] Strong inward migration suppresses regular moon retention")


def test_regular_and_irregular_moons_have_distinct_orbital_properties():
    print("\n" + SEP)
    print("TEST 3: Regular vs Irregular Orbital Structure")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.02, 8.0, np.random.default_rng(33))
    giant = _giant_planet(600.0, 9.0, formation_semi_major_au=8.5, migration_efficiency=0.30, giant_core_mass_earth=20.0)
    system = build_moon_system(giant, 1.0, 0.02, disk=disk, rng_seed=34)

    regular = system['regular_moons']
    irregular = system['irregular_moons']
    print(f"  regular={len(regular)}  irregular={len(irregular)}")
    print(f"  regular e max={max(m['eccentricity'] for m in regular):.4f}  irregular e mean={np.mean([m['eccentricity'] for m in irregular]):.4f}")
    print(f"  regular i max={max(m['inclination_deg'] for m in regular):.3f} deg  irregular i min={min(m['inclination_deg'] for m in irregular):.3f} deg")

    assert len(regular) > 0
    assert len(irregular) > 0
    assert max(m['eccentricity'] for m in regular) < np.mean([m['eccentricity'] for m in irregular])
    assert max(m['inclination_deg'] for m in regular) < min(m['inclination_deg'] for m in irregular)
    assert sum(1 for moon in irregular if moon['retrograde']) >= len(irregular) / 2
    print("  [OK] Regular moons stay cold/flat while irregulars remain eccentric and inclined")


def test_cpd_snow_line_creates_rock_ice_gradient():
    print("\n" + SEP)
    print("TEST 4: CPD Snow Line Produces Rocky-Icy Gradient")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.02, 8.0, np.random.default_rng(44))
    giant = _giant_planet(700.0, 7.5, formation_semi_major_au=6.5, migration_efficiency=0.28, giant_core_mass_earth=22.0)
    system = build_moon_system(giant, 1.0, 0.02, disk=disk, rng_seed=45)

    inner = [moon for moon in system['regular_moons'] if not moon['formed_beyond_cpd_snowline']]
    outer = [moon for moon in system['regular_moons'] if moon['formed_beyond_cpd_snowline']]
    print(f"  snow line={system['cpd']['snow_line_rp']:.2f} Rp  inner={len(inner)}  outer={len(outer)}")
    print(f"  inner mean ice={np.mean([m['ice_frac'] for m in inner]):.3f}  outer mean ice={np.mean([m['ice_frac'] for m in outer]):.3f}")

    assert len(inner) > 0
    assert len(outer) > 0
    assert np.mean([m['ice_frac'] for m in outer]) > np.mean([m['ice_frac'] for m in inner])
    assert any(moon['family'] in {'ganymede_like', 'callisto_like', 'titan_like'} for moon in outer)
    print("  [OK] CPD thermal structure creates inner rocky and outer icy moon families")


def test_neptune_class_hosts_form_sparser_moon_systems():
    print("\n" + SEP)
    print("TEST 5: Neptune-Class Hosts Support Sparse Moon Systems")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.018, 8.0, np.random.default_rng(52))
    neptune = _neptune_planet(17.0, 24.0, formation_semi_major_au=18.0, giant_core_mass_earth=10.5)
    jovian = _giant_planet(318.0, 24.0, formation_semi_major_au=18.0, giant_core_mass_earth=12.0)

    moons_neptune = build_moon_system(neptune, 1.0, 0.018, disk=disk, rng_seed=53)
    moons_jovian = build_moon_system(jovian, 1.0, 0.018, disk=disk, rng_seed=53)

    print(f"  Neptune CPD={moons_neptune['cpd']['mass_earth']:.5f} Me  moons={moons_neptune['summary']['n_regular']}+{moons_neptune['summary']['n_irregular']}")
    print(f"  Jovian  CPD={moons_jovian['cpd']['mass_earth']:.5f} Me  moons={moons_jovian['summary']['n_regular']}+{moons_jovian['summary']['n_irregular']}")
    print(f"  Neptune regime={moons_neptune['cpd']['host_regime']}  largest={moons_neptune['summary']['largest_moon_mass_earth']:.5f} Me")

    assert moons_neptune['cpd']['host_regime'] == 'neptunian'
    assert moons_neptune['cpd']['mass_earth'] < moons_jovian['cpd']['mass_earth']
    assert moons_neptune['summary']['n_regular'] <= 4
    assert (moons_neptune['summary']['n_regular'] + moons_neptune['summary']['n_irregular']) > 0
    assert moons_neptune['summary']['largest_moon_mass_earth'] <= moons_jovian['summary']['largest_moon_mass_earth']
    print("  [OK] Neptune-class hosts produce smaller but non-zero moon systems")


if __name__ == '__main__':
    _configure_stdout()
    test_more_massive_giants_build_more_massive_regular_systems()
    test_hot_jupiters_retain_fewer_regular_moons()
    test_regular_and_irregular_moons_have_distinct_orbital_properties()
    test_cpd_snow_line_creates_rock_ice_gradient()
    test_neptune_class_hosts_form_sparser_moon_systems()
    print("\n" + SEP)
    print("MOON TESTS PASSED")
    print(SEP)
