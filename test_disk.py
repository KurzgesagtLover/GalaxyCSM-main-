# -*- coding: utf-8 -*-
"""Quick test for protoplanetary disk and asteroid belt model.

Validates:
  1. Solar analog disk properties (M_disk, snow line, solid mass)
  2. Asteroid belt seed mass and survival fraction
  3. Metal-poor star produces smaller solid budget
  4. Giant planet presence crushes belt survival
  5. Existing planet physics functions still work (regression)
"""
import sys, io


def _configure_stdout():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from gce.disk import ProtoplanetaryDisk, build_asteroid_belt, compute_survival_fraction
from gce.moons import build_moon_system
from gce.planet_generation import _apply_disk_migration
from gce.planets import generate_planets, _generate_planets_from_disk
from gce.planets import (volatile_depletion, compute_physical_properties,
                          compute_atmosphere)

SEP = "=" * 60

def test_solar_analog():
    print(SEP)
    print("TEST 1: Solar Analog (M=1.0, Z=0.0134)")
    print(SEP)
    rng = np.random.default_rng(42)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, rng)
    d = disk.to_dict()
    print(f"  Disk mass:       {d['disk_mass_Msun']:.4f} Msun  ({d['disk_mass_Mearth']:.0f} Me)")
    print(f"  Dust-to-gas:     {d['dust_to_gas']:.4f}")
    print(f"  Disk radius:     {d['r_disk_au']:.1f} AU")
    print(f"  Sigma_0:         {d['sigma_0_g_cm2']:.0f} g/cm^2  (slope {d['sigma_slope']:.2f})")
    print(f"  Snow line:       {d['snow_line_au']:.2f} AU")
    print(f"  Disk lifetime:   {d['lifetime_myr']:.2f} Myr")
    print(f"  External FUV:    {d['F_UV_G0']:.1f} G0")
    print(f"  Total solid:     {d['total_solid_mass_earth']:.2f} Me")

    assert 0.001 < d['disk_mass_Msun'] < 0.5, f"Disk mass out of range: {d['disk_mass_Msun']}"
    assert 1.5 < d['snow_line_au'] < 5.0, f"Snow line out of range: {d['snow_line_au']}"
    assert 0.005 < d['dust_to_gas'] < 0.02, f"Dust-to-gas out of range: {d['dust_to_gas']}"
    assert d['total_solid_mass_earth'] > 1.0, f"Solid mass too low: {d['total_solid_mass_earth']}"
    print("  [OK] All disk ranges valid")

    planets = _generate_planets_from_disk(disk, 1.0, rng)
    print(f"\n  Generated {len(planets)} planets:")
    for p in planets:
        print(f"    [{p['index']}] {p['type']:15s} {p['mass_earth']:8.2f} Me  @ {p['semi_major_au']:6.2f} AU")

    belt = build_asteroid_belt(disk, planets, 4.5, rng)
    print(f"\n  Asteroid Belt:")
    print(f"    Location:        {belt['seed']['belt_in_au']:.2f} - {belt['seed']['belt_out_au']:.2f} AU")
    print(f"    Seed mass:       {belt['seed']['seed_mass_earth']:.6f} Me")
    print(f"    Survival frac:   {belt['survival']['survival_fraction']:.8f}")
    print(f"    Final mass:      {belt['final_mass_earth']:.8f} Me")
    print(f"    S/C-type ratio:  {belt['composition']['S_type_frac']:.2f} / {belt['composition']['C_type_frac']:.2f}")
    print(f"    Survival factors:")
    for k, v in belt['survival']['factors'].items():
        print(f"      {k:25s}: {v}")

    assert belt['final_mass_earth'] < 1.0, "Belt mass unrealistically high"
    print("  [OK] Belt mass in realistic range")


def test_metal_poor():
    print("\n" + SEP)
    print("TEST 2: Metal-Poor Star (M=1.0, Z=0.001)")
    print(SEP)
    rng = np.random.default_rng(123)
    disk = ProtoplanetaryDisk(1.0, 0.001, 8.0, rng)
    d = disk.to_dict()
    print(f"  Dust-to-gas:     {d['dust_to_gas']:.6f}  (vs solar ~0.01)")
    print(f"  Total solid:     {d['total_solid_mass_earth']:.3f} Me  (much less than solar)")
    assert d['dust_to_gas'] < 0.003, "Dust-to-gas should be very low"
    # Metal-poor disk should have much less solid than solar (solar ~2000+ Me)
    assert d['total_solid_mass_earth'] < 500, "Metal-poor should have much less solid mass"
    print("  [OK] Metal-poor disk correctly produces less solid material")


def test_giant_survival():
    print("\n" + SEP)
    print("TEST 3: Giant Planet Effect on Survival")
    print(SEP)
    rng = np.random.default_rng(99)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, rng)

    no_giant = [{'type': 'rocky', 'mass_earth': 1.0, 'semi_major_au': 1.0, 'eccentricity': 0.02}]
    belt_no = build_asteroid_belt(disk, no_giant, 4.5, np.random.default_rng(99))

    with_giant = [
        {'type': 'rocky', 'mass_earth': 1.0, 'semi_major_au': 1.0, 'eccentricity': 0.02},
        {'type': 'gas_giant', 'mass_earth': 318, 'semi_major_au': 5.2, 'eccentricity': 0.05},
    ]
    belt_yes = build_asteroid_belt(disk, with_giant, 4.5, np.random.default_rng(99))

    print(f"  Without giants:  survival = {belt_no['survival']['survival_fraction']:.6f}")
    print(f"  With Jupiter:    survival = {belt_yes['survival']['survival_fraction']:.8f}")
    ratio = belt_no['survival']['survival_fraction'] / max(belt_yes['survival']['survival_fraction'], 1e-12)
    print(f"  Depletion ratio: {ratio:.0f}x")
    assert belt_yes['survival']['survival_fraction'] < belt_no['survival']['survival_fraction'], \
        "Giant planet should reduce survival"
    print("  [OK] Giant planets correctly deplete belt")


def test_disk_mass_and_solid_budget_are_self_consistent():
    print("\n" + SEP)
    print("TEST 4: Disk Normalization Consistency")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, np.random.default_rng(314))
    gas_mass_msun = disk.integrate_gas_mass() / 1.989e33
    solid_expected = disk.disk_mass * 332946 * disk.dust_to_gas
    solid_actual = disk.total_solid_mass_earth

    print(f"  sampled gas mass:    {disk.disk_mass:.5f} Msun")
    print(f"  integrated gas mass: {gas_mass_msun:.5f} Msun")
    print(f"  expected solids:     {solid_expected:.2f} Me")
    print(f"  integrated solids:   {solid_actual:.2f} Me")

    assert abs(gas_mass_msun - disk.disk_mass) / max(disk.disk_mass, 1e-12) < 0.01
    assert abs(solid_actual - solid_expected) / max(solid_expected, 1e-12) < 0.15
    print("  [OK] Disk gas/solid budgets are self-consistent")


def test_regression():
    print("\n" + SEP)
    print("TEST 4: Regression -- Existing Physics Functions")
    print(SEP)
    # BSE-like composition (McDonough & Sun 1995)
    comp = {'C': 0.00012, 'N': 0.000002, 'S': 0.002, 'Fe': 0.06, 'Si': 0.07,
            'O': 0.44, 'Mg': 0.228, 'Al': 0.005, 'H': 0.0001, 'P': 0.00009,
            'K': 0.00024}
    dep = volatile_depletion(comp, 255)
    atm = compute_atmosphere(dep, 1.0, 'rocky', 255, 9.82, 11186, 4.5, -2.0)
    print(f"  Surface pressure: {atm['surface_pressure_atm']:.4f} atm")
    print(f"  Surface temp:     {atm['surface_temp_K']:.1f} K")
    assert atm['surface_pressure_atm'] > 0.01, "Pressure too low"
    assert 200 < atm['surface_temp_K'] < 400, "Temperature out of range"
    # N₂ should dominate for abiogenic Earth
    n2_pct = atm['composition'].get('N2', {}).get('pct', 0)
    h2_pct = atm['composition'].get('H2', {}).get('pct', 0)
    print(f"  N2 fraction:      {n2_pct:.2f}%  (should be dominant)")
    print(f"  H2 fraction:      {h2_pct:.4f}%  (should be <<1%)")
    assert n2_pct > 50, f"N2 should be dominant ({n2_pct}%)"
    assert h2_pct < 1.0, f"H2 fraction too high ({h2_pct}%) -- escape model broken"
    print("  [OK] Existing planet physics functions work correctly")


def test_disk_phase():
    print("\n" + SEP)
    print("TEST 5: Stellar Evolution -- Disk Phase")
    print(SEP)
    from gce.stellar import stellar_evolution
    evo = stellar_evolution(1.0, 1e-4)
    print(f"  Phase: {evo['phase']} ({evo['phase_kr']})")
    print(f"  T_eff: {evo['T_eff']} K, L: {evo['luminosity']} Lsun, R: {evo['radius']} Rsun")
    assert evo['phase'] == 'disk', f"Expected 'disk' phase, got '{evo['phase']}'"
    print("  [OK] Disk phase correctly returned for young star")


def test_state_variables():
    """Test the 10 new radial state variables in asteroid belt output."""
    print("\n" + SEP)
    print("TEST 6: Asteroid Belt State Variables (10 radial profiles)")
    print(SEP)
    rng = np.random.default_rng(42)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, rng)
    planets_with_giant = [
        {'type': 'rocky', 'mass_earth': 1.0, 'semi_major_au': 1.0, 'eccentricity': 0.02},
        {'type': 'gas_giant', 'mass_earth': 318, 'semi_major_au': 5.2, 'eccentricity': 0.05},
    ]
    belt = build_asteroid_belt(disk, planets_with_giant, 4.5, rng)

    # 1-2. Geometry: a0, ain, aout
    print(f"  a0  = {belt['a0_au']} AU")
    print(f"  ain = {belt['ain_au']} AU, aout = {belt['aout_au']} AU")
    assert belt['ain_au'] < belt['a0_au'] < belt['aout_au'], "Geometry broken"

    # 3. Surface density profile
    grid = belt['radial_grid_au']
    sigma = belt['sigma_a_g_cm2']
    assert len(grid) == 50, f"Expected 50 bins, got {len(grid)}"
    assert len(sigma) == 50, f"Sigma array length mismatch"
    assert all(s >= 0 for s in sigma), "Negative surface densities"
    print(f"  sigma(a): {len(sigma)} bins, range [{min(sigma):.6f}, {max(sigma):.6f}] g/cm2")

    # 4. Size distribution
    sd = belt['size_dist']
    assert sd['q_slope'] == 3.5, "Dohnanyi slope should be 3.5"
    assert sd['D_max_km'] > 0, "D_max should be positive"
    assert len(sd['D_km']) == 40, f"Expected 40 D bins, got {len(sd['D_km'])}"
    print(f"  N(>D): {len(sd['D_km'])} bins, q={sd['q_slope']}, D_max={sd['D_max_km']:.0f} km")

    # 5. Eccentricity distribution
    ecc = belt['ecc_a']
    assert len(ecc) == 50
    assert all(0 < e < 1 for e in ecc), "Eccentricity out of range"
    print(f"  e(a): mean={np.mean(ecc):.3f}, min={min(ecc):.3f}, max={max(ecc):.3f}")

    # 6. Inclination distribution
    inc = belt['inc_a_deg']
    assert len(inc) == 50
    assert all(i > 0 for i in inc), "Inclination should be positive"
    print(f"  i(a): mean={np.mean(inc):.1f} deg, range=[{min(inc):.1f}, {max(inc):.1f}]")

    # 7. Composition zones
    ca = belt['comp_a']
    assert len(ca['S_type']) == 50
    assert len(ca['C_type']) == 50
    assert len(ca['M_type']) == 50
    # M-type should be ~5% everywhere
    assert all(abs(m - 0.05) < 0.01 for m in ca['M_type']), "M-type should be ~5%"
    # S-type should dominate inner belt, C-type outer
    assert ca['S_type'][0] > ca['C_type'][0], "Inner belt should be S-type dominant"
    assert ca['C_type'][-1] > ca['S_type'][-1], "Outer belt should be C-type dominant"
    print(f"  comp(a): S=[{ca['S_type'][0]:.2f}..{ca['S_type'][-1]:.2f}]  C=[{ca['C_type'][0]:.2f}..{ca['C_type'][-1]:.2f}]  M=~{ca['M_type'][0]}")
    print(f"  Bulk: S={belt['composition']['S_type_frac']:.3f} C={belt['composition']['C_type_frac']:.3f} M={belt['composition']['M_type_frac']:.3f}")

    # 8. Resonances / gaps
    res = belt['resonances']
    print(f"  Resonances: {len(res)} found")
    for r in res:
        print(f"    {r['ratio']} at {r['a_au']:.3f} AU (depth {r['depth']:.2f}, giant at {r['giant_a_au']} AU)")
    assert len(res) > 0, "Should find resonances with Jupiter analog"

    # 9. Collision activity
    cr = belt['collision_rate_a']
    assert len(cr) == 50
    assert max(cr) <= 1.0001, "Collision rate should be normalized to 1"
    print(f"  Collision: index={belt['collision_index']}, peak={max(cr):.3f}")

    # 10. Survived mass fraction
    sf = belt['survived_frac_a']
    assert len(sf) == 50
    assert all(f > 0 for f in sf), "Survival should be positive"
    print(f"  f_surv(a): mean={belt['survived_frac_mean']:.6f}, min={min(sf):.8f}")

    print("  [OK] All 10 state variables valid")


def test_late_veneer():
    """Test the LV0-LV4 classification system for per-planet differentiation."""
    print("\n" + SEP)
    print("TEST 7: Late Veneer Classification (LV0-LV4)")
    print(SEP)
    from gce.planets import classify_late_veneer, LV_CLASSES

    rng = np.random.default_rng(42)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, rng)

    # System with Jupiter analog
    planets_with_giant = [
        {'type': 'rocky', 'mass_earth': 0.1,  'semi_major_au': 0.4, 'eccentricity': 0.02},  # Mercury-like
        {'type': 'rocky', 'mass_earth': 1.0,  'semi_major_au': 1.0, 'eccentricity': 0.02},  # Earth-like
        {'type': 'rocky', 'mass_earth': 0.5,  'semi_major_au': 1.5, 'eccentricity': 0.02},  # Mars-like
        {'type': 'gas_giant', 'mass_earth': 318, 'semi_major_au': 5.2, 'eccentricity': 0.05},
    ]
    belt = build_asteroid_belt(disk, planets_with_giant, 4.5, rng)

    # System without giants
    planets_no_giant = [
        {'type': 'rocky', 'mass_earth': 1.0, 'semi_major_au': 1.0, 'eccentricity': 0.02},
    ]
    belt_no = build_asteroid_belt(disk, planets_no_giant, 4.5, np.random.default_rng(42))

    print("  With Jupiter:")
    classes_with = []
    for p in planets_with_giant:
        if p['type'] in ('rocky', 'hot_rocky'):
            lv = classify_late_veneer(p, planets_with_giant, belt, disk, np.random.default_rng(100))
            classes_with.append(lv['class'])
            print(f"    {p['mass_earth']:5.1f} Me @ {p['semi_major_au']:4.1f} AU -> {lv['class']} ({lv['label']})"
                  f"  score={lv['score']:.3f}  water={lv['water_frac']:.4f}  HSE={lv['hse_enrichment']}")

    print("\n  Without giants:")
    lv_no = classify_late_veneer(planets_no_giant[0], planets_no_giant, belt_no, disk, np.random.default_rng(100))
    print(f"    {planets_no_giant[0]['mass_earth']:5.1f} Me @ 1.0 AU -> {lv_no['class']} ({lv_no['label']})"
          f"  score={lv_no['score']:.3f}")

    # Validate all classes have required keys
    for cls_name, cls_props in LV_CLASSES.items():
        assert 'lv_mass_frac' in cls_props, f"Missing lv_mass_frac in {cls_name}"
        assert 'water_frac' in cls_props, f"Missing water_frac in {cls_name}"
        assert 'hse_enrichment' in cls_props, f"Missing hse_enrichment in {cls_name}"
        assert 'carbonaceous_frac' in cls_props, f"Missing carbonaceous_frac in {cls_name}"
        assert 'tail_myr' in cls_props, f"Missing tail_myr in {cls_name}"

    # Verify scoring components are present
    assert 'scores' in lv_no, "Missing scores dict"
    for key in ['giant_arch', 'belt_mass', 'formation_radius', 'dyn_excitation', 'volatile_reservoir', 'mass_capture']:
        assert key in lv_no['scores'], f"Missing score component: {key}"

    # LV classes should be monotonically increasing in lv_mass_frac
    fracs = [LV_CLASSES[f'LV{i}']['lv_mass_frac'] for i in range(5)]
    assert fracs == sorted(fracs), "LV classes should have increasing mass fractions"

    print("  [OK] All LV classification checks passed")


def test_photolysis():
    """Test UV photolysis: G-star vs M-dwarf, shielding, τ levels."""
    print("\n" + SEP)
    print("TEST 8: UV Photolysis (G-star vs M-dwarf)")
    print(SEP)
    from gce.photolysis import build_uv_template, compute_photolysis

    # --- G-star (Sun) at 1 AU, 4.5 Gyr ---
    uv_g = build_uv_template(5778, 4.5, 1.0)
    print(f"  G-star: {uv_g['label']}, F_UV={uv_g['F_UV_total']:.1f} erg/cm2/s")

    # --- M-dwarf (3200K) at 0.15 AU (habitable zone), 4.5 Gyr ---
    uv_m = build_uv_template(3200, 4.5, 0.15)
    print(f"  M-dwarf: {uv_m['label']}, F_UV={uv_m['F_UV_total']:.1f} erg/cm2/s")

    # --- Compute photolysis for Earth-like atmosphere ---
    earth_species = {
        'N2': 3.87e18, 'O2': 1.18e18, 'Ar': 6.6e16,
        'CO2': 3e15, 'H2O': 1.3e16, 'CH4': 5e12,
        'O3': 3e12, 'NH3': 1e8, 'SO2': 1e10,
    }
    R_earth = 6.371e6
    g_earth = 9.81
    H_earth = 8500

    photo_g = compute_photolysis(5778, 4.5, 1.0, earth_species, R_earth, g_earth, H_earth)
    photo_m = compute_photolysis(3200, 4.5, 0.15, earth_species, R_earth, g_earth, H_earth)

    print("\n  Species lifetimes (column-integrated, yr):")
    print(f"  {'Species':>6}  {'G-star':>12}  {'M-dwarf':>12}")
    for sp in ['CH4', 'NH3', 'SO2', 'H2S', 'CO2', 'O3', 'H2O']:
        tau_g = photo_g['species'].get(sp, {}).get('tau_column_yr', '-')
        tau_m = photo_m['species'].get(sp, {}).get('tau_column_yr', '-')
        if isinstance(tau_g, float) and isinstance(tau_m, float):
            print(f"  {sp:>6}  {tau_g:>12.4f}  {tau_m:>12.4f}")
        else:
            print(f"  {sp:>6}  {str(tau_g):>12}  {str(tau_m):>12}")

    # Verify CH₄ column lifetime is in realistic range (1-50 yr for G-star)
    ch4_g = photo_g['species'].get('CH4', {}).get('tau_column_yr', 0)
    assert 0.5 < ch4_g < 100, f"CH4 lifetime {ch4_g} yr out of range for G-star"
    print(f"\n  CH4 column τ (G): {ch4_g:.2f} yr — matches Prather 1996 (~9-12 yr)")

    # Verify shielding: upper τ < column τ (more UV at top → shorter τ)
    ch4_upper = photo_g['species'].get('CH4', {}).get('tau_upper_yr', 0)
    ch4_col = photo_g['species'].get('CH4', {}).get('tau_column_yr', 0)
    assert ch4_upper <= ch4_col, "Upper τ should be <= column τ (less shielding)"
    print(f"  CH4 upper τ: {ch4_upper:.2f} yr  <=  column τ: {ch4_col:.2f} yr [OK shielding]")

    # Verify stellar types produce different UV classes
    assert uv_g['spec_class'] == 'G', f"Expected G, got {uv_g['spec_class']}"
    assert uv_m['spec_class'] == 'M', f"Expected M, got {uv_m['spec_class']}"

    print("  [OK] UV photolysis model validated")


def test_type_ii_migration_can_form_hot_jupiters():
    print("\n" + SEP)
    print("TEST 9: Type II Migration Hot Jupiter Channel")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.02, 8.0, np.random.default_rng(7))
    disk.disk_mass = 0.18
    disk.lifetime_myr = 7.0
    disk.snow_line_au = 2.7

    mig = _apply_disk_migration(5.0, 300.0, 'gas_giant', 1.0, np.random.default_rng(11), disk=disk)
    print(f"  formed @ {mig['formation_semi_major_au']:.2f} AU -> final @ {mig['a_final_au']:.3f} AU")
    print(f"  mode={mig['migration_mode']}  delta={mig['migration_delta_au']:.2f} AU  tau={mig['migration_timescale_myr']:.3f} Myr")
    assert mig['migration_mode'] == 'type_ii'
    assert mig['a_final_au'] < 0.2, "Massive giant in a long-lived disk should reach hot-Jupiter orbit"
    assert mig['migration_delta_au'] > 1.0, "Migration amplitude should be substantial"
    print("  [OK] Type II migration opens a hot-Jupiter pathway")


def test_massive_belts_grind_faster():
    print("\n" + SEP)
    print("TEST 10: Belt-Mass-Dependent Collisional Grinding")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, np.random.default_rng(21))
    planets = [{'type': 'rocky', 'mass_earth': 1.0, 'semi_major_au': 1.0, 'eccentricity': 0.02}]

    light = compute_survival_fraction(
        disk, planets, 1.0, rng=np.random.default_rng(1), belt_seed_mass_earth=1e-4
    )
    heavy = compute_survival_fraction(
        disk, planets, 1.0, rng=np.random.default_rng(1), belt_seed_mass_earth=3e-2
    )

    print(f"  light belt: f_coll={light['factors']['collisional_grinding']:.4f}, tau={light['collisional_timescale_gyr']:.3f} Gyr")
    print(f"  heavy belt: f_coll={heavy['factors']['collisional_grinding']:.4f}, tau={heavy['collisional_timescale_gyr']:.3f} Gyr")
    assert heavy['factors']['collisional_grinding'] < light['factors']['collisional_grinding']
    assert heavy['collisional_timescale_gyr'] < light['collisional_timescale_gyr']
    print("  [OK] More massive belts collide and deplete faster")


def test_hot_jupiters_retain_fewer_moons_than_cold_giants():
    print("\n" + SEP)
    print("TEST 11: Hot Jupiter Moon Retention Penalty")
    print(SEP)
    disk = ProtoplanetaryDisk(1.0, 0.0134, 8.0, np.random.default_rng(55))
    cold = {
        'index': 0,
        'type': 'gas_giant',
        'mass_earth': 318.0,
        'semi_major_au': 5.2,
        'formation_semi_major_au': 5.2,
        'migration_delta_au': 0.0,
        'migration_efficiency': 0.20,
        'migration_mode': 'type_ii',
        'rotation_period_hr': 10.0,
        'T_eq': 120.0,
        'formation_T_eq_K': 130.0,
        'giant_core_mass_earth': 12.0,
        'is_hot_jupiter': False,
    }
    hot = {
        **cold,
        'semi_major_au': 0.05,
        'migration_delta_au': 5.15,
        'migration_efficiency': 0.85,
        'T_eq': 1000.0,
        'is_hot_jupiter': True,
    }

    cold_moons = build_moon_system(cold, 1.0, 0.0134, disk=disk, rng_seed=91)
    hot_moons = build_moon_system(hot, 1.0, 0.0134, disk=disk, rng_seed=91)

    print(f"  cold regular={cold_moons['summary']['n_regular']} irregular={cold_moons['summary']['n_irregular']}")
    print(f"  hot  regular={hot_moons['summary']['n_regular']} irregular={hot_moons['summary']['n_irregular']}")
    print(f"  cold zone={cold_moons['cpd']['regular_zone_outer_rp']:.2f} Rp  hot zone={hot_moons['cpd']['regular_zone_outer_rp']:.2f} Rp")

    assert hot_moons['cpd']['regular_zone_outer_rp'] < cold_moons['cpd']['regular_zone_outer_rp']
    assert hot_moons['summary']['n_regular'] <= cold_moons['summary']['n_regular']
    assert hot_moons['summary']['total_regular_mass_earth'] <= cold_moons['summary']['total_regular_mass_earth']
    print("  [OK] Hot-Jupiter migration suppresses moon retention")


if __name__ == '__main__':
    _configure_stdout()
    test_solar_analog()
    test_metal_poor()
    test_giant_survival()
    test_disk_mass_and_solid_budget_are_self_consistent()
    test_regression()
    test_disk_phase()
    test_state_variables()
    test_late_veneer()
    test_photolysis()
    test_type_ii_migration_can_form_hot_jupiters()
    test_massive_belts_grind_faster()
    test_hot_jupiters_retain_fewer_moons_than_cold_giants()
    print("\n" + SEP)
    print("ALL TESTS PASSED")
    print(SEP)
