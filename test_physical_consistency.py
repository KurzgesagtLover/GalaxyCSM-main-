from gce.planet_atmosphere import _sub_neptune_atmo
from gce.photolysis import build_uv_template, compute_photolysis, estimate_flare_boost, integrated_xuv_history
from gce.planets import (
    _habitability_assessment,
    build_outgassing_reservoir,
    compute_atmosphere,
    compute_physical_properties,
    compute_radiation_defense,
    differentiate_full,
    estimate_equilibrium_temperature,
    estimate_spectral_habitable_zone,
    volatile_depletion,
)


LV_WET = {
    'class': 'LV3',
    'label': 'wet veneer',
    'score': 1.0,
    'lv_mass_frac': 0.015,
    'water_frac': 0.01,
    'hse_enrichment': 2.0,
    'carbonaceous_frac': 0.4,
    'volatile_enrichment': 2.5,
    'tail_myr': 400,
    'scores': {},
}


def test_formation_temperature_retains_volatiles():
    bulk = {'C': 1.0e-3, 'N': 4.0e-4, 'S': 1.0e-2, 'O': 0.40}
    t_form = estimate_equilibrium_temperature(1.0, 1.0, 0.2)
    t_red_giant = estimate_equilibrium_temperature(100.0, 1.0, 0.2)

    retained_form = volatile_depletion(bulk, t_form)
    retained_now = volatile_depletion(bulk, t_red_giant)

    assert retained_form['C'] > retained_now['C'] * 50
    assert retained_form['N'] > retained_now['N'] * 20
    assert retained_form['S'] > retained_now['S'] * 5

    print('formation-temperature retention: OK')
    print(f"  T_form={t_form:.1f} K  T_red_giant={t_red_giant:.1f} K")


def test_outgassing_reservoir_uses_surface_layers():
    bulk = {
        'O': 0.44,
        'Fe': 0.32,
        'Si': 0.21,
        'Mg': 0.15,
        'Al': 0.02,
        'C': 2.0e-3,
        'N': 2.0e-4,
        'S': 2.0e-2,
        'P': 6.0e-4,
    }
    diff = differentiate_full(1.0, bulk, -2.0, diff_progress=1.0, lv_class_params=LV_WET)
    surface = build_outgassing_reservoir(diff)

    assert surface.get('H', 0.0) > 0.0
    assert surface['Fe'] < bulk['Fe'] * 0.3
    assert surface['S'] < bulk['S']

    atmo = compute_atmosphere(surface, 1.0, 'rocky', 255.0, 9.81, 11186.0, 4.5, -2.0)
    assert atmo['composition'].get('H2O', {}).get('pct', 0.0) > 0.0
    assert atmo['composition'].get('N2', {}).get('partial_P_atm', 0.0) < 5.0
    assert atmo['composition'].get('CO2', {}).get('partial_P_atm', 0.0) < 5.0

    print('surface-reservoir outgassing: OK')
    print(
        f"  accessible H={surface['H']:.6e} Fe={surface['Fe']:.6e} S={surface['S']:.6e} "
        f"N2={atmo['composition'].get('N2', {}).get('partial_P_atm', 0.0):.3f} atm "
        f"CO2={atmo['composition'].get('CO2', {}).get('partial_P_atm', 0.0):.3f} atm"
    )


def test_uv_template_respects_luminosity_scaling():
    uv_solar = build_uv_template(5778, 4.5, 1.0, luminosity_lsun=1.0)
    uv_giant = build_uv_template(5778, 4.5, 1.0, luminosity_lsun=100.0)

    ratio = uv_giant['F_UV_total'] / max(uv_solar['F_UV_total'], 1e-30)
    assert ratio > 90.0

    print('uv luminosity scaling: OK')
    print(f"  UV ratio={ratio:.1f}x")


def test_uv_template_declines_with_age():
    uv_young = build_uv_template(5778, 0.05, 1.0, luminosity_lsun=1.0)
    uv_mid = build_uv_template(5778, 0.5, 1.0, luminosity_lsun=1.0)
    uv_old = build_uv_template(5778, 4.5, 1.0, luminosity_lsun=1.0)

    assert uv_young['F_UV_total'] > uv_mid['F_UV_total'] > uv_old['F_UV_total']

    uv_m_far = build_uv_template(3200, 4.5, 1.0)
    uv_m_hz = build_uv_template(3200, 4.5, 0.15)
    assert uv_m_hz['F_UV_total'] > uv_m_far['F_UV_total'] * 30.0

    print('uv age scaling: OK')
    print(
        f"  young={uv_young['F_UV_total']:.1f} "
        f"mid={uv_mid['F_UV_total']:.1f} old={uv_old['F_UV_total']:.1f} "
        f"M@HZ={uv_m_hz['F_UV_total']:.1f}"
    )


def test_abiogenic_o2_is_sink_limited():
    bulk = {
        'O': 0.44, 'Fe': 0.08, 'Si': 0.21, 'Mg': 0.16, 'Al': 0.02,
        'C': 1.2e-3, 'N': 2.0e-4, 'S': 8.0e-3, 'H': 1.2e-4, 'P': 2.0e-4,
    }
    atmo = compute_atmosphere(bulk, 1.0, 'rocky', 255.0, 9.81, 11186.0, 4.5, -2.0)

    assert atmo['abiogenic_o2_pal'] < 1e-6
    assert atmo['volcanic_o2_sink_mol_yr'] > 0.0
    assert atmo['composition'].get('O2', {}).get('partial_P_atm', 0.0) < 1e-4

    print('abiogenic O2 sink balance: OK')
    print(
        f"  O2={atmo['abiogenic_o2_pal']:.3e} PAL  "
        f"sink={atmo['volcanic_o2_sink_mol_yr']:.3e} mol/yr"
    )


def test_explicit_modern_earth_oxygenation_matches_anchor():
    bulk = {
        'O': 0.44, 'Fe': 0.32, 'Si': 0.21, 'Mg': 0.15, 'Al': 0.02,
        'C': 2.0e-3, 'N': 2.0e-4, 'S': 2.0e-2, 'P': 6.0e-4,
    }
    diff = differentiate_full(1.0, bulk, -2.0, diff_progress=1.0, lv_class_params=LV_WET)
    surface = build_outgassing_reservoir(diff)
    atmo = compute_atmosphere(
        surface, 1.0, 'rocky', 255.0, 9.81, 11186.0, 4.57, -2.0,
        surface_relative_humidity=0.65,
        biotic_o2_atm=0.2095,
    )

    o2_partial = atmo['composition'].get('O2', {}).get('partial_P_atm', 0.0)
    assert abs(atmo['surface_pressure_atm'] - 1.0) < 0.12
    assert abs(atmo['surface_temp_K'] - 288.0) < 6.5
    assert abs(o2_partial - 0.2095) < 0.05

    print('modern Earth oxygenation anchor: OK')
    print(
        f"  P={atmo['surface_pressure_atm']:.3f} atm  "
        f"T={atmo['surface_temp_K']:.1f} K  "
        f"O2={o2_partial:.3f} atm"
    )


def test_h2o_feedback_iterates_on_surface_temperature():
    bulk = {
        'O': 0.44, 'Fe': 0.06, 'Si': 0.21, 'Mg': 0.15, 'Al': 0.02,
        'C': 2.5e-3, 'N': 4.0e-4, 'S': 1.0e-2, 'H': 8.0e-4, 'P': 3.0e-4,
    }
    atmo = compute_atmosphere(bulk, 1.2, 'rocky', 265.0, 10.5, 12000.0, 4.0, -1.0)

    assert atmo['feedback_diagnostics']['iterations'] >= 2
    assert atmo['feedback_diagnostics']['cc_amplification'] > 1.0
    assert atmo['surface_temp_K'] > atmo['T_eq_K']

    print('iterative H2O feedback: OK')
    print(
        f"  iter={atmo['feedback_diagnostics']['iterations']}  "
        f"cc={atmo['feedback_diagnostics']['cc_amplification']:.2f}  "
        f"Tsurf={atmo['surface_temp_K']:.1f} K"
    )


def test_spectral_hz_depends_on_stellar_teff():
    hz_solar = estimate_spectral_habitable_zone(1.0, 5778.0)
    hz_m = estimate_spectral_habitable_zone(1.0, 3300.0)
    hz_f = estimate_spectral_habitable_zone(1.0, 6500.0)

    assert hz_m['conservative_inner_au'] < hz_solar['conservative_inner_au']
    assert hz_m['conservative_outer_au'] < hz_solar['conservative_outer_au']
    assert hz_f['conservative_outer_au'] > hz_solar['conservative_outer_au']

    print('spectral HZ scaling: OK')
    print(
        f"  solar={hz_solar['conservative_inner_au']:.3f}-{hz_solar['conservative_outer_au']:.3f} AU  "
        f"M={hz_m['conservative_inner_au']:.3f}-{hz_m['conservative_outer_au']:.3f} AU  "
        f"F={hz_f['conservative_inner_au']:.3f}-{hz_f['conservative_outer_au']:.3f} AU"
    )


def test_mk_dwarfs_have_longer_xuv_saturation():
    xuv_g = integrated_xuv_history(5778.0, 1.0, 1.0, luminosity_lsun=1.0)
    xuv_k = integrated_xuv_history(4400.0, 1.0, 0.45, luminosity_lsun=0.15)
    xuv_m = integrated_xuv_history(3300.0, 1.0, 0.12, luminosity_lsun=0.02)

    assert xuv_k['t_sat_gyr'] > xuv_g['t_sat_gyr']
    assert xuv_m['t_sat_gyr'] > xuv_k['t_sat_gyr']
    assert xuv_m['xuv_fluence_erg_cm2'] > xuv_g['xuv_fluence_erg_cm2']

    print('M/K dwarf long XUV saturation: OK')
    print(
        f"  G sat={xuv_g['t_sat_gyr']:.2f}  K sat={xuv_k['t_sat_gyr']:.2f}  "
        f"M sat={xuv_m['t_sat_gyr']:.2f} Gyr"
    )


def test_flare_boost_is_passed_into_uv_template():
    boost_m = estimate_flare_boost(3200.0, 0.2)
    boost_k = estimate_flare_boost(4400.0, 0.2)
    uv_quiet = build_uv_template(3200.0, 0.2, 0.12, flare_boost=1.0)
    uv_flare = build_uv_template(3200.0, 0.2, 0.12, flare_boost=boost_m)

    assert boost_m > boost_k > 1.0
    assert uv_flare['flare_boost'] == round(boost_m, 3)
    assert uv_flare['F_UV_total'] > uv_quiet['F_UV_total']

    print('flare boost threading: OK')
    print(
        f"  boost_M={boost_m:.2f} boost_K={boost_k:.2f} "
        f"UV quiet={uv_quiet['F_UV_total']:.1f} flare={uv_flare['F_UV_total']:.1f}"
    )


def test_flare_boost_changes_photolysis_rates():
    species = {'CH4': 5e12, 'CO2': 3e15, 'H2O': 1.3e16, 'O2': 1e15}
    photo_quiet = compute_photolysis(3200.0, 0.2, 0.12, species, 6.371e6, 9.81, 8500, flare_boost=1.0)
    photo_flare = compute_photolysis(3200.0, 0.2, 0.12, species, 6.371e6, 9.81, 8500, flare_boost=4.0)

    tau_quiet = photo_quiet['species']['CH4']['tau_column_yr']
    tau_flare = photo_flare['species']['CH4']['tau_column_yr']
    assert photo_flare['flare_boost'] == 4.0
    assert tau_flare < tau_quiet

    print('flare boost photolysis response: OK')
    print(f"  CH4 tau quiet={tau_quiet:.2f} yr flare={tau_flare:.2f} yr")


def test_flare_boost_changes_radiation_dose():
    comp = {
        'O2': {'mass_kg': 1.0e15},
        'O3': {'mass_kg': 1.0e12},
        'CO2': {'mass_kg': 3.0e15},
        'H2O': {'mass_kg': 1.0e16},
    }
    rad_quiet = compute_radiation_defense(
        1.0, 6371.0, 35.0, 5.0e18, 1.0, comp, 0.12, 3200.0, 0.02, 0.2, flare_boost=1.0
    )
    rad_flare = compute_radiation_defense(
        1.0, 6371.0, 35.0, 5.0e18, 1.0, comp, 0.12, 3200.0, 0.02, 0.2, flare_boost=5.0
    )

    assert rad_flare['flare_boost'] == 5.0
    assert rad_flare['incident_W_m2']['uv'] > rad_quiet['incident_W_m2']['uv']
    assert rad_flare['incident_W_m2']['x_euv'] > rad_quiet['incident_W_m2']['x_euv']
    assert rad_flare['wind_pressure_nPa'] > rad_quiet['wind_pressure_nPa']

    print('flare boost radiation dose: OK')
    print(
        f"  UV quiet={rad_quiet['incident_W_m2']['uv']:.4f} flare={rad_flare['incident_W_m2']['uv']:.4f}  "
        f"XEUV quiet={rad_quiet['incident_W_m2']['x_euv']:.4f} flare={rad_flare['incident_W_m2']['x_euv']:.4f}"
    )


def test_sub_neptune_envelope_loss_tracks_xuv_history():
    bulk = {'C': 1.5e-3, 'N': 3.0e-4, 'H': 6.0e-4}
    cool = _sub_neptune_atmo(
        6.0, bulk, T_eq=450.0, metallicity=0.0134, age_gyr=5.0,
        semi_major_au=0.45, star_teff=5778.0, luminosity_lsun=1.0,
    )
    hot = _sub_neptune_atmo(
        6.0, bulk, T_eq=900.0, metallicity=0.0134, age_gyr=5.0,
        semi_major_au=0.05, star_teff=3300.0, luminosity_lsun=0.02,
    )

    assert hot['xuv_history']['xuv_fluence_erg_cm2'] > cool['xuv_history']['xuv_fluence_erg_cm2']
    assert hot['photoevap_loss_metric'] > cool['photoevap_loss_metric']
    assert hot['f_envelope'] < cool['f_envelope']
    assert hot['atm_mass_kg'] > 0.0

    print('sub-neptune mass loss channels: OK')
    print(
        f"  cool f_env={cool['f_envelope']:.4f} hot f_env={hot['f_envelope']:.4f} "
        f"hot regime={hot['envelope_regime']}"
    )


def test_habitability_score_penalizes_runaway_worlds():
    planet = {
        'type': 'rocky',
        'semi_major_au': 1.0,
        'in_hz_dynamic': True,
        'in_hz_optimistic': True,
        'hz_bounds_au': {
            'conservative_inner': 0.95,
            'conservative_outer': 1.70,
        },
    }
    temperate_atmo = {
        'surface_temp_K': 288.0,
        'surface_pressure_atm': 1.0,
        'water_ocean_mass_kg': 1.2e21,
        'composition': {'H2O': {'vol_frac': 0.012}},
        'climate_regime': {
            'stable_surface_water': True,
            'moist_greenhouse': False,
            'runaway_greenhouse': False,
            'cold_trap_collapse_risk': False,
        },
    }
    runaway_atmo = {
        'surface_temp_K': 390.0,
        'surface_pressure_atm': 250.0,
        'water_ocean_mass_kg': 0.0,
        'composition': {'H2O': {'vol_frac': 0.002}},
        'climate_regime': {
            'stable_surface_water': False,
            'moist_greenhouse': True,
            'runaway_greenhouse': True,
            'cold_trap_collapse_risk': False,
        },
    }
    quiet_rad = {'surface_W_m2': {'uv': 1.0, 'x_euv': 0.01}}
    harsh_rad = {'surface_W_m2': {'uv': 18.0, 'x_euv': 0.6}}

    temperate = _habitability_assessment(planet, temperate_atmo, quiet_rad)
    runaway = _habitability_assessment(planet, runaway_atmo, harsh_rad)

    assert temperate['label'] == 'candidate'
    assert temperate['score'] > runaway['score']
    assert 'runaway_greenhouse' in runaway['reasons']
    assert 'radiation_stress' in runaway['reasons']

    print('habitability score ranking: OK')
    print(f"  temperate={temperate['score']:.3f} runaway={runaway['score']:.3f}")


def test_close_in_rocky_planets_tidally_lock():
    phys = compute_physical_properties(
        1.0, 'rocky', 0.03, 0.2, 24.0, 5.0, 0.01, 5.0, L_star_Lsun=0.008
    )

    assert phys['tidally_locked'] is True
    assert phys['rotation_effective_hr'] > 100.0
    assert phys['tidal_lock_timescale_gyr'] < phys['age_gyr']

    print('tidal locking timescale: OK')
    print(
        f"  t_lock={phys['tidal_lock_timescale_gyr']:.4f} Gyr  "
        f"rot_eff={phys['rotation_effective_hr']:.1f} h"
    )


def test_slow_rotator_clouds_cool_inner_edge_climate():
    bulk = {
        'O': 0.44, 'Fe': 0.06, 'Si': 0.21, 'Mg': 0.15, 'Al': 0.02,
        'C': 1.8e-3, 'N': 3.0e-4, 'S': 9.0e-3, 'H': 5.0e-4, 'P': 2.0e-4,
    }
    fast = compute_atmosphere(
        bulk, 1.0, 'rocky', 300.0, 9.81, 11186.0, 4.5, -1.0,
        rotation_period_hr=24.0, orbital_period_days=20.0, tidally_locked=False, tidal_heating_TW=0.0
    )
    locked = compute_atmosphere(
        bulk, 1.0, 'rocky', 300.0, 9.81, 11186.0, 4.5, -1.0,
        rotation_period_hr=480.0, orbital_period_days=20.0, tidally_locked=True, tidal_heating_TW=0.0
    )

    assert locked['feedback_diagnostics']['rotation_cloud_cooling_K'] > 0.0
    assert locked['surface_temp_K'] < fast['surface_temp_K']

    print('slow-rotator cloud cooling: OK')
    print(
        f"  fast={fast['surface_temp_K']:.1f} K  "
        f"locked={locked['surface_temp_K']:.1f} K  "
        f"cooling={locked['feedback_diagnostics']['rotation_cloud_cooling_K']:.1f} K"
    )


def test_runaway_greenhouse_flagged_for_steam_worlds():
    bulk = {
        'O': 0.44, 'Fe': 0.06, 'Si': 0.21, 'Mg': 0.15, 'Al': 0.02,
        'C': 2.5e-3, 'N': 4.0e-4, 'S': 1.0e-2, 'H': 1.2e-3, 'P': 3.0e-4,
    }
    atmo = compute_atmosphere(
        bulk, 1.3, 'rocky', 310.0, 11.0, 12500.0, 4.0, -0.5,
        rotation_period_hr=600.0, orbital_period_days=12.0, tidally_locked=True, tidal_heating_TW=0.0
    )

    assert atmo['climate_regime']['moist_greenhouse'] or atmo['climate_regime']['runaway_greenhouse']
    assert atmo['climate_regime']['stable_surface_water'] is False

    print('runaway/moist greenhouse diagnostics: OK')
    print(
        f"  moist={atmo['climate_regime']['moist_greenhouse']}  "
        f"runaway={atmo['climate_regime']['runaway_greenhouse']}  "
        f"Tsurf={atmo['surface_temp_K']:.1f} K"
    )


if __name__ == '__main__':
    test_formation_temperature_retains_volatiles()
    test_outgassing_reservoir_uses_surface_layers()
    test_uv_template_respects_luminosity_scaling()
    test_uv_template_declines_with_age()
    test_abiogenic_o2_is_sink_limited()
    test_h2o_feedback_iterates_on_surface_temperature()
    test_spectral_hz_depends_on_stellar_teff()
    test_mk_dwarfs_have_longer_xuv_saturation()
    test_flare_boost_is_passed_into_uv_template()
    test_flare_boost_changes_photolysis_rates()
    test_flare_boost_changes_radiation_dose()
    test_sub_neptune_envelope_loss_tracks_xuv_history()
    test_habitability_score_penalizes_runaway_worlds()
    test_close_in_rocky_planets_tidally_lock()
    test_slow_rotator_clouds_cool_inner_edge_climate()
    test_runaway_greenhouse_flagged_for_steam_worlds()
    print('physical consistency checks passed')
