from gce.config import Z_SUN
from gce.stellar import (
    _overview_alive_until,
    _overview_phase_bucket,
    generate_galaxy,
    stellar_evolution as public_stellar_evolution,
)
from gce.stellar_properties import _ms_lifetime, _ms_state
from gce.stellar_remnants import ns_cooling, wd_cooling
from gce.stellar_tracks import stellar_evolution as track_stellar_evolution


def test_properties_module():
    lum_solar, temp_solar, radius_solar = _ms_state(1.0, metallicity_z=0.02)
    lum_poor, temp_poor, radius_poor = _ms_state(1.0, metallicity_z=1e-4)

    assert temp_poor > temp_solar
    assert radius_poor < radius_solar
    assert lum_poor > 0 and lum_solar > 0

    print('stellar_properties: OK')
    print(f'  T_solar={temp_solar:.1f}K T_poor={temp_poor:.1f}K')


def test_remnants_module():
    wd_young = wd_cooling(0.05, wd_mass=0.6)
    wd_mid = wd_cooling(5.0, wd_mass=0.6)
    wd_old = wd_cooling(20.0, wd_mass=0.6)
    assert wd_young['luminosity'] > wd_mid['luminosity'] > wd_old['luminosity']
    assert wd_young['temperature'] > wd_mid['temperature'] > wd_old['temperature']

    ns_young = ns_cooling(1e-4)
    ns_old = ns_cooling(0.1)
    assert ns_young['temperature'] > ns_old['temperature']
    assert ns_young['luminosity'] > ns_old['luminosity']

    print('stellar_remnants: OK')
    print(f"  WD_L={wd_young['luminosity']:.3e}->{wd_old['luminosity']:.3e}")
    print(f"  NS_T={ns_young['temperature']:.2e}->{ns_old['temperature']:.2e}")


def test_public_reexport_and_tracks():
    assert public_stellar_evolution is track_stellar_evolution

    t_ms = _ms_lifetime(1.0, metallicity_z=0.02)
    ms_state = public_stellar_evolution(1.0, 0.5 * t_ms, metallicity_z=0.02)
    post_ms_state = public_stellar_evolution(1.0, 1.2 * t_ms, metallicity_z=0.02)
    massive_state = public_stellar_evolution(20.0, 1.1 * _ms_lifetime(20.0, metallicity_z=0.02), metallicity_z=0.02)

    assert ms_state['phase'] == 'MS'
    assert post_ms_state['phase'] in {'AGB', 'post-AGB', 'PN', 'WD'}
    assert massive_state['phase'] in {'NS', 'SN'}

    print('stellar_tracks/public re-export: OK')
    print(f"  1Msun phases={ms_state['phase']}->{post_ms_state['phase']}")
    print(f"  20Msun phase={massive_state['phase']}")


def test_solar_anchor_state():
    state = public_stellar_evolution(1.0, 4.57, metallicity_z=Z_SUN, model='auto')
    assert abs(state['T_eff'] - 5772.0) < 120.0
    assert abs(state['luminosity'] - 1.0) < 0.15
    assert abs(state['radius'] - 1.0) < 0.08

    print('solar anchor calibration: OK')
    print(
        f"  Teff={state['T_eff']}K L={state['luminosity']:.3f} "
        f"R={state['radius']:.3f}Rsun"
    )


def test_galaxy_overview_uses_current_phase_metadata():
    assert _overview_phase_bucket('RGB') == 'giant'
    assert _overview_phase_bucket('NS') == 'dead'
    assert _overview_phase_bucket('pre-MS') == 'pre-MS'

    assert _overview_alive_until(1.0, metallicity_z=0.02) > _ms_lifetime(1.0, metallicity_z=0.02)
    assert _overview_alive_until(80.0, metallicity_z=0.02) < _ms_lifetime(80.0, metallicity_z=0.02)

    data = generate_galaxy(n_stars=8, params={'t_max': 1.0}, seed=7)
    stars = data['stars']
    for key in (
        'phase_current', 'phase_bucket', 'phase_kr_current', 'current_mass', 'spectral_current',
        'birth_radius_kpc', 'birth_guiding_radius_kpc', 'guiding_radius_kpc', 'guiding_radius_delta_kpc',
        'current_radius_kpc', 'radial_migration_delta_kpc', 'radial_churning_delta_kpc',
        'radial_blurring_delta_kpc',
        'radial_migration_mean_shift_kpc', 'radial_migration_sigma_kpc',
        'orbital_eccentricity', 'sigma_R_km_s', 'sigma_phi_km_s', 'sigma_z_km_s',
        'circular_velocity_km_s', 'angular_momentum_kpc_km_s',
        'v_R_km_s', 'v_phi_km_s', 'v_z_km_s', 'vertical_scale_height_kpc',
        'guiding_r_zone', 'current_r_zone', 'current_x', 'current_y', 'current_z',
        'has_moon_system', 'has_regular_moons', 'has_irregular_moons',
        'has_large_moon', 'has_resonant_moon_chain', 'moon_count_estimate',
    ):
        assert key in stars
        assert len(stars[key]) == 8

    for birth_r, guiding_r, current_r in zip(
        stars['birth_radius_kpc'],
        stars['guiding_radius_kpc'],
        stars['current_radius_kpc'],
    ):
        assert 0.5 <= birth_r <= 20.0
        assert 0.5 <= guiding_r <= 20.0
        assert 0.5 <= current_r <= 20.0

    for birth_rg, guiding_r, delta_rg in zip(
        stars['birth_guiding_radius_kpc'],
        stars['guiding_radius_kpc'],
        stars['guiding_radius_delta_kpc'],
    ):
        assert abs((guiding_r - birth_rg) - delta_rg) < 1e-6

    for guiding_r, current_r, delta_blur in zip(
        stars['guiding_radius_kpc'],
        stars['current_radius_kpc'],
        stars['radial_blurring_delta_kpc'],
    ):
        assert abs((current_r - guiding_r) - delta_blur) < 1e-6

    for birth_r, current_r, delta_total in zip(
        stars['birth_radius_kpc'],
        stars['current_radius_kpc'],
        stars['radial_migration_delta_kpc'],
    ):
        assert abs((current_r - birth_r) - delta_total) < 1e-6

    for ecc in stars['orbital_eccentricity']:
        assert 0.0 <= ecc <= 0.35

    for rg, vc, lz, rc, vphi in zip(
        stars['guiding_radius_kpc'],
        stars['circular_velocity_km_s'],
        stars['angular_momentum_kpc_km_s'],
        stars['current_radius_kpc'],
        stars['v_phi_km_s'],
    ):
        assert abs(lz - rg * vc) < 1e-6
        assert abs(vphi - lz / max(rc, 1e-8)) < 1e-6

    print('galaxy overview current-state metadata: OK')
    print(f"  phases={set(stars['phase_bucket'])}")


if __name__ == '__main__':
    test_properties_module()
    test_remnants_module()
    test_public_reexport_and_tracks()
    test_solar_anchor_state()
    test_galaxy_overview_uses_current_phase_metadata()
    print('stellar module checks passed')
