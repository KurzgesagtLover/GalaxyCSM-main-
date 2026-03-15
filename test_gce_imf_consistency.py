import numpy as np

from gce.config import EL_IDX
from gce.physics import (
    imf_number_fraction,
    imf_number_per_msun,
    imf_return_fraction,
    sample_kroupa_imf,
)
from gce.solver import GCESolver
from gce.stellar import _ms_lifetime, _sample_popiii_imf, hr_track, stellar_evolution
from gce.yields import agb_yields


def assert_close(name, actual, expected, rel=5e-2, abs_tol=0.0):
    if not np.isclose(actual, expected, rtol=rel, atol=abs_tol):
        raise AssertionError(f"{name}: got {actual}, expected {expected}")


def check_kroupa_sampler():
    rng = np.random.default_rng(1234)
    masses = sample_kroupa_imf(rng, 200000, m_lo=0.08, m_hi=100.0)
    bins = [
        (0.08, 0.5, 'low-mass'),
        (0.5, 8.0, 'intermediate-mass'),
        (8.0, 100.0, 'massive'),
    ]
    for lo, hi, label in bins:
        empirical = float(np.mean((masses >= lo) & (masses < hi)))
        expected = imf_number_fraction(lo, hi)
        assert_close(f'kroupa {label}', empirical, expected, rel=0.03, abs_tol=2e-3)
    print('kroupa sampler segments: OK')


def check_solver_imf_stats():
    solver = GCESolver({'t_max': 0.2, 'r_max': 1.5, 'dr': 1.0})
    stats = solver.imf_stats
    assert_close('ccsn_number_per_msun', stats['ccsn_number_per_msun'], imf_number_per_msun(8.0, 100.0), rel=1e-8)
    assert_close('ccsn_return_fraction', stats['ccsn_return_fraction'], imf_return_fraction(8.0, 100.0), rel=5e-5)
    if not stats['agb_number_per_msun'] > stats['ccsn_number_per_msun']:
        raise AssertionError('AGB progenitors should outnumber CCSN progenitors in a Kroupa IMF')

    result = solver.solve()
    peak_rate = float(np.max(np.array(result['sn2_rate_physical'])))
    peak_expected = float(np.max(np.array(result['sfr'])) * stats['ccsn_number_per_msun'])
    assert_close('physical sn2 peak rate', peak_rate, peak_expected, rel=1e-10)
    print('solver IMF stats: OK')


def check_popiii_sampler():
    rng = np.random.default_rng(77)
    popiii = _sample_popiii_imf(rng, 100000)
    normal = sample_kroupa_imf(np.random.default_rng(77), 100000, m_lo=0.08, m_hi=100.0)
    if float(np.min(popiii)) < 10.0 or float(np.max(popiii)) > 150.0:
        raise AssertionError('Pop III sampler escaped configured mass bounds')
    if not float(np.mean(popiii)) > float(np.mean(normal)) * 8.0:
        raise AssertionError('Pop III sampler is not sufficiently top-heavy relative to the default IMF')
    print('Pop III IMF override: OK')


def check_metallicity_lifetime():
    tau_solar = _ms_lifetime(1.0, metallicity_z=0.02)
    tau_poor = _ms_lifetime(1.0, metallicity_z=1e-4)
    if not tau_poor > tau_solar:
        raise AssertionError('Metal-poor lifetime should be longer in the adopted fit')
    print('metallicity-dependent lifetime: OK')


def check_metallicity_track_scaling():
    evo_solar = stellar_evolution(1.0, 1.0, metallicity_z=0.02)
    evo_poor = stellar_evolution(1.0, 1.0, metallicity_z=1e-4)
    if not evo_poor['T_eff'] > evo_solar['T_eff']:
        raise AssertionError('Metal-poor MS track should be hotter at fixed mass')
    if not evo_poor['radius'] < evo_solar['radius']:
        raise AssertionError('Metal-poor MS track should be more compact at fixed mass')

    track_solar = hr_track(1.0, age_max=4.0, metallicity_z=0.02)
    track_poor = hr_track(1.0, age_max=4.0, metallicity_z=1e-4)
    if not track_poor[10]['T_eff'] > track_solar[10]['T_eff']:
        raise AssertionError('Metal-poor H-R track should stay hotter along the early MS')
    print('metallicity-dependent H-R track: OK')


def check_massive_star_winds():
    tau_rich = _ms_lifetime(40.0, metallicity_z=0.02)
    tau_poor = _ms_lifetime(40.0, metallicity_z=1e-4)
    evo_rich = stellar_evolution(40.0, 0.84 * tau_rich, metallicity_z=0.02)
    evo_poor = stellar_evolution(40.0, 0.84 * tau_poor, metallicity_z=1e-4)
    if not evo_rich['current_mass'] < evo_poor['current_mass']:
        raise AssertionError('Metal-rich massive stars should lose more mass to winds')
    print('massive-star wind scaling: OK')


def check_hr_track_point_cap():
    track = hr_track(1.0, age_max=20.0, n_points=24, metallicity_z=0.02)
    if len(track) > 24:
        raise AssertionError('hr_track should honor the requested n_points cap')
    ages = [pt['age'] for pt in track]
    if ages != sorted(ages):
        raise AssertionError('hr_track ages should stay monotonic after resampling')
    print('H-R track point cap: OK')


def check_agb_source_state():
    solver = GCESolver({'t_max': 2.0, 'r_max': 1.5, 'dr': 1.0})
    it = int(np.searchsorted(solver.t, 1.25))
    sfr_hist = np.linspace(0.6, 1.4, solver.nt)
    Z_hist = np.linspace(1e-4, 0.02, solver.nt)
    state = solver._agb_source_state(2.5, solver.t[it], it, sfr_hist, Z_hist)
    if state is None:
        raise AssertionError('Expected an AGB source state for a 2.5 Msun progenitor')
    expected_z = float(np.interp(state['t_form'], solver.t[:it+1], Z_hist[:it+1]))
    expected_sfr = float(np.interp(state['t_form'], solver.t[:it+1], sfr_hist[:it+1]))
    assert_close('AGB source metallicity', state['Z_form'], expected_z, rel=1e-8, abs_tol=1e-10)
    assert_close('AGB source SFR', state['sfr_form'], expected_sfr, rel=1e-8, abs_tol=1e-10)
    assert_close('AGB lifetime closure', state['tau'], _ms_lifetime(2.5, metallicity_z=state['Z_form']), rel=2e-2)
    print('AGB birth-state interpolation: OK')


def check_agb_yield_surface():
    z_solar = 0.0134
    ba_lo = agb_yields(1.49, z_solar)[EL_IDX['Ba']]
    ba_hi = agb_yields(1.51, z_solar)[EL_IDX['Ba']]
    mo_lo = agb_yields(3.99, z_solar)[EL_IDX['Mo']]
    mo_hi = agb_yields(4.01, z_solar)[EL_IDX['Mo']]
    if max(ba_lo, ba_hi) > 0 and abs(ba_hi - ba_lo) / max(ba_lo, ba_hi) > 0.2:
        raise AssertionError('AGB Ba yields still jump too sharply around 1.5 Msun')
    if max(mo_lo, mo_hi) > 0 and abs(mo_hi - mo_lo) / max(mo_lo, mo_hi) > 0.2:
        raise AssertionError('AGB Mo yields still jump too sharply around 4 Msun')

    ba_subsolar = agb_yields(2.5, z_solar * 0.3)[EL_IDX['Ba']]
    ba_solar = agb_yields(2.5, z_solar)[EL_IDX['Ba']]
    ba_supersolar = agb_yields(2.5, z_solar * 2.0)[EL_IDX['Ba']]
    if not ba_subsolar > ba_solar > ba_supersolar:
        raise AssertionError('Intermediate-mass AGB heavy-s yields should peak at subsolar metallicity')

    n_low_z = agb_yields(6.0, z_solar * 0.1)[EL_IDX['N']]
    n_high_z = agb_yields(6.0, z_solar * 2.0)[EL_IDX['N']]
    if not n_high_z > n_low_z:
        raise AssertionError('Hot-bottom-burning nitrogen should increase toward higher metallicity')
    print('smooth AGB yield surface: OK')


if __name__ == '__main__':
    check_kroupa_sampler()
    check_solver_imf_stats()
    check_popiii_sampler()
    check_metallicity_lifetime()
    check_metallicity_track_scaling()
    check_massive_star_winds()
    check_hr_track_point_cap()
    check_agb_source_state()
    check_agb_yield_surface()
    print('GCE/IMF consistency checks passed')
