import os
import time

from gce.config import DEFAULT_PRECISE_TRACK_PACK
from gce.stellar import hr_track, stellar_evolution
from gce.stellar_track_provider import get_provider_stats, reset_provider_stats
from gce.stellar_track_interpolator import load_track_pack


def test_demo_track_pack_exists_and_loads():
    assert os.path.exists(DEFAULT_PRECISE_TRACK_PACK)
    pack = load_track_pack(DEFAULT_PRECISE_TRACK_PACK)
    assert len(pack.mass_grid) >= 10
    assert len(pack.metallicity_grid) >= 3
    assert ('MS' in pack.phase_order) or (len(pack.phase_order) > 0)


def test_precise_grid_hits_exact_track_point_close_to_heuristic():
    reset_provider_stats()
    heuristic = stellar_evolution(1.0, 1.0, metallicity_z=0.02, model='heuristic')
    precise = stellar_evolution(1.0, 1.0, metallicity_z=0.02, model='auto')
    stats = get_provider_stats()
    assert stats['precise_hits'] >= 1
    assert precise['phase'] == heuristic['phase']
    assert abs(precise['luminosity'] - heuristic['luminosity']) / max(heuristic['luminosity'], 1e-8) < 0.05
    assert abs(precise['T_eff'] - heuristic['T_eff']) / max(float(heuristic['T_eff']), 1.0) < 0.05
    assert abs(precise['radius'] - heuristic['radius']) / max(heuristic['radius'], 1e-8) < 0.05


def test_out_of_range_precise_request_falls_back():
    reset_provider_stats()
    state = stellar_evolution(140.0, 2.0, metallicity_z=0.02, model='auto')
    stats = get_provider_stats()
    assert state['phase'] in {'MS', 'LBV', 'hypergiant', 'WR', 'SN', 'BH'}
    assert stats['fallbacks'] >= 1
    assert 'mass_out_of_range' in stats['reasons']


def test_hr_track_auto_mode_preserves_contract():
    track = hr_track(1.0, age_max=20.0, n_points=24, metallicity_z=0.02, stellar_model='auto')
    assert len(track) <= 24
    assert [pt['age'] for pt in track] == sorted(pt['age'] for pt in track)
    assert all('phase' in pt and 'phase_kr' in pt and 'color' in pt for pt in track)


def test_precise_mode_smoke_benchmark():
    t0 = time.perf_counter()
    for _ in range(200):
        stellar_evolution(1.0, 1.0, metallicity_z=0.02, model='auto')
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
