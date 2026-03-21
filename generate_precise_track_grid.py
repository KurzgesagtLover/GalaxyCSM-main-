"""Generate a bundled demo stellar track pack for precise interpolation."""

from __future__ import annotations

import json
import os

import numpy as np

from gce.config import DEFAULT_PRECISE_TRACK_PACK
from gce.stellar import _ms_lifetime
from gce.stellar_properties import PHASE_KR
from gce.stellar_tracks import stellar_evolution


MASS_GRID = [0.12, 0.25, 0.4, 0.7, 1.0, 1.5, 3.0, 6.0, 8.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0]
METALLICITY_GRID = [1e-4, 0.004, 0.02, 0.04]


def build_ages(mass, metallicity_z):
    t_ms = _ms_lifetime(mass, metallicity_z=metallicity_z)
    total = min(max(t_ms * 1.55 + 0.5, t_ms * 1.4), 5000.0)
    pre_end = min(0.002, t_ms * 0.05)

    ages = set()
    ages.update(np.linspace(0.0, pre_end, 18))
    ages.update(np.linspace(pre_end, t_ms * 0.5, 22))
    ages.update(np.linspace(t_ms * 0.5, t_ms * 0.85, 22))
    ages.update(np.linspace(t_ms * 0.85, t_ms * 0.95, 18))
    ages.update(np.linspace(t_ms * 0.95, t_ms * 1.05, 24))
    ages.update(np.linspace(t_ms * 1.05, t_ms * 1.15, 22))
    ages.update(np.linspace(t_ms * 1.15, t_ms * 1.25, 20))
    ages.update(np.linspace(t_ms * 1.25, t_ms * 1.4, 18))
    if total > t_ms * 1.4:
        ages.update(np.linspace(t_ms * 1.4, total, 18))
    ages = sorted(float(age) for age in ages if 0.0 <= age <= total)
    return ages


def compute_phase_fraction(phases, ages):
    fractions = [0.0] * len(phases)
    start = 0
    while start < len(phases):
        end = start
        while end + 1 < len(phases) and phases[end + 1] == phases[start]:
            end += 1
        if end == start:
            fractions[start] = 0.0
        else:
            age_start = ages[start]
            age_end = ages[end]
            span = age_end - age_start
            for idx in range(start, end + 1):
                if span <= 0:
                    fractions[idx] = round((idx - start) / max(end - start, 1), 6)
                else:
                    fractions[idx] = round((ages[idx] - age_start) / span, 6)
        start = end + 1
    return fractions


def build_track(initial_mass, metallicity_z):
    ages = build_ages(initial_mass, metallicity_z)
    phases = []
    luminosity = []
    temperature = []
    radius_rsun = []
    current_mass = []
    max_radius_rsun = []
    flare_activity = []

    for age in ages:
        evo = stellar_evolution(
            initial_mass,
            age,
            metallicity_z=metallicity_z,
            model='heuristic',
        )
        phases.append(evo['phase'])
        luminosity.append(round(float(evo['luminosity']), 8))
        temperature.append(round(float(evo['T_eff']), 5))
        radius_rsun.append(round(float(evo['radius']), 8))
        current_mass.append(round(float(evo['current_mass']), 8))
        max_radius_rsun.append(round(float(evo['max_radius_au']) / 0.00465, 8))
        flare_activity.append(round(float(evo.get('flare_activity', 0.0)), 6))

    return {
        'initial_mass': initial_mass,
        'metallicity_z': metallicity_z,
        'ages_gyr': [round(float(age), 8) for age in ages],
        'phases': phases,
        'phase_fraction': compute_phase_fraction(phases, ages),
        'luminosity': luminosity,
        'T_eff': temperature,
        'radius_rsun': radius_rsun,
        'current_mass': current_mass,
        'max_radius_rsun': max_radius_rsun,
        'flare_activity': flare_activity,
    }


def build_payload():
    tracks = []
    for metallicity_z in METALLICITY_GRID:
        for initial_mass in MASS_GRID:
            tracks.append(build_track(initial_mass, metallicity_z))
    return {
        'schema_version': 1,
        'coordinate_system': 'phase_fraction_v1',
        'source': 'bundled_demo_grid_generated_from_heuristic_baseline',
        'generated_by': 'generate_precise_track_grid.py',
        'phase_order': list(PHASE_KR.keys()),
        'tracks': tracks,
    }


def main():
    payload = build_payload()
    os.makedirs(os.path.dirname(DEFAULT_PRECISE_TRACK_PACK), exist_ok=True)
    with open(DEFAULT_PRECISE_TRACK_PACK, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(',', ':'))
    print(f'Wrote {DEFAULT_PRECISE_TRACK_PACK}')
    print(f"tracks={len(payload['tracks'])}")


if __name__ == '__main__':
    main()
