"""Micro-benchmark for heuristic vs precise stellar track engines."""

from __future__ import annotations

import time

import numpy as np

from gce.stellar import stellar_evolution
from gce.stellar_track_provider import get_provider_stats, reset_provider_stats


def benchmark(model, masses, ages, metallicities, repeats=1):
    reset_provider_stats()
    t0 = time.perf_counter()
    sink = 0.0
    for _ in range(repeats):
        for mass, age, metallicity in zip(masses, ages, metallicities):
            state = stellar_evolution(mass, age, metallicity_z=metallicity, model=model)
            sink += state['luminosity'] + state['radius']
    elapsed = time.perf_counter() - t0
    stats = get_provider_stats()
    return elapsed, sink, stats


def main():
    rng = np.random.default_rng(7)
    n = 2500
    masses = rng.uniform(0.12, 80.0, size=n)
    metallicities = rng.choice([1e-4, 0.004, 0.02, 0.04], size=n)
    ages = rng.uniform(0.0, 14.0, size=n)

    heuristic_elapsed, _, _ = benchmark('heuristic', masses, ages, metallicities, repeats=2)
    precise_elapsed, _, precise_stats = benchmark('auto', masses, ages, metallicities, repeats=2)

    print(f'heuristic_elapsed_s={heuristic_elapsed:.4f}')
    print(f'auto_elapsed_s={precise_elapsed:.4f}')
    if heuristic_elapsed > 0:
        print(f'slowdown_factor={precise_elapsed / heuristic_elapsed:.2f}')
    print(f'provider_stats={precise_stats}')


if __name__ == '__main__':
    main()
