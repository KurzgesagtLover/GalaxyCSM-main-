"""Sulfur overabundance diagnostic script."""

import numpy as np

from gce.config import ELEMENTS, EL_IDX, SOLAR_X
from gce.planets import M_EARTH, compute_atmosphere, differentiate_full, volatile_depletion


def main():
    print("=" * 70)
    print("TEST 1: S mass budget trace (Earth-like planet, 1.0 Mearth)")
    print("=" * 70)

    bulk = {el: SOLAR_X[EL_IDX[el]] for el in ELEMENTS if el not in ('H', 'He')}
    print(f"\n[ISM] S mass fraction: {bulk['S']:.6e}")
    print(f"[ISM] S in 1 Mearth planet: {bulk['S'] * M_EARTH:.3e} kg")

    depleted = volatile_depletion(bulk, 255)
    print(f"\n[Volatile Depletion] S after (T_eq=255K): {depleted['S']:.6e}")
    print(f"[Volatile Depletion] S retention: {depleted['S'] / bulk['S']:.4f}")

    ox_iw = -2.0
    diff = differentiate_full(1.0, depleted, ox_iw, 1.0)
    s_core = diff['core'].get('S', 0)
    s_mantle = diff['mantle'].get('S', 0)
    s_crust = diff['crust'].get('S', 0)
    print("\n[Differentiation] S distribution:")
    print(f"  Core:   {s_core:.6e}  ({s_core * 100:.2f}%)")
    print(f"  Mantle: {s_mantle:.6e}  ({s_mantle * 100:.2f}%)")
    print(f"  Crust:  {s_crust:.6e}  ({s_crust * 100:.2f}%)")

    atmo = compute_atmosphere(depleted, 1.0, 'rocky', 255, 9.8, 11200, 4.5, ox_iw)
    print("\n[Atmosphere] Composition:")
    for mol, data in sorted(atmo['composition'].items(), key=lambda x: -x[1]['pct']):
        print(f"  {mol:6s}: {data['pct']:8.4f}%  (mass: {data['mass_kg']:.3e} kg)")

    so2_pct = atmo['composition'].get('SO2', {}).get('pct', 0)
    h2s_pct = atmo['composition'].get('H2S', {}).get('pct', 0)
    print(f"\n[RESULT] Total S species: {so2_pct + h2s_pct:.4f}%")
    print(f"  SO2: {so2_pct:.4f}%")
    print(f"  H2S: {h2s_pct:.4f}%")

    f_age = min(1.0, (4.5 / 4.5) ** 0.5)
    s_outgas_eff = 2.0e-5 * f_age
    s_total_outgassed = depleted['S'] * M_EARTH * s_outgas_eff
    ox_frac = np.clip((ox_iw + 3) / 6, 0, 1)
    so2_mass = s_total_outgassed * ox_frac * (64.066 / 32.065)
    h2s_mass = s_total_outgassed * (1 - ox_frac) * (34.081 / 32.065)
    print("\n[Manual Trace]")
    print(f"  f_age={f_age:.4f}, S_outgas_eff={s_outgas_eff:.6e}")
    print(f"  depleted S bulk frac = {depleted['S']:.6e}")
    print(f"  S_total outgassed = {s_total_outgassed:.3e} kg")
    print(f"  ox_frac = {ox_frac:.4f} (IW={ox_iw})")
    print(f"  SO2 mass = {so2_mass:.3e} kg")
    print(f"  H2S mass = {h2s_mass:.3e} kg")

    n2_mass = atmo['composition'].get('N2', {}).get('mass_kg', 0)
    print("\n[Earth Benchmark]")
    print(f"  N2 mass: {n2_mass:.3e} kg  (Earth: 3.87e+18 kg)")
    print(f"  SO2+H2S: {so2_mass + h2s_mass:.3e} kg  (Earth: trace)")

    print("\n" + "=" * 70)
    print("TEST 2: S = 0 test - verify rest of atmosphere is normal")
    print("=" * 70)
    bulk_no_s = dict(depleted)
    bulk_no_s['S'] = 0.0
    atmo2 = compute_atmosphere(bulk_no_s, 1.0, 'rocky', 255, 9.8, 11200, 4.5, ox_iw)
    for mol, data in sorted(atmo2['composition'].items(), key=lambda x: -x[1]['pct']):
        print(f"  {mol:6s}: {data['pct']:8.4f}%  (mass: {data['mass_kg']:.3e} kg)")

    print("\n" + "=" * 70)
    print("TEST 3: Very oxidizing planet (IW = +2) - should minimize H2S")
    print("=" * 70)
    atmo3 = compute_atmosphere(depleted, 1.0, 'rocky', 255, 9.8, 11200, 4.5, +2.0)
    so2_3 = atmo3['composition'].get('SO2', {}).get('pct', 0)
    h2s_3 = atmo3['composition'].get('H2S', {}).get('pct', 0)
    print(f"\n  SO2: {so2_3:.4f}%  |  H2S: {h2s_3:.4f}%")
    for mol, data in sorted(atmo3['composition'].items(), key=lambda x: -x[1]['pct'])[:6]:
        print(f"  {mol:6s}: {data['pct']:8.4f}%")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)


if __name__ == '__main__':
    main()
