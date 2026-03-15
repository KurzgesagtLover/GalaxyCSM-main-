from gce.planets import (
    compute_atmosphere,
    compute_physical_properties,
    core_thermal_model,
    magnetic_field,
    volatile_depletion,
)


def main():
    comp = {
        'C': 0.003,
        'N': 0.001,
        'S': 0.005,
        'Fe': 0.06,
        'Si': 0.07,
        'O': 0.01,
        'Mg': 0.01,
        'Al': 0.005,
        'H': 0.01,
    }

    dep = volatile_depletion(comp, 255)
    print("=== Volatile Depletion (T_eq=255K, Earth-like) ===")
    for el in ['C', 'N', 'S', 'H', 'O', 'Fe']:
        retained = dep.get(el, 0) / max(comp.get(el, 1e-9), 1e-9)
        print(f"  {el}: {dep.get(el, 0):.6f}  (ret: {retained:.3f})")

    phys = compute_physical_properties(1.0, 'rocky', 1.0, 1.0, 24.0, 23.5, 0.0167, 4.5, 1.0)
    atm = compute_atmosphere(dep, 1.0, 'rocky', 255, 9.82, 11186, 4.5, -1.5)
    thermal = core_thermal_model(1.0, 0.325, dep, 4.5)
    mag = magnetic_field(1.0, 0.325, thermal['T_core'], 24.0, thermal['q_cmb'], thermal['core_liquid'])

    print("\n=== Atmosphere (Earth-like, T_eq=255K) ===")
    for k, v in atm.items():
        if k != 'composition':
            print(f"  {k}: {v}")
    print("\n  Composition:")
    for mol, data in atm['composition'].items():
        print(f"    {mol}: {data['pct']:.4f}%")

    print("\n=== Derived Physics ===")
    print(f"  Radius: {phys['R_mean_Re']:.3f} Re")
    print(f"  Density: {phys['density_kg_m3']:.1f} kg/m^3")
    print(f"  Core T: {thermal['T_core']:.1f} K")
    print(f"  Surface B: {mag['B_surface_uT']:.2f} uT")
    print("\nDone")


if __name__ == '__main__':
    main()
