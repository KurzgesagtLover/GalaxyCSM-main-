import numpy as np
from gce.solver import GCESolver

solver = GCESolver({'t_max': 13.8, 'r_min': 0.5, 'r_max': 15.5, 'dr': 1.0})
res = solver.solve()
r = np.array(res['radius'], dtype=float)
Z = np.array(res['metallicity'], dtype=float)
feh = np.array(res['XH']['Fe'], dtype=float)
final_Z = Z[:, -1]
final_feh = feh[:, -1]
mask = (r >= 4.0) & (r <= 12.0)
coef = np.polyfit(r[mask], final_feh[mask], 1)
print('radii', r.tolist())
print('final_Z', np.round(final_Z, 5).tolist())
print('final_feh', np.round(final_feh, 3).tolist())
print('center_outer_delta_feh', round(float(final_feh[0] - final_feh[-1]), 3))
print('slope_dex_per_kpc', round(float(coef[0]), 4))
print('monotonic_nonincreasing', bool(np.all(np.diff(final_feh) <= 0.02)))
