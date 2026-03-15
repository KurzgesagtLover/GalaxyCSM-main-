"""Interior and dynamo helpers for planetary models."""

import numpy as np

from .config import EL_IDX, SOLAR_X

M_EARTH = 5.972e24
R_EARTH = 6.371e6
R_GAS = 8.314

RADIO_ISOTOPES = {
    'U238': (4.468, 9.46e-5, 'Fe', 0.020),
    'U235': (0.704, 5.69e-4, 'Fe', 0.0065),
    'Th232': (14.05, 2.64e-5, 'Fe', 0.079),
    'K40': (1.250, 3.48e-9, 'Fe', 240.0),
}


def core_thermal_model(planet_mass, core_frac, bulk_comp, age_gyr, cosmic_time=13.8):
    """Core temperature, radiogenic heating, and secular cooling."""
    M = planet_mass * M_EARTH
    R = R_EARTH * planet_mass ** 0.27
    R_core = R * core_frac ** (1 / 3)
    M_core = core_frac * M

    T_initial = 6500 * planet_mass ** 0.3
    tau_cool = 8.0 * planet_mass ** 0.4

    fe_frac = bulk_comp.get('Fe', 0.06)
    fe_solar = SOLAR_X[EL_IDX['Fe']]
    metal_ratio = fe_frac / max(fe_solar, 1e-12)

    H_total, H_initial = 0.0, 0.0
    for _, (half_life, heat_prod, _, conc_ppm) in RADIO_ISOTOPES.items():
        lam = np.log(2) / half_life
        conc = conc_ppm * 1e-6 * metal_ratio
        H_total += heat_prod * conc * np.exp(-lam * age_gyr)
        H_initial += heat_prod * conc

    Cp = 800
    T_radio = min((H_initial * M) / (max(M_core, 1) * Cp) * tau_cool * 1e9 * 3.156e7, 2000)
    T_secular = T_initial * np.exp(-age_gyr / tau_cool)
    T_radio_now = T_radio * (1 - np.exp(-age_gyr / tau_cool))
    if age_gyr > tau_cool * 3:
        T_secular = T_initial * (age_gyr / tau_cool) ** (-1.5)

    univ_cmb_temp = 2.725 * (13.8 / max(cosmic_time, 0.01)) ** (1 / 3)
    T_core = max(T_secular + T_radio_now, univ_cmb_temp)

    P_center_Pa = 360e9 * planet_mass ** 0.75
    P_cmb_Pa = P_center_Pa * (1 - core_frac ** (2 / 3))
    gamma_gruneisen = 1.5
    K_T = 1300e9 * planet_mass ** 0.2
    adiabat_ratio = np.exp(-gamma_gruneisen * (P_center_Pa - P_cmb_Pa) / K_T)
    adiabat_ratio = np.clip(adiabat_ratio, 0.55, 0.90)
    T_cmb = T_core * adiabat_ratio

    k_core = 40
    q_cmb = k_core * (T_core - T_cmb) / (0.1 * max(R_core, 1e3))
    Q_cmb = q_cmb * 4 * np.pi * R_core ** 2
    P_radio = H_total * M

    P_core_Pa = 340e9 * planet_mass ** 0.65
    T_solidus = 5000 * (P_core_Pa / 340e9) ** 0.5
    inner_core_frac = min((1 - T_core / T_solidus) ** 0.5, 0.9) if T_core < T_solidus else 0.0
    core_liquid = T_core > T_solidus * 0.85

    return {
        'T_core': round(float(T_core), 0),
        'T_cmb': round(float(T_cmb), 0),
        'T_solidus': round(float(T_solidus), 0),
        'q_cmb': round(float(q_cmb), 4),
        'Q_cmb_TW': round(float(Q_cmb * 1e-12), 4),
        'P_radio_TW': round(float(P_radio * 1e-12), 6),
        'inner_core_frac': round(float(inner_core_frac), 3),
        'core_liquid': bool(core_liquid),
        'fully_solid': bool(T_core < T_solidus * 0.5),
        'R_core_km': round(float(R_core * 1e-3), 1),
        'R_planet_km': round(float(R * 1e-3), 1),
    }


def core_viscosity(T_core, P_core_GPa):
    """Iron core viscosity (liquid vs solid)."""
    T_solidus = 5000 * (P_core_GPa / 340) ** 0.5
    if T_core > T_solidus:
        eta_0 = 1e-3
        E_a = 40e3
        eta = eta_0 * np.exp(E_a / (R_GAS * max(T_core, 1000)))
        return round(float(eta), 6), 'liquid'
    eta_solid = max(1e18 * np.exp(-0.001 * (T_core - 1000)), 1e15)
    return round(float(min(eta_solid, 1e25)), 2), 'solid'


def magnetic_field(planet_mass, core_frac, T_core, rotation_period_hr=24.0,
                   q_cmb=0.0, core_liquid=True):
    """Surface magnetic field via dynamo scaling."""
    R = R_EARTH * planet_mass ** 0.27
    R_core = R * core_frac ** (1 / 3)
    M_core = core_frac * planet_mass * M_EARTH
    rho_core = M_core / ((4 / 3) * np.pi * max(R_core, 1) ** 3)

    q_ad = 0.015 * planet_mass ** 0.2
    dynamo_active = core_liquid and (q_cmb > q_ad)
    if not dynamo_active:
        return {
            'B_surface_uT': 0.0,
            'dynamo_active': False,
            'field_type': '잔류자기장' if not core_liquid else '대류 부족',
        }

    q_conv = max(q_cmb - q_ad, 1e-6)
    mu_0 = 4 * np.pi * 1e-7
    B_core = 1e-3 * (mu_0 * rho_core) ** 0.5 * (q_conv * R_core / rho_core) ** (1 / 3)
    geom = (R_core / max(R, 1)) ** 3
    B_surface = B_core * geom * 0.3
    B_surface *= (24.0 / max(rotation_period_hr, 1)) ** 0.15
    B_uT = max(B_surface * 1e6, 0)

    if B_uT > 100:
        ftype = '강한 자기장'
    elif B_uT > 10:
        ftype = '중간 자기장'
    elif B_uT > 1:
        ftype = '약한 자기장'
    else:
        ftype = '극미약 자기장'

    return {
        'B_surface_uT': round(float(B_uT), 2),
        'dynamo_active': True,
        'field_type': ftype,
    }
