"""Atmospheric and radiation helpers for planetary models."""

import numpy as np

L_SUN = 3.828e26
AU_M = 1.496e11
M_EARTH = 5.972e24


def _planck_spectral_radiance(lam_m, T_eff):
    h = 6.62607015e-34
    c = 2.99792458e8
    k_b = 1.380649e-23
    x = np.clip(h * c / (lam_m * k_b * max(T_eff, 1.0)), 0.0, 700.0)
    denom = np.expm1(x)
    denom = np.where(denom <= 0, np.inf, denom)
    return 2.0 * h * c**2 / (lam_m**5 * denom)


def _band_energy_fraction(T_eff, lam_lo_nm, lam_hi_nm, total_lo_nm=100.0, total_hi_nm=20000.0):
    lam_total = np.geomspace(total_lo_nm, total_hi_nm, 2048) * 1e-9
    lam_band = np.geomspace(max(lam_lo_nm, total_lo_nm), min(lam_hi_nm, total_hi_nm), 512) * 1e-9
    total = np.trapezoid(_planck_spectral_radiance(lam_total, T_eff), lam_total)
    if total <= 0 or lam_band.size == 0:
        return 0.0
    band = np.trapezoid(_planck_spectral_radiance(lam_band, T_eff), lam_band)
    return float(np.clip(band / total, 0.0, 1.0))


def _photospheric_band_fractions(T_eff):
    f_uv = _band_energy_fraction(T_eff, 100.0, 400.0)
    f_vis = _band_energy_fraction(T_eff, 400.0, 700.0)
    f_ir = max(1.0 - f_uv - f_vis, 0.0)
    norm = max(f_uv + f_vis + f_ir, 1e-12)
    return f_uv / norm, f_vis / norm, f_ir / norm


def _gas_giant_atmo(mass_earth, T_eq=124, metallicity=0.0134, age_gyr=4.5):
    """Gas giant atmosphere with metallicity-dependent composition."""
    M_jup = 317.8
    m_ratio = mass_earth / M_jup
    Z_solar = 0.0134
    Z_star_rel = metallicity / Z_solar
    Z_enrich = 5.0 * Z_star_rel * m_ratio ** (-0.45)
    Z_enrich = np.clip(Z_enrich, 1.0, 100.0)

    Y_He = 0.238 + 0.01 * np.log10(max(m_ratio, 0.01))
    Y_He = np.clip(Y_He, 0.22, 0.28)
    he_h2_ratio = Y_He / (2 * (1 - Y_He))
    x_He_base = he_h2_ratio / (1 + he_h2_ratio)
    x_H2_base = 1 - x_He_base

    T_int = 100 * m_ratio ** 0.1 * max(age_gyr / 4.5, 0.01) ** (-0.3)
    T_int = np.clip(T_int, 30, 500)
    T_1bar = max(T_eq, T_int)

    x_CH4 = 1.8e-3 * Z_enrich * (1 / (1 + np.exp((T_1bar - 1200) / 150)))
    x_CO = 1.0e-4 * Z_enrich * (1 / (1 + np.exp((1200 - T_1bar) / 150)))
    x_NH3 = 2.0e-4 * Z_enrich * (1 / (1 + np.exp((T_1bar - 400) / 80)))
    x_H2O = 4.0e-4 * Z_enrich * (1 / (1 + np.exp((T_1bar - 350) / 100)))
    x_H2S = 3.0e-5 * Z_enrich * (1 / (1 + np.exp((T_1bar - 500) / 100)))
    x_PH3 = 6.0e-7 * Z_enrich

    x_metals = min(x_CH4 + x_CO + x_NH3 + x_H2O + x_H2S + x_PH3, 0.15)
    x_H2 = x_H2_base * (1 - x_metals)
    x_He = x_He_base * (1 - x_metals)
    mu = x_H2 * 2.016 + x_He * 4.003 + x_CH4 * 16.04 + x_CO * 28.01 + x_NH3 * 17.03 + x_H2O * 18.02 + x_H2S * 34.08 + x_PH3 * 34.0
    mu = max(mu, 2.0)

    g_surface = 24.8 * m_ratio ** 0.5
    k_B = 1.381e-23
    scale_H = k_B * T_1bar / (mu * 1.66e-27 * g_surface) / 1000
    rho_1bar = mu * 1.66e-27 * 101325 / (k_B * T_1bar)
    v_sound = np.sqrt(1.4 * k_B * T_1bar / (mu * 1.66e-27))

    def _mk(sp, xf):
        return {'vol_frac': round(xf, 8), 'pct': round(xf * 100, 4), 'partial_P_atm': round(xf, 8), 'mass_kg': 0}

    comp = {}
    for sp, xf in [('H2', x_H2), ('He', x_He), ('CH4', x_CH4), ('CO', x_CO), ('NH3', x_NH3), ('H2O', x_H2O), ('H2S', x_H2S), ('PH3', x_PH3)]:
        if xf > 1e-8:
            comp[sp] = _mk(sp, xf)

    return {
        'surface_pressure_atm': None,
        'surface_pressure_bar': None,
        'surface_temp_K': round(float(T_1bar), 1),
        'T_eq_K': round(float(T_eq), 1),
        'T_internal_K': round(float(T_int), 1),
        'greenhouse_K': round(float(max(T_1bar - T_eq, 0)), 1),
        'scale_height_km': round(float(scale_H), 1),
        'tropopause_km': round(float(scale_H * 2.5), 1),
        'homosphere_km': round(float(scale_H * 20), 1),
        'air_density_kg_m3': round(float(rho_1bar), 4),
        'sound_speed_m_s': round(float(v_sound), 0),
        'exosphere_temp_K': round(float(800 * m_ratio ** 0.15), 0),
        'mean_mol_mass': round(float(mu), 3),
        'Z_enrichment': round(float(Z_enrich), 1),
        'Y_He': round(float(Y_He), 4),
        'atm_mass_kg': None,
        'atm_relative_mass': None,
        'albedo_bond_eff': round(float(np.clip(0.34 + 0.1 * np.log10(max(m_ratio, 0.01)), 0.15, 0.55)), 3),
        'composition': comp,
    }


def compute_radiation_defense(mass_earth, r_earth, b_surface_uT, atm_mass_kg,
                              p_surf_bar, comp, a_AU, T_eff, L_star_Lsun, age_gyr,
                              flare_boost=1.0):
    """Compute incident band fluxes and atmospheric/magnetic attenuation."""
    S_total = (L_star_Lsun * L_SUN) / (4 * np.pi * (a_AU * AU_M) ** 2)

    f_uv_photo, f_vis_photo, f_ir_photo = _photospheric_band_fractions(T_eff)
    flare_boost = float(max(flare_boost, 1.0))
    flare_excess = flare_boost - 1.0
    flare_uv_multiplier = 1.0
    flare_xuv_multiplier = 1.0
    flare_wind_multiplier = 1.0
    if T_eff <= 3700:
        flare_uv_multiplier += 0.45 * flare_excess
        flare_xuv_multiplier += 0.65 * flare_excess
        flare_wind_multiplier += 0.30 * flare_excess
    elif T_eff <= 5200:
        flare_uv_multiplier += 0.20 * flare_excess
        flare_xuv_multiplier += 0.35 * flare_excess
        flare_wind_multiplier += 0.15 * flare_excess

    f_x_base = 1e-4 if T_eff > 4000 else 1e-3
    activity_multiplier = max((0.1 / max(age_gyr, 0.01)) ** 0.8, 1.0)
    f_x_euv = min(f_x_base * activity_multiplier * flare_xuv_multiplier, 0.2)
    photo_scale = max(1.0 - f_x_euv, 0.0)

    inc = {
        'x_euv': S_total * f_x_euv,
        'uv': S_total * photo_scale * f_uv_photo * flare_uv_multiplier,
        'vis': S_total * photo_scale * f_vis_photo,
        'ir': S_total * photo_scale * f_ir_photo,
    }

    trans = {'x_euv': 1.0, 'uv': 1.0, 'vis': 1.0, 'ir': 1.0}

    if T_eff > 5200:
        R_star_rel = 1.0
        v_wind = 4e5
    elif T_eff > 3700:
        R_star_rel = 0.7
        v_wind = 3.5e5
    else:
        R_star_rel = 0.3
        v_wind = 2.5e5

    f_x_activity = max((0.1 / max(age_gyr, 0.01)) ** 1.2, 1.0) * flare_wind_multiplier
    Mdot_rel = R_star_rel ** 2 * f_x_activity
    n_wind = 5e6 * Mdot_rel / max(a_AU, 0.01) ** 2
    P_dyn_wind = 0.5 * 1.67e-27 * n_wind * v_wind ** 2

    mu_0 = 4 * np.pi * 1e-7
    B_surface = b_surface_uT * 1e-6
    if P_dyn_wind > 0 and B_surface > 0:
        r_standoff = (B_surface ** 2 / (2 * mu_0 * P_dyn_wind)) ** (1 / 6)
        r_standoff = min(r_standoff, 50.0)
    else:
        r_standoff = 0.5 if B_surface == 0 else 10.0
    mag_shield = np.clip((r_standoff - 1.0) / 4.0, 0.0, 1.0)

    trans['x_euv'] = np.exp(-p_surf_bar * 1000) * (1 - mag_shield * 0.9)

    col_mass = atm_mass_kg / (4 * np.pi * (r_earth * 1000) ** 2)
    mass_O3 = comp.get('O3', {}).get('mass_kg', 0)
    mass_O2 = comp.get('O2', {}).get('mass_kg', 0)
    mass_CO2 = comp.get('CO2', {}).get('mass_kg', 0)
    A_planet = 4 * np.pi * (r_earth * 1000) ** 2
    N_O3 = mass_O3 / (48 * 1.66e-27) / (A_planet * 1e4) if mass_O3 > 0 else 0
    N_O2 = mass_O2 / (32 * 1.66e-27) / (A_planet * 1e4) if mass_O2 > 0 else 0
    N_CO2 = mass_CO2 / (44 * 1.66e-27) / (A_planet * 1e4) if mass_CO2 > 0 else 0

    tau_uv = N_O3 * 1.1e-17 + N_O2 * 1e-23 + N_CO2 * 1e-21
    tau_rayleigh = col_mass * 0.02
    trans['uv'] = float(np.exp(-min(tau_uv + tau_rayleigh, 300)))
    trans['vis'] = float(np.exp(-tau_rayleigh * 0.05))

    mass_H2O = comp.get('H2O', {}).get('mass_kg', 0)
    N_H2O = mass_H2O / (18 * 1.66e-27) / (A_planet * 1e4) if mass_H2O > 0 else 0
    tau_ir = N_H2O * 1e-22 + N_CO2 * 3e-21
    trans['ir'] = float(np.exp(-min(tau_ir, 300)))

    surf = {k: inc[k] * trans[k] for k in inc}
    return {
        'incident_W_m2': {k: round(v, 4) for k, v in inc.items()},
        'surface_W_m2': {k: round(v, 4) for k, v in surf.items()},
        'transmittance': {k: round(v, 6) for k, v in trans.items()},
        'magnetopause_r': round(float(r_standoff), 2),
        'mag_shielding_pct': round(float(mag_shield * 100), 1),
        'wind_pressure_nPa': round(float(P_dyn_wind * 1e9), 2),
        'flare_boost': round(float(flare_boost), 3),
        'flare_uv_multiplier': round(float(flare_uv_multiplier), 3),
        'flare_xuv_multiplier': round(float(flare_xuv_multiplier), 3),
    }


def _sub_neptune_atmo(mass_earth, bulk_comp, T_eq=200, metallicity=0.0134, age_gyr=4.5,
                      semi_major_au=1.0, star_teff=5778, luminosity_lsun=1.0):
    """Sub-Neptune atmosphere with photoevaporation and core-powered loss.

    This is still semi-analytic, but it now separates:
      1. Initial H/He envelope acquisition
      2. XUV-driven photoevaporation
      3. Core-powered mass loss from residual cooling luminosity
      4. Atmospheric regime after envelope erosion
    """
    from .photolysis import integrated_xuv_history

    z_solar = 0.0134
    z_star_rel = max(metallicity / z_solar, 0.05)
    core_radius_re = mass_earth ** 0.27
    base_radius_m = 6.371e6 * core_radius_re

    # Envelope fraction at disk dispersal: increases with core mass and metallicity.
    f_env_init = 0.012 * mass_earth ** 0.55 * z_star_rel ** 0.35
    f_env_init = float(np.clip(f_env_init, 0.002, 0.18))

    xuv_history = integrated_xuv_history(
        star_teff, age_gyr, semi_major_au, luminosity_lsun=luminosity_lsun
    )
    xuv_fluence = xuv_history['xuv_fluence_erg_cm2']
    xuv_norm = xuv_fluence / 5.5e18

    # Lower-mass, hotter planets are less able to retain primordial envelopes.
    photoevap_metric = (
        0.28
        * xuv_norm
        * (max(T_eq, 200.0) / 700.0) ** 0.6
        * (max(mass_earth, 0.5) / 5.0) ** -1.15
        * core_radius_re ** 0.4
    )
    corepowered_metric = (
        0.18
        * np.clip((T_eq - 350.0) / 450.0, 0.0, None) ** 1.6
        * max(age_gyr, 0.05) ** 0.35
        * (max(mass_earth, 0.5) / 5.0) ** -0.7
    )

    total_loss_metric = max(photoevap_metric + corepowered_metric, 0.0)
    f_env = f_env_init * np.exp(-total_loss_metric)
    f_env = float(np.clip(f_env, 1e-5, f_env_init))

    if f_env < 3e-4:
        envelope_regime = 'secondary_atmosphere'
    elif f_env < 3e-3:
        envelope_regime = 'transition'
    else:
        envelope_regime = 'retained_envelope'
    stripped = envelope_regime == 'secondary_atmosphere'

    c_rel = np.clip(bulk_comp.get('C', 1.0e-3) / 1.0e-3, 0.2, 5.0)
    n_rel = np.clip(bulk_comp.get('N', 2.0e-4) / 2.0e-4, 0.2, 5.0)
    h_rel = np.clip(bulk_comp.get('H', 2.0e-4) / 2.0e-4, 0.2, 5.0)

    if envelope_regime == 'secondary_atmosphere':
        water_cool = 1.0 / (1.0 + np.exp((T_eq - 650.0) / 90.0))
        methane_cool = 1.0 / (1.0 + np.exp((T_eq - 750.0) / 80.0))
        x_h2o = 0.08 + 0.25 * h_rel * water_cool
        x_co2 = 0.10 + 0.18 * c_rel * (1.0 - 0.4 * methane_cool)
        x_n2 = 0.25 + 0.20 * n_rel
        x_ch4 = 0.01 + 0.05 * c_rel * methane_cool
        x_h2 = 0.01
        x_he = 0.0
        mix = {
            'N2': x_n2, 'CO2': x_co2, 'H2O': x_h2o, 'CH4': x_ch4, 'H2': x_h2,
        }
        p_surf = np.clip(2.0 + 18.0 * mass_earth ** 0.35, 0.5, 120.0)
        radius_factor = 1.03
        albedo = 0.24
    elif envelope_regime == 'transition':
        z_enrich = np.clip(25.0 * z_star_rel * mass_earth ** -0.25, 5.0, 150.0)
        x_h2o = 0.03 * z_enrich / 20.0
        x_ch4 = 0.015 * z_enrich / 20.0 * (1.0 / (1.0 + np.exp((T_eq - 850.0) / 120.0)))
        x_nh3 = 0.01 * z_enrich / 20.0 * (1.0 / (1.0 + np.exp((T_eq - 500.0) / 90.0)))
        x_co2 = 0.008 * z_enrich / 20.0
        x_metals = min(x_h2o + x_ch4 + x_nh3 + x_co2, 0.35)
        x_h2 = 0.72 * (1.0 - x_metals)
        x_he = 0.28 * (1.0 - x_metals)
        mix = {
            'H2': x_h2, 'He': x_he, 'H2O': x_h2o, 'CH4': x_ch4, 'NH3': x_nh3, 'CO2': x_co2,
        }
        p_surf = np.clip(12.0 * (f_env / 3e-3) ** 0.7 * mass_earth ** 0.45, 2.0, 1500.0)
        radius_factor = 1.10 + 0.15 * (f_env / 3e-3) ** 0.25
        albedo = 0.28
    else:
        z_enrich = np.clip(18.0 * z_star_rel * mass_earth ** -0.22 * (1.0 + 0.08 * photoevap_metric), 3.0, 250.0)
        x_h2o = 0.02 * z_enrich / 20.0
        x_ch4 = 0.02 * z_enrich / 20.0 * (1.0 / (1.0 + np.exp((T_eq - 900.0) / 120.0)))
        x_nh3 = 0.008 * z_enrich / 20.0 * (1.0 / (1.0 + np.exp((T_eq - 450.0) / 80.0)))
        x_co2 = 0.003 * z_enrich / 20.0
        x_co = 0.002 * z_enrich / 20.0 * (1.0 / (1.0 + np.exp((700.0 - T_eq) / 120.0)))
        x_metals = min(x_h2o + x_ch4 + x_nh3 + x_co2 + x_co, 0.28)
        x_h2 = 0.74 * (1.0 - x_metals)
        x_he = 0.26 * (1.0 - x_metals)
        mix = {
            'H2': x_h2, 'He': x_he, 'H2O': x_h2o, 'CH4': x_ch4,
            'NH3': x_nh3, 'CO2': x_co2, 'CO': x_co,
        }
        p_surf = np.clip(25.0 * (f_env / 0.003) ** 0.9 * mass_earth ** 0.45, 5.0, 8000.0)
        radius_factor = 1.12 + 0.65 * (f_env / 0.03) ** 0.45
        albedo = np.clip(0.30 + 0.08 * f_env / 0.03, 0.25, 0.55)

    total_x = max(sum(mix.values()), 1e-12)
    mix = {sp: xf / total_x for sp, xf in mix.items() if xf > 1e-8}

    mol_weights = {
        'H2': 2.016, 'He': 4.003, 'CH4': 16.04, 'H2O': 18.02,
        'NH3': 17.03, 'CO2': 44.01, 'CO': 28.01, 'N2': 28.014,
    }
    mu = sum(mix[sp] * mol_weights[sp] for sp in mix)
    mu = max(mu, 2.0)

    radius_factor = float(np.clip(radius_factor, 1.0, 2.5))
    r_planet = base_radius_m * radius_factor
    r_planet_re = r_planet / 6.371e6
    g_surface = 9.81 * mass_earth / max(r_planet_re ** 2, 1e-4)

    if envelope_regime == 'secondary_atmosphere':
        atm_mass = p_surf * 101325 * 4 * np.pi * r_planet**2 / max(g_surface, 1e-6)
    else:
        atm_mass = f_env * mass_earth * M_EARTH

    greenhouse = T_eq * (0.05 * np.log1p(p_surf) + 0.015 * np.log1p(max(mu - 2.0, 0.0)))
    if envelope_regime == 'secondary_atmosphere':
        greenhouse += 8.0 * np.log1p(p_surf)
    surface_temp = T_eq + greenhouse

    k_b = 1.381e-23
    scale_h = k_b * surface_temp / (mu * 1.66e-27 * max(g_surface, 1e-6)) / 1000.0
    rho = mu * 1.66e-27 * p_surf * 101325 / (k_b * max(surface_temp, 1.0))
    v_sound = np.sqrt(1.35 * k_b * max(surface_temp, 1.0) / (mu * 1.66e-27))

    comp = {}
    for sp, xf in mix.items():
        mass_frac = xf * mol_weights[sp] / mu
        mass_kg = atm_mass * mass_frac
        comp[sp] = {
            'vol_frac': round(float(xf), 8),
            'pct': round(float(xf * 100.0), 4),
            'partial_P_atm': round(float(xf * p_surf), 4),
            'mass_kg': round(float(mass_kg), 2),
        }

    return {
        'surface_pressure_atm': round(float(p_surf), 3),
        'surface_pressure_bar': round(float(p_surf), 3),
        'surface_temp_K': round(float(surface_temp), 1),
        'T_eq_K': round(float(T_eq), 1),
        'greenhouse_K': round(float(greenhouse), 1),
        'f_envelope_init': round(float(f_env_init), 6),
        'f_envelope': round(float(f_env), 6),
        'envelope_stripped': stripped,
        'envelope_regime': envelope_regime,
        'photoevap_loss_metric': round(float(photoevap_metric), 4),
        'corepowered_loss_metric': round(float(corepowered_metric), 4),
        'scale_height_km': round(float(scale_h), 1),
        'tropopause_km': round(float(scale_h * 3.0), 1),
        'homosphere_km': round(float(scale_h * 15.0), 1),
        'air_density_kg_m3': round(float(rho), 4),
        'sound_speed_m_s': round(float(v_sound), 0),
        'exosphere_temp_K': round(float(min(4000.0, 700.0 + 120.0 * np.log1p(xuv_norm) + 0.4 * T_eq)), 0),
        'mean_mol_mass': round(float(mu), 3),
        'atm_mass_kg': float(f'{atm_mass:.3e}'),
        'atm_relative_mass': float(f'{atm_mass / (mass_earth * M_EARTH):.3e}'),
        'albedo_bond_eff': round(float(albedo), 3),
        'xuv_history': {
            'spec_class': xuv_history['spec_class'],
            't_sat_gyr': round(float(xuv_history['t_sat_gyr']), 4),
            'xuv_fluence_erg_cm2': float(f"{xuv_fluence:.3e}"),
        },
        'composition': comp,
    }
