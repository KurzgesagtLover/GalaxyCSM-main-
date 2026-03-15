"""
Protoplanetary Disk and Asteroid Belt Formation Model.

References:
  - Hayashi (1981), PThPS 70, 35              — MMSN surface density Σ₀=1700 g/cm², slope -1.5
  - Andrews et al. (2013), ApJ 771             — Disk mass–star mass: M_disk ∝ 0.1 M_star
  - Haisch, Lada & Lada (2001), ApJL 553       — Disk lifetime median ~3 Myr
  - Williams & Cieza (2011), ARAA 49           — Dust-to-gas ratio ~0.01 at solar Z
  - Armitage (2010), Astrophysics of PF        — Disk evolution theory
  - Raymond & Izidoro (2017), Sci. Adv. 3      — Empty primordial belt, implantation model
  - Walsh et al. (2011), Nature 475            — Grand Tack model, giant migration
  - Morbidelli et al. (2012), ARAA 50          — Terrestrial planet formation, resonance sweep
  - Johansen et al. (2014), PPVI               — Streaming instability, planetesimal efficiency
  - Birnstiel et al. (2012), A&A 539           — Dust evolution, pressure traps
  - Nesvorný (2018), ARAA 56                   — Nice model, late instability
"""
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================
AU_CM    = 1.496e13       # cm
M_SUN_G  = 1.989e33       # g
M_EARTH_G = 5.972e27      # g
M_EARTH  = 5.972e24       # kg
L_SUN    = 3.828e26       # W
Z_SUN    = 0.0134         # Asplund+2009 total metallicity

# Hayashi (1981) MMSN normalization
SIGMA_0_MMSN = 1700.0     # g/cm² at 1 AU for 0.01 M☉ disk

# Snow line ice enhancement factor (Hayashi 1981)
SNOW_ENHANCEMENT = 4.2     # solid surface density jumps ~4.2× beyond snow line

# Disk lifetime constants (Haisch+2001)
DISK_LIFETIME_MEDIAN = 3.0  # Myr

# Planetesimal formation efficiency range (Johansen+2014)
# Only a small fraction of the local solid mass forms planetesimals;
# the rest is accreted by the growing planets or lost to drift.
EPSILON_PF_MIN = 0.001
EPSILON_PF_MAX = 0.01

# Pressure trap enhancement range (Birnstiel+2012)
PTRAP_MIN = 1.0
PTRAP_MAX = 3.0


# ============================================================
# PROTOPLANETARY DISK
# ============================================================
class ProtoplanetaryDisk:
    """Protoplanetary disk model with physically motivated parameters.

    All disk properties are derived from the host star mass, local
    metallicity, galactic environment (FUV), and stochastic scatter
    calibrated to observations.

    References:
        Andrews+2013 (disk mass), Hayashi 1981 (MMSN),
        Haisch+2001 (lifetime), Williams & Cieza 2011 (dust-to-gas).
    """

    def __init__(self, star_mass, metallicity_z, r_galactic_kpc=8.0, rng=None):
        """
        Parameters
        ----------
        star_mass : float
            Stellar mass in solar masses.
        metallicity_z : float
            Total metallicity mass fraction from GCE.
        r_galactic_kpc : float
            Galactocentric radius in kpc (for FUV estimation).
        rng : np.random.Generator or None
        """
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.star_mass = star_mass
        self.metallicity_z = metallicity_z
        self.disk_mass_g = 0.0

        # --- Disk total mass (Andrews+2013) ---
        # M_disk ≈ 0.1 × M_star with ~0.5 dex log-normal scatter
        log_scatter = rng.normal(0, 0.5)
        self.disk_mass = 0.1 * star_mass * 10**log_scatter
        self.disk_mass = np.clip(self.disk_mass, 1e-4, 0.5 * star_mass)
        self.disk_mass_g = self.disk_mass * M_SUN_G

        # --- Dust-to-gas ratio (Armitage 2010, Williams & Cieza 2011) ---
        # Solar: f_dg ≈ 0.01; scales linearly with metallicity
        self.dust_to_gas = 0.01 * metallicity_z / max(Z_SUN, 1e-8)
        self.dust_to_gas = np.clip(self.dust_to_gas, 1e-5, 0.1)

        # --- Disk outer radius (Andrews+2013) ---
        # R_disk ≈ 30 AU × (M_star)^0.5, with scatter
        r_scatter = 10**rng.normal(0, 0.2)
        self.r_disk_au = 30.0 * star_mass**0.5 * r_scatter
        self.r_disk_au = np.clip(self.r_disk_au, 5.0, 500.0)

        # --- Surface density slope (Hayashi 1981: -1.5 ± ~0.3) ---
        self.sigma_slope = np.clip(rng.normal(-1.5, 0.3), -2.5, -0.5)

        # --- Snow line during the disk phase ---
        # The water snow line is set by pre-main-sequence irradiation plus
        # viscous disk heating rather than the mature main-sequence luminosity.
        # This keeps the condensation front farther out for young systems,
        # closer to Kennedy & Kenyon (2008) / Oka et al. (2011).
        pre_ms_lum_rel = np.clip(1.8 * star_mass**1.6, 0.05, 80.0)
        snow_irr_au = 2.7 * pre_ms_lum_rel**0.5
        accretion_rate_1e8 = np.clip((self.disk_mass / 0.05) * star_mass**1.5, 0.05, 50.0)
        snow_visc_au = 1.7 * accretion_rate_1e8**(4.0 / 9.0) * max(star_mass, 0.1)**(1.0 / 3.0)
        self.snow_line_au = max(snow_irr_au, snow_visc_au)
        self.snow_line_au = np.clip(self.snow_line_au, 0.5, 50.0)

        # --- External FUV irradiation (Galactic environment) ---
        # Dense clusters (r < 4 kpc): high FUV; field (r > 8 kpc): low FUV
        # Units: G₀ (Habing field, 1.6e-3 erg/cm²/s)
        base_fuv = 10**(3.5 - 0.35 * r_galactic_kpc)  # ~3000 at 1kpc, ~10 at 8kpc
        fuv_scatter = 10**rng.normal(0, 0.5)
        self.F_UV_G0 = np.clip(base_fuv * fuv_scatter, 1.0, 1e5)

        # --- Disk lifetime (Haisch+2001) ---
        # τ ≈ 3 Myr × M_star^(-0.5), reduced by strong FUV photoevaporation
        fuv_reduction = np.exp(-self.F_UV_G0 / 3000.0)
        lifetime_scatter = 10**rng.normal(0, 0.3)
        self.lifetime_myr = (DISK_LIFETIME_MEDIAN * star_mass**(-0.5)
                             * fuv_reduction * lifetime_scatter)
        self.lifetime_myr = np.clip(self.lifetime_myr, 0.5, 20.0)
        self.lifetime_gyr = self.lifetime_myr * 1e-3

        # --- Surface density normalization ---
        # Normalize Sigma_0 so that integrating the adopted power-law profile
        # over the chosen disk extent reproduces the sampled total disk mass.
        gas_mass_profile_norm = self._integrate_power_law_mass(
            0.05, self.r_disk_au, sigma_0=1.0, slope=self.sigma_slope
        )
        if gas_mass_profile_norm <= 0:
            gas_mass_profile_norm = 1.0
        self.sigma_0 = self.disk_mass_g / gas_mass_profile_norm
        self._solid_redistribution_norm = 1.0
        self._solid_redistribution_norm = self._compute_solid_redistribution_norm()

        # --- Total solid mass budget ---
        self.total_solid_mass_g = self._integrate_solid_mass(0.1, self.r_disk_au)
        self.total_solid_mass_earth = self.total_solid_mass_g / M_EARTH_G

    # --------------------------------------------------------
    # Surface density profiles
    # --------------------------------------------------------
    def gas_surface_density(self, r_au):
        """Gas surface density Σ_gas(r) in g/cm² (Hayashi 1981)."""
        r_au = np.atleast_1d(np.float64(r_au))
        return self.sigma_0 * (r_au / 1.0)**self.sigma_slope

    def _integrate_power_law_mass(self, r_in, r_out, sigma_0=None, slope=None, n_points=400):
        """Integrate a power-law surface density annulus into grams."""
        sigma_0 = self.sigma_0 if sigma_0 is None else sigma_0
        slope = self.sigma_slope if slope is None else slope
        r = np.linspace(max(r_in, 0.05), r_out, n_points)
        sigma = sigma_0 * (r / 1.0) ** slope
        r_cm = r * AU_CM
        dr_cm = np.gradient(r_cm)
        dM = sigma * 2 * np.pi * r_cm * dr_cm
        return max(np.sum(dM), 0.0)

    def integrate_gas_mass(self, r_in=0.05, r_out=None):
        """Integrate the current gas disk mass between two radii."""
        if r_out is None:
            r_out = self.r_disk_au
        return self._integrate_power_law_mass(r_in, r_out)

    def solid_surface_density(self, r_au):
        """Solid surface density Σ_solid(r) in g/cm² with snow line enhancement."""
        r_au = np.atleast_1d(np.float64(r_au))
        sigma_gas = self.gas_surface_density(r_au)
        sigma_solid = sigma_gas * self.dust_to_gas

        # Snow line enhancement (Hayashi 1981): ×4.2 beyond snow line
        beyond_snow = r_au > self.snow_line_au
        # Smooth transition over ~0.5 AU width
        transition = 1.0 + (SNOW_ENHANCEMENT - 1.0) / (
            1.0 + np.exp(-(r_au - self.snow_line_au) / 0.3))
        sigma_solid = sigma_solid * transition / max(self._solid_redistribution_norm, 1e-8)
        return sigma_solid

    def _compute_solid_redistribution_norm(self, r_in=0.1, r_out=None, n_points=400):
        """Normalize snow-line enhancement so it redistributes rather than creates solids."""
        if r_out is None:
            r_out = self.r_disk_au
        r = np.linspace(max(r_in, 0.05), r_out, n_points)
        sigma_gas = self.gas_surface_density(r) * self.dust_to_gas
        transition = 1.0 + (SNOW_ENHANCEMENT - 1.0) / (
            1.0 + np.exp(-(r - self.snow_line_au) / 0.3))
        r_cm = r * AU_CM
        dr_cm = np.gradient(r_cm)
        base_mass = np.sum(sigma_gas * 2 * np.pi * r_cm * dr_cm)
        enhanced_mass = np.sum(sigma_gas * transition * 2 * np.pi * r_cm * dr_cm)
        if base_mass <= 0:
            return 1.0
        return max(float(enhanced_mass / base_mass), 1.0)

    def _integrate_solid_mass(self, r_in, r_out, n_points=200):
        """Integrate total solid mass in annulus [r_in, r_out] AU → grams."""
        r = np.linspace(max(r_in, 0.05), r_out, n_points)
        sigma = self.solid_surface_density(r)  # g/cm²
        # dM = Σ × 2πr dr, with r in cm
        r_cm = r * AU_CM
        dr_cm = np.gradient(r_cm)
        dM = sigma * 2 * np.pi * r_cm * dr_cm
        return max(np.sum(dM), 0.0)

    # --------------------------------------------------------
    # Pressure trap factor (Birnstiel+2012)
    # --------------------------------------------------------
    def pressure_trap_factor(self, r_au):
        """Estimate pressure bump enhancement at given radius.

        Pressure traps at snow line, dead zone edges, and planet gaps
        concentrate dust and enhance local solid density (Birnstiel+2012).
        Returns multiplicative factor 1.0–3.0.
        """
        # Enhanced trapping near snow line and 2:1 resonance locations
        dist_snow = abs(r_au - self.snow_line_au) / max(self.snow_line_au, 0.1)
        trap = 1.0 + (PTRAP_MAX - 1.0) * np.exp(-dist_snow**2 / 0.1)
        # Dead zone inner edge (~0.5–1 AU) can also trap material
        dist_dz = abs(r_au - 0.7) / 0.5
        trap += 0.5 * np.exp(-dist_dz**2 / 0.2)
        return np.clip(trap, PTRAP_MIN, PTRAP_MAX)

    # --------------------------------------------------------
    # Planetesimal formation efficiency (Johansen+2014)
    # --------------------------------------------------------
    def planetesimal_efficiency(self, r_au):
        """Local planetesimal formation efficiency epsilon_pf.

        Streaming instability forms planetesimals only in regions of
        high dust concentration.  Global efficiency is ~0.1-1% of
        local solid mass (Johansen+2014, Simon+2016).
        Returns fraction 0.001-0.01.
        """
        base = EPSILON_PF_MIN
        # Enhanced beyond snow line (higher solid/gas ratio)
        if r_au > self.snow_line_au:
            base = EPSILON_PF_MIN + 0.003 * min(self.dust_to_gas / 0.01, 2.0)
        # Metallicity boost: higher Z -> easier streaming instability
        z_boost = min(self.metallicity_z / Z_SUN, 2.0)
        eff = base * z_boost
        return np.clip(eff, EPSILON_PF_MIN, EPSILON_PF_MAX)

    # --------------------------------------------------------
    # Summary dict for serialization
    # --------------------------------------------------------
    def to_dict(self):
        """Return disk properties as JSON-serializable dict."""
        return {
            'star_mass': round(self.star_mass, 4),
            'disk_mass_Msun': round(float(self.disk_mass), 6),
            'disk_mass_Mearth': round(float(self.disk_mass * 332946), 1),
            'dust_to_gas': round(float(self.dust_to_gas), 6),
            'metallicity_z': round(float(self.metallicity_z), 6),
            'r_disk_au': round(float(self.r_disk_au), 1),
            'sigma_slope': round(float(self.sigma_slope), 2),
            'sigma_0_g_cm2': round(float(self.sigma_0), 1),
            'snow_line_au': round(float(self.snow_line_au), 2),
            'lifetime_myr': round(float(self.lifetime_myr), 2),
            'F_UV_G0': round(float(self.F_UV_G0), 1),
            'total_solid_mass_earth': round(float(self.total_solid_mass_earth), 3),
        }


# ============================================================
# ASTEROID BELT SEED MASS
# ============================================================
def compute_belt_seed_mass(disk, belt_in_au=None, belt_out_au=None, rng=None):
    """Compute the initial seed mass of an asteroid belt.

    The belt occupies a radial annulus (default: 2.0–3.5 AU scaled by
    snow line position). Seed mass is:

        seed = solid_mass × pressure_trap × planetesimal_efficiency

    Parameters
    ----------
    disk : ProtoplanetaryDisk
    belt_in_au, belt_out_au : float or None
        Inner/outer belt edges. If None, placed relative to snow line.
    rng : np.random.Generator or None

    Returns
    -------
    dict with seed_mass_earth, belt_in, belt_out, components
    """
    if rng is None:
        rng = disk.rng

    # Default belt location: just inside snow line to just outside
    # Solar system: 2.1–3.3 AU, snow line ~2.7 AU
    if belt_in_au is None:
        belt_in_au = max(0.5, disk.snow_line_au * 0.75)
    if belt_out_au is None:
        belt_out_au = min(disk.r_disk_au * 0.5, disk.snow_line_au * 1.3)

    if belt_out_au <= belt_in_au:
        return {
            'seed_mass_earth': 0.0, 'belt_in_au': round(belt_in_au, 2),
            'belt_out_au': round(belt_out_au, 2), 'solid_mass_earth': 0.0,
            'pressure_trap_factor': 1.0, 'planetesimal_efficiency': 0.0,
            'snow_enhancement_active': False,
        }

    # 1. Raw solid mass in the belt annulus
    solid_mass_g = disk._integrate_solid_mass(belt_in_au, belt_out_au)
    solid_mass_earth = solid_mass_g / M_EARTH_G

    # 2. Mean pressure trap factor across the belt
    r_mid = 0.5 * (belt_in_au + belt_out_au)
    ptrap = float(disk.pressure_trap_factor(r_mid))

    # 3. Planetesimal formation efficiency
    eps_pf = float(disk.planetesimal_efficiency(r_mid))

    # 4. Snow line enhancement is already included in solid_surface_density,
    #    note whether the belt straddles the snow line
    snow_active = bool(belt_in_au < disk.snow_line_au < belt_out_au)

    seed_mass_earth = solid_mass_earth * ptrap * eps_pf

    return {
        'seed_mass_earth': round(float(seed_mass_earth), 6),
        'belt_in_au': round(float(belt_in_au), 2),
        'belt_out_au': round(float(belt_out_au), 2),
        'solid_mass_earth': round(float(solid_mass_earth), 4),
        'pressure_trap_factor': round(ptrap, 3),
        'planetesimal_efficiency': round(eps_pf, 5),
        'snow_enhancement_active': snow_active,
    }


# ============================================================
# SURVIVAL FRACTION
# ============================================================
def compute_survival_fraction(disk, planets, age_gyr, rng=None, belt_seed_mass_earth=None):
    """Compute the fraction of belt seed mass that survives dynamical erosion.

    Six depletion mechanisms (Raymond+17, Walsh+11, Morbidelli+12,
    Nesvorný 2018):

    1. Giant planet presence      — gravitational clearing
    2. Giant planet migration     — Grand Tack sweeping (Walsh+11)
    3. Resonance scanning         — mean-motion resonance sweep (Morbidelli+12)
    4. Planetary embryo scattering — embryo excitation
    5. Late instability           — Nice model (Nesvorný 2018)
    6. Collisional grinding       — cascade erosion

    Parameters
    ----------
    disk : ProtoplanetaryDisk
    planets : list of dict
        Planet dictionaries from generate_planets().
    age_gyr : float
        System age in Gyr.
    rng : np.random.Generator or None

    Returns
    -------
    dict with survival_fraction, individual factors, and flags
    """
    if rng is None:
        rng = disk.rng

    survival = 1.0
    factors = {}

    # Classify planets
    giants = [p for p in planets if p.get('type') == 'gas_giant']
    ice_giants = [p for p in planets if p.get('type') == 'mini_neptune']
    rocky = [p for p in planets if p.get('type') in ('rocky', 'hot_rocky')]
    n_giant = len(giants)
    n_all = len(planets)

    # ---- 1. Giant planet existence (Raymond & Izidoro 2017) ----
    # Each giant depletes the belt by a factor ~100-1000×
    if n_giant > 0:
        # Closer giants are more destructive
        closest_giant_au = min(g['semi_major_au'] for g in giants)
        depletion_per_giant = np.clip(
            0.001 * (5.0 / max(closest_giant_au, 0.5))**2, 0.0001, 0.01)
        f_giant = depletion_per_giant ** n_giant
    else:
        f_giant = 1.0
    factors['giant_presence'] = round(float(f_giant), 6)
    survival *= f_giant

    # ---- 2. Giant planet migration amplitude (Walsh+11 Grand Tack) ----
    # If a giant formed far out and migrated inward, it sweeps through
    # the belt region. Migration amplitude estimated from eccentricity
    # and orbit proximity.
    if n_giant > 0:
        # Estimate migration amplitude from giant eccentricity
        max_ecc = max(g.get('eccentricity', 0.05) for g in giants)
        migration_amplitude = max_ecc * 5.0  # AU equivalent
        # Scatter to account for uncertain migration history
        migration_amplitude *= (1 + rng.exponential(0.5))
        f_migration = np.exp(-migration_amplitude / 0.5)
        f_migration = np.clip(f_migration, 0.001, 1.0)
    else:
        f_migration = 1.0
        migration_amplitude = 0.0
    factors['giant_migration'] = round(float(f_migration), 6)
    survival *= f_migration

    # ---- 3. Resonance scanning (Morbidelli+2012) ----
    # Giant planets' mean-motion resonances sweep through the belt
    # as the planets migrate, exciting and ejecting asteroids.
    if n_giant > 0:
        f_resonance = max(0.01, 1.0 - 0.3 * n_giant)
    elif len(ice_giants) > 0:
        f_resonance = max(0.1, 1.0 - 0.1 * len(ice_giants))
    else:
        f_resonance = 1.0
    factors['resonance_scanning'] = round(float(f_resonance), 4)
    survival *= f_resonance

    # ---- 4. Planetary embryo scattering ----
    # Embryos (rocky planets and proto-planets) scatter belt material
    # through close encounters.
    n_embryos = len(rocky) + len(ice_giants)
    f_embryo = max(0.01, 1.0 - 0.1 * n_embryos)
    factors['embryo_scattering'] = round(float(f_embryo), 4)
    survival *= f_embryo

    # ---- 5. Late instability / Nice model (Nesvorný 2018) ----
    # Probability of late instability increases with number of giant
    # planets and orbital compactness.
    if n_giant >= 2:
        p_instability = np.clip(0.5 + 0.1 * (n_giant - 2), 0.3, 0.9)
    elif n_giant == 1 and len(ice_giants) >= 1:
        p_instability = 0.3
    else:
        p_instability = 0.05
    late_instability = bool(rng.random() < p_instability)
    if late_instability:
        f_instability = 0.1  # ~90% mass loss during instability
    else:
        f_instability = 1.0
    factors['late_instability'] = round(float(f_instability), 4)
    factors['instability_occurred'] = late_instability
    survival *= f_instability

    # ---- 6. Collisional grinding / cascade ----
    # More massive belts collide and grind down faster because the intrinsic
    # collision probability and catastrophic impactor abundance both rise
    # with the planetesimal surface density (Wyatt+2007, Bottke+2005).
    if belt_seed_mass_earth is None:
        belt_seed_mass_earth = compute_belt_seed_mass(disk, rng=rng)['seed_mass_earth']
    belt_seed_mass_earth = max(float(belt_seed_mass_earth), 1e-8)
    belt_center_au = 0.5 * (max(0.5, disk.snow_line_au * 0.75) + min(disk.r_disk_au * 0.5, disk.snow_line_au * 1.3))
    mass_ratio = belt_seed_mass_earth / 5e-4
    radial_factor = (max(belt_center_au, 0.5) / 2.7) ** 1.4
    t_coll = 0.6 * radial_factor * mass_ratio ** -0.6
    t_coll = np.clip(t_coll, 0.03, 6.0)
    collisional_floor = 0.003 + 0.02 / (1.0 + mass_ratio ** 0.5)
    f_collisional = collisional_floor + (1.0 - collisional_floor) * np.exp(-age_gyr / t_coll)
    f_collisional = np.clip(f_collisional, 0.001, 1.0)
    factors['collisional_grinding'] = round(float(f_collisional), 4)
    survival *= f_collisional

    survival = np.clip(survival, 1e-10, 1.0)

    return {
        'survival_fraction': round(float(survival), 8),
        'factors': factors,
        'n_giants': n_giant,
        'n_embryos': n_embryos,
        'migration_amplitude_au': round(float(migration_amplitude), 2),
        'collisional_timescale_gyr': round(float(t_coll), 4),
        'belt_seed_mass_earth': round(float(belt_seed_mass_earth), 6),
    }


# ============================================================
# ASTEROID BELT BUILDER  (with full radial state variables)
# ============================================================
N_BELT_BINS = 50   # radial resolution for belt profiles

def build_asteroid_belt(disk, planets, age_gyr, rng=None):
    """Build complete asteroid belt model with radial state variables.

    final_belt_mass = seed_mass * survival_fraction

    State variables (all resolved on a radial grid a[]):
      1. a0           — belt center radius (AU)
      2. ain, aout    — inner / outer boundary (AU)
      3. sigma_a      — surface density profile (g/cm2)
      4. size_dist    — cumulative size distribution N(>D)
      5. ecc_a        — eccentricity distribution e(a)
      6. inc_a        — inclination distribution i(a) (deg)
      7. comp_a       — composition zone distribution (S/C/M fractions)
      8. resonances   — resonance locations and gap structure
      9. collision_activity — radial collision rate proxy
     10. survived_frac — radial survived mass fraction f_surv(a)

    References:
        Dohnanyi (1969)   — collisional equilibrium size distribution
        Bottke+2005       — asteroid belt structure
        DeMeo & Carry 14  — compositional gradient
        Dermott+1984      — Kirkwood gaps / resonances

    Parameters
    ----------
    disk : ProtoplanetaryDisk
    planets : list of dict
    age_gyr : float
    rng : np.random.Generator or None
    """
    if rng is None:
        rng = disk.rng

    seed = compute_belt_seed_mass(disk, rng=rng)
    surv = compute_survival_fraction(
        disk, planets, age_gyr, rng=rng, belt_seed_mass_earth=seed['seed_mass_earth']
    )

    ain  = seed['belt_in_au']
    aout = seed['belt_out_au']
    a0   = 0.5 * (ain + aout)
    belt_width = aout - ain

    final_mass_earth = seed['seed_mass_earth'] * surv['survival_fraction']

    # Empty belt guard
    if belt_width <= 0 or final_mass_earth <= 0:
        empty = _empty_belt_state(ain, aout, a0, seed, surv, final_mass_earth)
        return empty

    # ----------------------------------------------------------------
    # Radial grid for all profiles
    # ----------------------------------------------------------------
    a = np.linspace(ain, aout, N_BELT_BINS)
    da = a[1] - a[0] if len(a) > 1 else 1.0

    # Classify planets for resonance/gap calculations
    giants = [p for p in planets if p.get('type') == 'gas_giant']

    # ================================================================
    # 1-2. a0, ain, aout  (already computed above)
    # ================================================================

    # ================================================================
    # 3. Surface density profile  sigma(a)  [g/cm2]
    #    Start from protoplanetary disk solid density, then apply
    #    planetesimal efficiency and survival to get present-day density.
    # ================================================================
    sigma_proto = disk.solid_surface_density(a)            # g/cm2
    eps_arr = np.array([disk.planetesimal_efficiency(ai) for ai in a])
    ptrap_arr = np.array([disk.pressure_trap_factor(ai) for ai in a])

    # Resonance gaps carve density depletions (computed below, applied here)
    gap_depletion = _resonance_gap_profile(a, giants, disk.snow_line_au)

    sigma_belt = (sigma_proto * eps_arr * ptrap_arr
                  * surv['survival_fraction'] * gap_depletion)

    # ================================================================
    # 4. Size distribution  N(>D) — Dohnanyi (1969) collisional cascade
    #    dN/dD ~ D^(-q),  q = 3.5 for collisional equilibrium
    #    N(>D) ~ D^(1-q) = D^(-2.5)
    #    Normalize to total belt mass; D_max ~ 1000 km (Ceres-class)
    # ================================================================
    q_slope = 3.5                      # Dohnanyi equilibrium
    D_max_km = 900.0 * (final_mass_earth / 5e-4)**0.3   # scale with mass
    D_max_km = np.clip(D_max_km, 10.0, 2000.0)
    D_min_km = 0.001                   # 1 m
    # Sample diameters for the cumulative distribution
    D_km = np.geomspace(D_min_km, D_max_km, 40)
    N_gt_D = (D_km / D_min_km)**(1 - q_slope)      # relative N(>D)
    # Scale so N(>1 km) matches expected population
    # Real belt: ~1.5 million bodies > 1 km for 4.5e-4 Me
    n_1km_ref = 1.5e6 * (final_mass_earth / max(4.5e-4, 1e-12))
    idx_1km = np.searchsorted(D_km, 1.0)
    if idx_1km < len(N_gt_D) and N_gt_D[idx_1km] > 0:
        N_gt_D = N_gt_D * (n_1km_ref / N_gt_D[idx_1km])

    # ================================================================
    # 5. Eccentricity distribution  e(a)
    #    Mean eccentricity increases near resonances and where giants
    #    have scattered material.  Background ~ 0.1, near resonances ~ 0.3
    #    Reference: Bottke+2005
    # ================================================================
    e_background = 0.05 + 0.10 * (1 - surv['survival_fraction'])
    e_background = np.clip(e_background, 0.02, 0.35)
    ecc_a = np.full_like(a, e_background)
    # Excite near resonance gaps
    ecc_a += 0.15 * (1 - gap_depletion)
    # Excite near giant planets
    for g in giants:
        dist = np.abs(a - g['semi_major_au']) / max(g['semi_major_au'], 0.5)
        ecc_a += 0.2 * np.exp(-dist**2 / 0.05)
    ecc_a = np.clip(ecc_a, 0.01, 0.6)

    # ================================================================
    # 6. Inclination distribution  i(a)  [deg]
    #    Dynamically excited by embryos and giants.
    #    i ~ e / 2 (equipartition), plus secular perturbation.
    #    Reference: Minton & Malhotra (2010)
    # ================================================================
    inc_background = 3.0 + 8.0 * (1 - surv['survival_fraction'])
    inc_background = np.clip(inc_background, 1.0, 25.0)
    inc_a = np.full_like(a, inc_background)
    # Enhanced inclination near resonances
    inc_a += 5.0 * (1 - gap_depletion)
    # Secular excitation from giant planets
    for g in giants:
        dist = np.abs(a - g['semi_major_au']) / max(g['semi_major_au'], 0.5)
        inc_a += 5.0 * np.exp(-dist**2 / 0.1)
    inc_a = np.clip(inc_a, 0.5, 40.0)

    # ================================================================
    # 7. Composition zone distribution  comp(a)
    #    S-type (silicate) dominates inner belt (a < snow line)
    #    C-type (carbonaceous) dominates outer belt (a > snow line)
    #    M-type (metallic) ~5% throughout
    #    Reference: DeMeo & Carry (2014), Gradie & Tedesco (1982)
    # ================================================================
    # Smooth S-to-C transition across the snow line
    transition_width = 0.3 * belt_width
    snow_rel = (a - disk.snow_line_au) / max(transition_width, 0.1)
    c_frac_a = 0.1 + 0.8 / (1 + np.exp(-snow_rel * 3))   # sigmoid
    m_frac_a = np.full_like(a, 0.05)                        # ~5% metallic
    s_frac_a = np.clip(1.0 - c_frac_a - m_frac_a, 0.0, 1.0)
    # Bundle as dict of arrays
    comp_a = {
        'S_type': np.round(s_frac_a, 4).tolist(),
        'C_type': np.round(c_frac_a, 4).tolist(),
        'M_type': np.round(m_frac_a, 4).tolist(),
    }

    # Bulk fractions (mass-weighted)
    sigma_total = np.sum(sigma_belt)
    if sigma_total > 0:
        s_bulk = float(np.sum(s_frac_a * sigma_belt) / sigma_total)
        c_bulk = float(np.sum(c_frac_a * sigma_belt) / sigma_total)
        m_bulk = float(np.sum(m_frac_a * sigma_belt) / sigma_total)
    else:
        s_bulk, c_bulk, m_bulk = 0.5, 0.45, 0.05

    # ================================================================
    # 8. Resonance / gap structure
    #    Kirkwood gaps at mean-motion resonances with the dominant giant
    #    planet.  a_res = a_giant * (p/q)^(2/3)  (Kepler's 3rd law)
    #    Reference: Dermott+1984
    # ================================================================
    resonances = _compute_resonances(a, ain, aout, giants, gap_depletion)

    # ================================================================
    # 9. Collision activity  C(a)
    #    Proxy: proportional to sigma^2 / (e * P_orb), where
    #    P_orb ~ a^1.5 (Keplerian).
    #    Higher density + lower eccentricity = more collisions.
    #    Reference: Bottke+2005, Farinella & Davis (1992)
    # ================================================================
    P_orb = a**1.5  # relative orbital period
    collision_rate = sigma_belt**2 / (np.maximum(ecc_a, 0.01) * P_orb)
    # Normalize to peak = 1
    cr_max = np.max(collision_rate) if np.max(collision_rate) > 0 else 1.0
    collision_rate = collision_rate / cr_max

    # Global collision activity index (0 = dead belt, 1 = very active)
    collision_index = float(np.mean(collision_rate))
    collision_index = round(np.clip(collision_index, 0, 1), 4)

    # ================================================================
    # 10. Survived mass fraction  f_surv(a)
    #     Radially resolved: survival is worse near gaps and giant orbits
    # ================================================================
    f_surv_a = np.full_like(a, surv['survival_fraction'])
    # Extra depletion at gap locations
    f_surv_a *= gap_depletion
    # Extra depletion near giants
    for g in giants:
        dist = np.abs(a - g['semi_major_au']) / max(g['semi_major_au'], 0.5)
        f_surv_a *= np.clip(1.0 - 0.8 * np.exp(-dist**2 / 0.03), 0.01, 1.0)
    f_surv_a = np.clip(f_surv_a, 1e-6, 1.0)

    # ================================================================
    # Assemble output
    # ================================================================
    a_list = np.round(a, 4).tolist()

    return {
        # Scalars
        'final_mass_earth': round(float(final_mass_earth), 8),
        'final_mass_kg': round(float(final_mass_earth * M_EARTH), 2),
        'seed': seed,
        'survival': surv,

        # 1-2. Geometry
        'a0_au': round(float(a0), 3),
        'ain_au': round(float(ain), 3),
        'aout_au': round(float(aout), 3),
        'belt_width_au': round(float(belt_width), 3),

        # 3. Surface density profile
        'radial_grid_au': a_list,
        'sigma_a_g_cm2': np.round(sigma_belt, 6).tolist(),

        # 4. Size distribution
        'size_dist': {
            'D_km': np.round(D_km, 6).tolist(),
            'N_gt_D': np.round(N_gt_D, 1).tolist(),
            'q_slope': q_slope,
            'D_max_km': round(float(D_max_km), 1),
        },

        # 5. Eccentricity distribution
        'ecc_a': np.round(ecc_a, 4).tolist(),

        # 6. Inclination distribution
        'inc_a_deg': np.round(inc_a, 2).tolist(),

        # 7. Composition zones
        'comp_a': comp_a,
        'composition': {
            'S_type_frac': round(s_bulk, 3),
            'C_type_frac': round(c_bulk, 3),
            'M_type_frac': round(m_bulk, 3),
        },

        # 8. Resonances / gaps
        'resonances': resonances,

        # 9. Collision activity
        'collision_rate_a': np.round(collision_rate, 4).tolist(),
        'collision_index': collision_index,

        # 10. Survived mass fraction
        'survived_frac_a': np.round(f_surv_a, 6).tolist(),
        'survived_frac_mean': round(float(np.mean(f_surv_a)), 6),

        # Legacy fields
        'belt_center_au': round(float(a0), 2),
        'inclination_dispersion_deg': round(float(np.mean(inc_a)), 1),
    }


# ============================================================
# HELPER: Resonance gap profile
# ============================================================
def _resonance_gap_profile(a, giants, snow_line_au):
    """Compute resonance-driven density depletion profile.

    Kirkwood gaps at mean-motion resonances with giant planets:
        a_res = a_giant * (p/q)^(2/3)

    Returns array of depletion factors (0=full gap, 1=no gap).
    Reference: Dermott+1984, Kirkwood (1867).
    """
    gap = np.ones_like(a)
    # Resonance ratios with Jupiter analogs: 4:1, 3:1, 5:2, 7:3, 2:1
    res_ratios = [
        (4, 1, 0.90),   # 4:1 — strong depletion
        (3, 1, 0.85),   # 3:1 — Kirkwood gap, very strong
        (5, 2, 0.70),   # 5:2 — moderate gap
        (7, 3, 0.50),   # 7:3 — weak gap
        (2, 1, 0.80),   # 2:1 — strong Kirkwood gap
    ]
    gap_width_au = 0.05  # half-width of each gap

    for g in giants:
        a_g = g['semi_major_au']
        for p, q, strength in res_ratios:
            a_res = a_g * (float(q) / float(p))**(2.0/3.0)
            # Gaussian gap profile
            gap *= 1.0 - strength * np.exp(-((a - a_res) / gap_width_au)**2)

    gap = np.clip(gap, 0.01, 1.0)
    return gap


def _compute_resonances(a, ain, aout, giants, gap_depletion):
    """Compute resonance locations and gap properties.

    Returns list of resonance dicts with location, ratio,
    width, depth, and associated giant planet info.
    """
    res_ratios = [
        (4, 1, '4:1'), (3, 1, '3:1'), (5, 2, '5:2'),
        (7, 3, '7:3'), (2, 1, '2:1'),
    ]
    resonances = []
    for g in giants:
        a_g = g['semi_major_au']
        for p, q, label in res_ratios:
            a_res = a_g * (float(q) / float(p))**(2.0/3.0)
            if ain <= a_res <= aout:
                # Find depth at resonance from gap_depletion profile
                idx = np.argmin(np.abs(a - a_res))
                depth = round(1.0 - float(gap_depletion[idx]), 3)
                resonances.append({
                    'a_au': round(float(a_res), 4),
                    'ratio': label,
                    'depth': depth,
                    'giant_a_au': round(float(a_g), 3),
                })
    return resonances


def _empty_belt_state(ain, aout, a0, seed, surv, final_mass_earth):
    """Return an empty belt with zeroed state arrays."""
    return {
        'final_mass_earth': round(float(final_mass_earth), 8),
        'final_mass_kg': round(float(final_mass_earth * M_EARTH), 2),
        'seed': seed, 'survival': surv,
        'a0_au': round(float(a0), 3),
        'ain_au': round(float(ain), 3),
        'aout_au': round(float(aout), 3),
        'belt_width_au': 0.0,
        'radial_grid_au': [], 'sigma_a_g_cm2': [],
        'size_dist': {'D_km': [], 'N_gt_D': [], 'q_slope': 3.5, 'D_max_km': 0},
        'ecc_a': [], 'inc_a_deg': [],
        'comp_a': {'S_type': [], 'C_type': [], 'M_type': []},
        'composition': {'S_type_frac': 0, 'C_type_frac': 0, 'M_type_frac': 0},
        'resonances': [],
        'collision_rate_a': [], 'collision_index': 0,
        'survived_frac_a': [], 'survived_frac_mean': 0,
        'belt_center_au': round(float(a0), 2),
        'inclination_dispersion_deg': 0,
    }
