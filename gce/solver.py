"""
Multi-zone Galactic Chemical Evolution solver.

Evolves gas composition in concentric annuli under:
  - CCSNe (instantaneous), Type Ia (DTD), AGB (delayed), NSM (DTD)
  - Double-exponential gas infall (inside-out)
  - Mass-proportional outflow
"""
import numpy as np
from .config import (ELEMENTS, N_ELEMENTS, EL_IDX,
                     BBN_ABUNDANCE, SOLAR_X, Z_SUN, coerce_solver_params)
from .physics import (imf_norm, stellar_lifetime, dying_mass_at,
                      remnant_mass, dtd_powerlaw, imf_number_per_msun,
                      imf_return_fraction)
from .yields import (ccsne_yields, TYPE_IA_YIELDS, agb_yields, nsm_yields,
                     collapsar_yields, CCSNE_HE_YIELD)


class GCESolver:
    """One-shot multi-zone GCE solver."""

    def __init__(self, params=None):
        self.p = coerce_solver_params(params)
        self._build_time_grid()
        self._build_radial_grid()
        self._m_imf, self._phi_imf = imf_norm()
        self.imf_stats = self._build_imf_stats()

    # --------------------------------------------------------
    # Grids
    # --------------------------------------------------------
    def _build_time_grid(self):
        t1 = np.arange(0.0, 1.0,  0.01)
        t2 = np.arange(1.0, 5.0,  0.05)
        t3 = np.arange(5.0, self.p['t_max'] + 0.05, 0.1)
        self.t = np.concatenate([t1, t2, t3])
        self.nt = len(self.t)
        self.dt = np.zeros(self.nt)
        self.dt[1:] = np.diff(self.t)
        self.dt[0] = self.dt[1]

    def _build_radial_grid(self):
        p = self.p
        self.r = np.arange(p['r_min'], p['r_max'], p['dr'])
        self.nr = len(self.r)

    def _build_imf_stats(self):
        return {
            'ccsn_number_per_msun': imf_number_per_msun(8.0, 100.0),
            'ccsn_return_fraction': imf_return_fraction(8.0, 100.0),
            'agb_number_per_msun': imf_number_per_msun(1.0, 8.0),
        }

    def _interp_history(self, t_query, history, it):
        return float(np.interp(t_query, self.t[:it+1], history[:it+1]))

    def _agb_source_state(self, m_agb, t_now, it, sfr_hist, Z_hist, max_iter=3):
        """Estimate birth state for an AGB progenitor dying at the current time."""
        if it <= 0:
            return None

        tau_agb = float(stellar_lifetime(m_agb, max(Z_hist[it - 1], 1e-6))[0])
        for _ in range(max_iter):
            t_form = t_now - tau_agb
            if not (0.0 <= t_form < t_now):
                return None
            z_form = max(self._interp_history(t_form, Z_hist, it), 1e-6)
            tau_new = float(stellar_lifetime(m_agb, z_form)[0])
            if abs(tau_new - tau_agb) <= 1e-4 * max(tau_agb, 1e-6):
                tau_agb = tau_new
                break
            tau_agb = tau_new

        t_form = t_now - tau_agb
        if not (0.0 <= t_form < t_now):
            return None
        return {
            't_form': t_form,
            'tau': tau_agb,
            'sfr_form': self._interp_history(t_form, sfr_hist, it),
            'Z_form': max(self._interp_history(t_form, Z_hist, it), 1e-6),
        }

    # --------------------------------------------------------
    # Gas infall (double-exponential, inside-out)
    # --------------------------------------------------------
    def _infall(self, r, t):
        p = self.p
        tau_thin = p['infall_tau_thin'] * (1.0 + r / 12.0)
        rd = p['infall_rd']
        thick = (p['infall_sigma_thick0'] * np.exp(-r / rd)
                 / p['infall_tau_thick']
                 * np.exp(-t / p['infall_tau_thick']))
        thin  = (p['infall_sigma_thin0'] * np.exp(-r / rd)
                 / tau_thin
                 * np.exp(-t / tau_thin))
        return thick + thin   # M☉ pc⁻² Gyr⁻¹

    # --------------------------------------------------------
    # Star-formation rate (Kennicutt-Schmidt)
    # --------------------------------------------------------
    def _sfr(self, sigma_gas):
        e = self.p['sfr_efficiency']
        n = self.p['sfr_exponent']
        return e * np.clip(sigma_gas, 0, None)**n

    # --------------------------------------------------------
    # Pre-compute DTD matrices
    # --------------------------------------------------------
    def _dtd_matrix(self, t_min, slope):
        """Lower-triangular matrix  M[i,j] = DTD(t[i]-t[j]) * dt[j]."""
        delays = self.t[:, None] - self.t[None, :]     # (nt, nt)
        raw = dtd_powerlaw(delays, t_min, slope)
        raw = np.tril(raw, -1)
        # normalise so that ∫ DTD dt = 1
        norm = np.trapezoid(dtd_powerlaw(self.t, t_min, slope), self.t)
        if norm > 0:
            raw /= norm
        return raw * self.dt[None, :]                   # include dt_j

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    def solve(self):
        p = self.p
        nt, nr, nel = self.nt, self.nr, N_ELEMENTS

        # Outputs
        self.X    = np.zeros((nel, nr, nt))   # mass fractions
        self.Mgas = np.zeros((nr, nt))
        self.Mstar= np.zeros((nr, nt))
        self.SFR  = np.zeros((nr, nt))
        self.Z_   = np.zeros((nr, nt))        # total metallicity

        # Pre-compute DTD matrices (shared across zones)
        Mia  = self._dtd_matrix(p['ia_t_min'],  p['ia_dtd_slope'])
        Mnsm = self._dtd_matrix(p['nsm_t_min'], p['nsm_dtd_slope'])
        y_ia = TYPE_IA_YIELDS * p.get('yield_ia_multiplier', 1.0)
        y_coll = collapsar_yields(p.get('collapsar_ejecta', 0.05),
                                  p.get('yield_r_multiplier', 1.0))
        y_nsm = nsm_yields(p['nsm_ejecta'], p.get('yield_r_multiplier', 1.0))

        for ir in range(nr):
            self._evolve_zone(ir, Mia, Mnsm, y_ia, y_nsm, y_coll)

        return self._results()

    # --------------------------------------------------------
    # Evolve one radial zone
    # --------------------------------------------------------
    def _evolve_zone(self, ir, Mia, Mnsm, y_ia, y_nsm, y_coll):
        p = self.p
        r = self.r[ir]
        nt = self.nt
        nel = N_ELEMENTS

        # Local state
        mg  = 0.1                          # initial tiny gas seed
        ms  = 0.0
        X   = BBN_ABUNDANCE.copy()
        sfr_hist = np.zeros(nt)
        Z_hist   = np.zeros(nt)

        for it in range(nt):
            t  = self.t[it]
            dt = self.dt[it]

            # ---- infall ----
            dm_in = self._infall(r, t) * dt
            # infall composition: primordial
            X_in = BBN_ABUNDANCE

            # ---- SFR ----
            sfr = self._sfr(mg)
            dm_sf = min(sfr * dt, 0.9 * mg)
            sfr_hist[it] = sfr

            # ---- metallicity ----
            Z_cur = 1.0 - X[EL_IDX['H']] - X[EL_IDX['He']]
            Z_cur = max(Z_cur, 0.0)
            Z_hist[it] = Z_cur

            # ---- CCSNe (instantaneous) ----
            ycc = ccsne_yields(Z_cur)
            R_cc = self.imf_stats['ccsn_return_fraction']
            enrich_cc = dm_sf * ycc
            ret_cc    = dm_sf * R_cc        # returned gas (unprocessed)

            # ---- Type Ia (DTD convolution) ----
            rate_ia = p['ia_N_per_Msun'] * Mia[it, :it+1] @ sfr_hist[:it+1]
            enrich_ia = rate_ia * y_ia * dt
            ret_ia    = rate_ia * 1.4 * dt  # ~Chandrasekhar mass

            # ---- AGB (25-bin integration, 1.0–8.0 M☉) ----
            # Proper IMF-weighted integration for s-process resolution
            enrich_agb = np.zeros(nel)
            ret_agb = 0.0
            N_AGB_BINS = 25
            m_agb_edges = np.linspace(1.0, 8.0, N_AGB_BINS + 1)
            m_agb_centers = 0.5 * (m_agb_edges[:-1] + m_agb_edges[1:])
            dm_agb = np.diff(m_agb_edges)
            m_imf, phi_n = self._m_imf, self._phi_imf
            for ib in range(N_AGB_BINS):
                m_agb = m_agb_centers[ib]
                source = self._agb_source_state(m_agb, t, it, sfr_hist, Z_hist)
                if source is not None:
                    # IMF weight: ?(m) * ?m (number of stars in this mass bin)
                    phi_at_m = np.interp(m_agb, m_imf, phi_n)
                    weight = phi_at_m * dm_agb[ib]
                    n_agb = source['sfr_form'] * dt * weight * p.get('agb_frequency_multiplier', 1.0)
                    y_a = agb_yields(m_agb, source['Z_form'], p.get('yield_s_multiplier', 1.0))
                    m_rem_a = remnant_mass(np.array([m_agb]))[0]
                    enrich_agb += n_agb * y_a
                    ret_agb    += n_agb * (m_agb - m_rem_a)

            # ---- NSM (DTD convolution) ----
            rate_nsm = p['nsm_N_per_Msun'] * Mnsm[it, :it+1] @ sfr_hist[:it+1]
            enrich_nsm = rate_nsm * y_nsm * dt

            # ---- Collapsar / Jet-SNe r-process (instantaneous, tied to CCSNe) ----
            collapsar_frac = p.get('collapsar_frac', 0.01)
            n_collapsar = dm_sf * self.imf_stats['ccsn_number_per_msun'] * collapsar_frac
            enrich_coll = n_collapsar * y_coll

            # ---- outflow ----
            dm_out = p['outflow_eta'] * dm_sf

            # ---- mass budget ----
            total_enrich = enrich_cc + enrich_ia + enrich_agb + enrich_nsm + enrich_coll
            dm_gas = -dm_sf + ret_cc + ret_ia + ret_agb + dm_in - dm_out
            mg_new = max(mg + dm_gas, 1e-4)

            # ---- abundance update ----
            m_el = mg * X
            # lock-up in stars
            m_el -= dm_sf * X
            # returned unprocessed gas (CCSNe + AGB)
            m_el += ret_cc * X
            m_el += ret_agb * X
            # He enrichment from CCSNe (newly produced He, not returned)
            m_el[EL_IDX['He']] += dm_sf * CCSNE_HE_YIELD
            # newly synthesised metals
            m_el += total_enrich
            # infall
            m_el += dm_in * X_in
            # outflow (proportional to current X)
            m_el -= dm_out * X

            mg = mg_new
            X = np.clip(m_el / mg, 0.0, 1.0)
            # Physical renormalisation: preserve evolved H/He ratio,
            # only rescale to ensure sum(X) = 1 (prevents numerical drift)
            total_X = np.sum(X)
            if total_X > 0:
                X *= 1.0 / total_X
            Z_new = 1.0 - X[EL_IDX['H']] - X[EL_IDX['He']]
            Z_new = max(Z_new, 0.0)

            # Stellar mass: subtract ALL channels' returned mass
            ms += dm_sf - ret_cc - ret_ia - ret_agb

            # ---- store ----
            self.X[:, ir, it]  = X
            self.Mgas[ir, it]  = mg
            self.Mstar[ir, it] = ms
            self.SFR[ir, it]   = sfr
            self.Z_[ir, it]    = Z_new

    # --------------------------------------------------------
    # Package results as dict for JSON serialisation
    # --------------------------------------------------------
    def _results(self):
        idx_H  = EL_IDX['H']
        idx_He = EL_IDX['He']

        # [X/H] = log10(X_i/X_H) - log10(X_i☉/X_H☉)
        XH = {}
        for el in ELEMENTS:
            if el in ('H', 'He'):
                continue
            ie = EL_IDX[el]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.log10(self.X[ie] / np.clip(self.X[idx_H], 1e-30, None))
                solar_ratio = np.log10(SOLAR_X[ie] / SOLAR_X[idx_H])
            xh = ratio - solar_ratio
            XH[el] = np.nan_to_num(xh, nan=-9.0, posinf=0.0, neginf=-9.0)

        ccsn_rate = self.SFR * self.imf_stats['ccsn_number_per_msun']
        return {
            'time': self.t.tolist(),
            'radius': self.r.tolist(),
            'elements': ELEMENTS,
            'metallicity': self.Z_.tolist(),
            'gas_mass': self.Mgas.tolist(),
            'stellar_mass': self.Mstar.tolist(),
            'sfr': self.SFR.tolist(),
            'sn2_rate': (self.SFR / (np.max(self.SFR) + 1e-9)).tolist(),
            'sn2_rate_physical': ccsn_rate.tolist(),
            'imf_stats': self.imf_stats,
            'XH': {el: v.tolist() for el, v in XH.items()},
            'mass_fractions': {
                el: self.X[EL_IDX[el]].tolist() for el in ELEMENTS
            },
        }
