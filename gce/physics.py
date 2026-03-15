"""
Astrophysical functions: IMF, stellar lifetimes, and delay-time distributions.
"""
import numpy as np

_KROUPA_SEGMENTS = (
    (0.01, 0.08, 0.08 ** 1.0 * 0.5 ** 1.0, 0.3),
    (0.08, 0.5, 0.5 ** 1.0, 1.3),
    (0.5, np.inf, 1.0, 2.3),
)


# ============================================================
# Initial Mass Function - Kroupa (2001)
# ============================================================

def _power_law_integral(m_lo, m_hi, alpha, amplitude=1.0, moment=0.0):
    """Integral of amplitude * m^(moment-alpha) from m_lo to m_hi."""
    if m_hi <= m_lo:
        return 0.0
    expo = moment - alpha + 1.0
    if np.isclose(expo, 0.0):
        return float(amplitude * np.log(m_hi / m_lo))
    return float(amplitude * (m_hi ** expo - m_lo ** expo) / expo)


def _kroupa_weight(m_lo, m_hi, moment=0.0):
    total = 0.0
    for seg_lo, seg_hi, amp, alpha in _KROUPA_SEGMENTS:
        lo = max(m_lo, seg_lo)
        hi = min(m_hi, seg_hi)
        if hi <= lo:
            continue
        total += _power_law_integral(lo, hi, alpha, amplitude=amp, moment=moment)
    return total


def kroupa_imf(m):
    """Un-normalized dN/dm for a Kroupa IMF."""
    m = np.atleast_1d(np.float64(m))
    phi = np.zeros_like(m)
    for seg_lo, seg_hi, amp, alpha in _KROUPA_SEGMENTS:
        mask = (m >= seg_lo) & (m < seg_hi)
        phi[mask] = amp * m[mask] ** (-alpha)
    phi[m < _KROUPA_SEGMENTS[0][0]] = 0.0
    return phi


def imf_mass_normalization(m_lo=0.08, m_hi=100.0):
    """Mass integral of the IMF over the requested range."""
    return _kroupa_weight(m_lo, m_hi, moment=1.0)


def imf_norm(m_lo=0.08, m_hi=100.0, n=2000):
    """Return (m_grid, phi_normed) with int(m * phi dm) = 1."""
    m = np.logspace(np.log10(m_lo), np.log10(m_hi), n)
    phi = kroupa_imf(m)
    norm = imf_mass_normalization(m_lo, m_hi)
    return m, phi / norm


def imf_number_per_msun(m_min, m_max, m_lo=0.08, m_hi=100.0):
    """Number of stars in [m_min, m_max] per solar mass formed."""
    lo = max(m_min, m_lo)
    hi = min(m_max, m_hi)
    if hi <= lo:
        return 0.0
    return _kroupa_weight(lo, hi, moment=0.0) / imf_mass_normalization(m_lo, m_hi)


def imf_number_fraction(m_min, m_max, m_lo=0.08, m_hi=100.0):
    """Fraction of stars by number in [m_min, m_max] within the sampled IMF range."""
    total_number = _kroupa_weight(m_lo, m_hi, moment=0.0)
    if total_number <= 0:
        return 0.0
    lo = max(m_min, m_lo)
    hi = min(m_max, m_hi)
    if hi <= lo:
        return 0.0
    return _kroupa_weight(lo, hi, moment=0.0) / total_number


def sample_powerlaw_imf(rng, n, m_lo, m_hi, alpha):
    """Sample dN/dm ~ m^-alpha on [m_lo, m_hi] using inverse CDF."""
    if n <= 0:
        return np.empty(0, dtype=float)
    u = rng.uniform(0.0, 1.0, int(n))
    expo = 1.0 - alpha
    if np.isclose(expo, 0.0):
        return m_lo * (m_hi / m_lo) ** u
    lo_term = m_lo ** expo
    hi_term = m_hi ** expo
    return (lo_term + u * (hi_term - lo_term)) ** (1.0 / expo)


def sample_kroupa_imf(rng, n, m_lo=0.08, m_hi=100.0):
    """Exact inverse-CDF sampling for the truncated Kroupa IMF."""
    if n <= 0:
        return np.empty(0, dtype=float)

    segments = []
    weights = []
    for seg_lo, seg_hi, amp, alpha in _KROUPA_SEGMENTS:
        lo = max(m_lo, seg_lo)
        hi = min(m_hi, seg_hi)
        if hi <= lo:
            continue
        segments.append((lo, hi, alpha))
        weights.append(_power_law_integral(lo, hi, alpha, amplitude=amp, moment=0.0))

    weights = np.asarray(weights, dtype=float)
    probs = weights / np.sum(weights)
    choices = rng.choice(len(segments), size=int(n), p=probs)
    masses = np.empty(int(n), dtype=float)
    for idx, (lo, hi, alpha) in enumerate(segments):
        mask = choices == idx
        if np.any(mask):
            masses[mask] = sample_powerlaw_imf(rng, int(np.sum(mask)), lo, hi, alpha)
    return masses


# ============================================================
# Stellar lifetimes - Raiteri et al. (1996) inspired fit
# ============================================================

def stellar_lifetime(m, Z=0.02):
    """Lifetime in Gyr for a star of mass m (Msun) and metallicity Z."""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    Z = np.asarray(Z, dtype=np.float64)
    z_term = 0.1 * np.log10(np.clip(Z, 1e-6, 0.06) / 0.02)
    log_tau = 10.0 - 3.6 * np.log10(m) + 0.9 * (np.log10(m) ** 2) - z_term
    tau_yr = 10.0 ** log_tau
    return np.clip(tau_yr / 1e9, 3e-3, 1e3)


def dying_mass_at(delay, Z=0.02, m_lo=0.08, m_hi=100.0):
    """Mass of the star that dies after *delay* Gyr (inverse lifetime)."""
    delay = np.atleast_1d(np.float64(delay))
    m_grid = np.logspace(np.log10(m_hi), np.log10(m_lo), 4000)
    tau_grid = stellar_lifetime(m_grid, Z)
    m_dying = np.interp(delay, tau_grid, m_grid)
    return np.clip(m_dying, m_lo, m_hi)


# ============================================================
# Remnant mass and IMF-integrated return fractions
# ============================================================

def remnant_mass(m):
    """Compact remnant mass (Msun) from initial mass m."""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    mr = np.where(
        m < 9,
        0.106 * m + 0.446,
        np.where(m < 25, 1.4, np.where(m < 40, 0.24 * m - 4.6, 0.5 * m)),
    )
    return np.clip(mr, 0.5, m)


def imf_return_fraction(m_min=8.0, m_max=100.0, m_lo=0.08, m_hi=100.0, n=8192):
    """Gas return fraction from stars in [m_min, m_max] per solar mass formed."""
    lo = max(m_min, m_lo)
    hi = min(m_max, m_hi)
    if hi <= lo:
        return 0.0
    m = np.logspace(np.log10(lo), np.log10(hi), n)
    phi = kroupa_imf(m) / imf_mass_normalization(m_lo, m_hi)
    return float(np.trapezoid((m - remnant_mass(m)) * phi, m))


# ============================================================
# Delay-time distributions
# ============================================================

def dtd_powerlaw(t, t_min, slope=-1.0):
    """Generic power-law DTD: DTD(t) ~ t^slope for t >= t_min."""
    t = np.atleast_1d(np.float64(t))
    dtd = np.zeros_like(t, dtype=np.float64)
    mask = t >= t_min
    if np.any(mask):
        dtd[mask] = np.power(t[mask], slope)
    return dtd
