"""Track-provider boundary for precise stellar interpolation."""

from __future__ import annotations

import os
from functools import lru_cache

from .config import DEFAULT_PRECISE_TRACK_PACK, coerce_stellar_model
from .stellar_track_interpolator import interpolate_track_pack, load_track_pack


_PROVIDER_STATS = {
    'precise_hits': 0,
    'fallbacks': 0,
    'reasons': {},
}


def get_precise_track_state(mass, age_gyr, metallicity_z, track_pack_path=None):
    """Return a raw interpolated stellar state or ``None`` when unavailable."""
    track_pack = _load_precise_track_pack(track_pack_path)
    if track_pack is None:
        _register_fallback('track_pack_missing')
        return None

    state, reason = interpolate_track_pack(track_pack, mass, age_gyr, metallicity_z)
    if state is None:
        _register_fallback(reason or 'interpolation_failed')
        return None

    _PROVIDER_STATS['precise_hits'] += 1
    return state


def get_provider_stats(reset=False):
    snapshot = {
        'precise_hits': int(_PROVIDER_STATS['precise_hits']),
        'fallbacks': int(_PROVIDER_STATS['fallbacks']),
        'reasons': dict(_PROVIDER_STATS['reasons']),
    }
    if reset:
        reset_provider_stats()
    return snapshot


def reset_provider_stats():
    _PROVIDER_STATS['precise_hits'] = 0
    _PROVIDER_STATS['fallbacks'] = 0
    _PROVIDER_STATS['reasons'] = {}


def resolve_stellar_model(model=None):
    return coerce_stellar_model(model)


def clear_precise_track_cache():
    _load_precise_track_pack.cache_clear()


@lru_cache(maxsize=4)
def _load_precise_track_pack(track_pack_path=None):
    path = track_pack_path or DEFAULT_PRECISE_TRACK_PACK
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    return load_track_pack(path)


def _register_fallback(reason):
    _PROVIDER_STATS['fallbacks'] += 1
    bucket = _PROVIDER_STATS['reasons']
    bucket[reason] = bucket.get(reason, 0) + 1
