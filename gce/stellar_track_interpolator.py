"""Helpers for loading and interpolating precomputed stellar tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


SCHEMA_VERSION = 1
LOG_FIELDS = ('luminosity', 'temperature', 'radius_rsun', 'max_radius_rsun')
LINEAR_FIELDS = ('current_mass', 'flare_activity')
REQUIRED_TRACK_FIELDS = (
    'ages_gyr',
    'phases',
    'phase_fraction',
    'luminosity',
    'T_eff',
    'radius_rsun',
    'current_mass',
    'max_radius_rsun',
    'flare_activity',
)


@dataclass(frozen=True)
class TrackPack:
    path: str
    mass_grid: tuple[float, ...]
    metallicity_grid: tuple[float, ...]
    phase_order: tuple[str, ...]
    tracks: dict[tuple[float, float], "PreparedTrack"]
    metadata: dict


class PreparedTrack:
    """Normalized track arrays ready for fast phase-aware sampling."""

    def __init__(self, initial_mass, metallicity_z, payload):
        self.initial_mass = float(initial_mass)
        self.metallicity_z = float(metallicity_z)
        self.ages = np.asarray(payload['ages_gyr'], dtype=float)
        self.phases = np.asarray(payload['phases'], dtype=object)
        self.phase_fraction = np.asarray(payload['phase_fraction'], dtype=float)
        self.luminosity = np.asarray(payload['luminosity'], dtype=float)
        self.temperature = np.asarray(payload['T_eff'], dtype=float)
        self.radius_rsun = np.asarray(payload['radius_rsun'], dtype=float)
        self.current_mass = np.asarray(payload['current_mass'], dtype=float)
        self.max_radius_rsun = np.asarray(payload['max_radius_rsun'], dtype=float)
        self.flare_activity = np.asarray(payload['flare_activity'], dtype=float)

        expected = len(self.ages)
        if expected == 0:
            raise ValueError('stellar track must contain at least one point')
        for field in REQUIRED_TRACK_FIELDS[1:]:
            values = np.asarray(payload[field], dtype=object if field == 'phases' else float)
            if len(values) != expected:
                raise ValueError(f"stellar track field '{field}' length mismatch")
        if np.any(np.diff(self.ages) < 0):
            raise ValueError('stellar track ages must be sorted in ascending order')

    def sample(self, age_gyr):
        age = float(np.clip(age_gyr, self.ages[0], self.ages[-1]))
        hi = int(np.searchsorted(self.ages, age, side='left'))
        if hi <= 0:
            return self._point(0)
        if hi >= len(self.ages):
            return self._point(len(self.ages) - 1)
        lo = hi - 1
        if np.isclose(age, self.ages[lo]):
            return self._point(lo)
        if np.isclose(age, self.ages[hi]):
            return self._point(hi)
        if self.phases[lo] != self.phases[hi]:
            return self._point(lo if (age - self.ages[lo]) <= (self.ages[hi] - age) else hi)

        span = max(self.ages[hi] - self.ages[lo], 1e-12)
        weight = float((age - self.ages[lo]) / span)
        return {
            'phase': str(self.phases[lo]),
            'temperature': _interp_value(self.temperature[lo], self.temperature[hi], weight, use_log=True),
            'luminosity': _interp_value(self.luminosity[lo], self.luminosity[hi], weight, use_log=True),
            'radius_rsun': _interp_value(self.radius_rsun[lo], self.radius_rsun[hi], weight, use_log=True),
            'current_mass': _interp_value(self.current_mass[lo], self.current_mass[hi], weight, use_log=False),
            'max_radius_rsun': max(
                _interp_value(self.max_radius_rsun[lo], self.max_radius_rsun[hi], weight, use_log=True),
                _interp_value(self.radius_rsun[lo], self.radius_rsun[hi], weight, use_log=True),
            ),
            'flare_activity': _interp_value(self.flare_activity[lo], self.flare_activity[hi], weight, use_log=False),
            'age_gyr': age,
        }

    def _point(self, idx):
        return {
            'phase': str(self.phases[idx]),
            'temperature': float(self.temperature[idx]),
            'luminosity': float(self.luminosity[idx]),
            'radius_rsun': float(self.radius_rsun[idx]),
            'current_mass': float(self.current_mass[idx]),
            'max_radius_rsun': float(max(self.max_radius_rsun[idx], self.radius_rsun[idx])),
            'flare_activity': float(self.flare_activity[idx]),
            'age_gyr': float(self.ages[idx]),
        }


def load_track_pack(path):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    schema_version = int(payload.get('schema_version', 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported stellar track schema version {schema_version}, expected {SCHEMA_VERSION}"
        )

    track_items = {}
    for item in payload.get('tracks', []):
        mass = float(item['initial_mass'])
        metallicity = float(item['metallicity_z'])
        track_items[(mass, metallicity)] = PreparedTrack(mass, metallicity, item)

    if not track_items:
        raise ValueError('stellar track pack did not contain any tracks')

    return TrackPack(
        path=path,
        mass_grid=tuple(sorted({mass for mass, _ in track_items.keys()})),
        metallicity_grid=tuple(sorted({metallicity for _, metallicity in track_items.keys()})),
        phase_order=tuple(payload.get('phase_order', [])),
        tracks=track_items,
        metadata={
            'coordinate_system': payload.get('coordinate_system', 'phase_fraction_v1'),
            'source': payload.get('source'),
            'generated_by': payload.get('generated_by'),
        },
    )


def interpolate_track_pack(track_pack, mass, age_gyr, metallicity_z):
    mass_bracket = _bracket_axis(track_pack.mass_grid, float(mass))
    if mass_bracket is None:
        return None, 'mass_out_of_range'
    metallicity_bracket = _bracket_axis(track_pack.metallicity_grid, float(metallicity_z))
    if metallicity_bracket is None:
        return None, 'metallicity_out_of_range'

    samples = []
    for m_val, m_weight in mass_bracket:
        for z_val, z_weight in metallicity_bracket:
            track = track_pack.tracks.get((float(m_val), float(z_val)))
            if track is None:
                continue
            weight = float(m_weight * z_weight)
            samples.append((weight, track.sample(age_gyr)))

    if not samples:
        return None, 'missing_neighbor_track'

    total_weight = float(sum(weight for weight, _ in samples))
    if total_weight <= 0:
        return None, 'zero_interpolation_weight'
    samples = [(weight / total_weight, sample) for weight, sample in samples]

    phase_weights = {}
    for weight, sample in samples:
        phase_weights[sample['phase']] = phase_weights.get(sample['phase'], 0.0) + weight
    dominant_phase, dominant_weight = max(phase_weights.items(), key=lambda item: item[1])

    same_phase = [(weight, sample) for weight, sample in samples if sample['phase'] == dominant_phase]
    if same_phase and sum(weight for weight, _ in same_phase) >= 0.35:
        active = _normalize_weights(same_phase)
    else:
        active = samples
        dominant_phase = max(samples, key=lambda item: item[0])[1]['phase']
        dominant_weight = phase_weights.get(dominant_phase, 0.0)

    state = {
        'phase': dominant_phase,
        'temperature': _blend_field(active, 'temperature', use_log=True),
        'luminosity': _blend_field(active, 'luminosity', use_log=True),
        'radius_rsun': _blend_field(active, 'radius_rsun', use_log=True),
        'current_mass': _blend_field(active, 'current_mass', use_log=False),
        'max_radius_rsun': _blend_field(active, 'max_radius_rsun', use_log=True),
        'flare_activity': _blend_field(active, 'flare_activity', use_log=False),
        'provider_confidence': dominant_weight,
    }
    state['max_radius_rsun'] = max(state['max_radius_rsun'], state['radius_rsun'])

    if len(phase_weights) > 2 and dominant_weight < 0.45:
        return None, 'phase_conflict'
    return state, None


def _blend_field(weighted_samples, field, use_log):
    if use_log:
        logs = []
        for weight, sample in weighted_samples:
            logs.append(weight * np.log10(max(float(sample[field]), 1e-15)))
        return float(10 ** np.sum(logs))
    return float(sum(weight * float(sample[field]) for weight, sample in weighted_samples))


def _interp_value(lo, hi, weight, use_log):
    lo = float(lo)
    hi = float(hi)
    weight = float(np.clip(weight, 0.0, 1.0))
    if use_log:
        lo = np.log10(max(lo, 1e-15))
        hi = np.log10(max(hi, 1e-15))
        return float(10 ** ((1.0 - weight) * lo + weight * hi))
    return float((1.0 - weight) * lo + weight * hi)


def _normalize_weights(weighted_samples):
    total = float(sum(weight for weight, _ in weighted_samples))
    if total <= 0:
        return weighted_samples
    return [(weight / total, sample) for weight, sample in weighted_samples]


def _bracket_axis(values, target):
    arr = np.asarray(values, dtype=float)
    if target < arr[0] or target > arr[-1]:
        return None
    idx = int(np.searchsorted(arr, target, side='left'))
    if idx < len(arr) and np.isclose(arr[idx], target):
        return [(float(arr[idx]), 1.0)]
    if idx == 0:
        return [(float(arr[0]), 1.0)]
    if idx >= len(arr):
        return [(float(arr[-1]), 1.0)]
    lo = float(arr[idx - 1])
    hi = float(arr[idx])
    frac = (target - lo) / max(hi - lo, 1e-12)
    return [(lo, 1.0 - frac), (hi, frac)]
