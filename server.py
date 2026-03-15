"""Flask API for Galaxy CSM — Enhanced.
Supports galaxy generation, star detail with H-R evolution,
planet differentiation, and future simulation extension.
"""
from collections import OrderedDict
import math
from secrets import token_urlsafe
from threading import Lock
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from gce.stellar import generate_galaxy, stellar_evolution, hr_track, _ms_lifetime
from gce.planets import build_planet_system
from gce.config import DEFAULT_PARAMS, DEFAULT_VIEW_T_MAX

app = Flask(__name__, static_folder='static')
CORS(app)
_cache_lock = Lock()
_cache_entries = OrderedDict()
_latest_cache_id = None
_MAX_CACHE_ENTRIES = 4
DEFAULT_STAR_COUNT = 150000
MAX_STAR_COUNT = 200000


def _coerce_float(value, name, min_value=None, max_value=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return parsed


def _coerce_int(value, name, min_value=None, max_value=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return parsed


def _parse_query_float(name, *, default=None, min_value=None, max_value=None):
    raw = request.args.get(name)
    if raw is None:
        return default
    return _coerce_float(raw, name, min_value=min_value, max_value=max_value)


def _store_cache(gce, stars):
    global _latest_cache_id

    cache_id = token_urlsafe(9)
    with _cache_lock:
        _cache_entries[cache_id] = {
            'gce': gce,
            'stars': stars,
            'created_at': time.time(),
        }
        _cache_entries.move_to_end(cache_id)
        _latest_cache_id = cache_id
        while len(_cache_entries) > _MAX_CACHE_ENTRIES:
            _cache_entries.popitem(last=False)
    return cache_id


def _resolve_cache():
    requested_id = request.args.get('cache_id')

    with _cache_lock:
        cache_id = requested_id or _latest_cache_id
        if cache_id is None:
            return None, None

        entry = _cache_entries.get(cache_id)
        if entry is None:
            return cache_id, None

        _cache_entries.move_to_end(cache_id)
        return cache_id, entry

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/defaults')
def defaults():
    return jsonify({
        'n_stars': DEFAULT_STAR_COUNT,
        **DEFAULT_PARAMS,
        'view_t_max': DEFAULT_VIEW_T_MAX,
    })

@app.route('/api/galaxy', methods=['POST'])
def api_galaxy():
    try:
        payload = request.get_json(silent=True)
        params = dict(payload or {})
        n = _coerce_int(params.pop('n_stars', DEFAULT_STAR_COUNT), 'n_stars', min_value=1, max_value=MAX_STAR_COUNT)
        t_max = _coerce_float(params.pop('t_max', DEFAULT_PARAMS['t_max']), 't_max', min_value=0.1, max_value=DEFAULT_VIEW_T_MAX)
        view_t_max = max(
            _coerce_float(params.pop('view_t_max', DEFAULT_VIEW_T_MAX), 'view_t_max', min_value=0.1, max_value=DEFAULT_VIEW_T_MAX),
            t_max,
        )
        params['t_max'] = t_max
        t0 = time.time()
        data = generate_galaxy(n_stars=n, params=params)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    data['elapsed'] = round(time.time() - t0, 2)
    data['view_t_max'] = view_t_max
    data['cache_id'] = _store_cache(data['gce'], data['stars'])
    # Do not delete data['gce'] so that client has radiation modes
    return jsonify(data)

@app.route('/api/star/<int:star_id>')
def api_star(star_id):
    cache_id, cache = _resolve_cache()
    if cache is None:
        if cache_id is not None:
            return jsonify({'error': 'Unknown cache_id. Run galaxy again.'}), 404
        return jsonify({'error': 'Run galaxy first'}), 400

    s = cache['stars']
    if star_id < 0 or star_id >= len(s['mass']):
        return jsonify({'error': 'Invalid star ID'}), 404
    try:
        ct = _parse_query_float('t', default=13.8, min_value=0.0, max_value=DEFAULT_VIEW_T_MAX)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    mass = s['mass'][star_id]
    birth = s['birth'][star_id]
    birth_z = float(s.get('birth_z', [0.02] * len(s['mass']))[star_id])
    age = max(ct - birth, 0)
    t_ms = _ms_lifetime(mass, metallicity_z=birth_z)

    evo = stellar_evolution(mass, age, metallicity_z=birth_z)

    # Optional planet physics parameters from query string
    esi_w = {}
    for k in ('w_radius', 'w_density', 'w_escape', 'w_temp'):
        try:
            v = _parse_query_float(k, default=None, min_value=0.0, max_value=10.0)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        if v is not None:
            esi_w[k] = v
    try:
        lv_frac = _parse_query_float('lv_frac', default=None, min_value=0.0, max_value=0.1)
        disprop_sc = _parse_query_float('disprop_scale', default=None, min_value=0.0, max_value=5.0)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    result = build_planet_system(
        star_id=star_id, star_mass=mass,
        birth_time=birth, r_zone=s['r_zone'][star_id],
        gce_result=cache['gce'], current_time=ct,
        evo=evo,
        esi_weights=esi_w if esi_w else None,
        lv_frac=lv_frac,
        disprop_scale=disprop_sc
    )
    result['cache_id'] = cache_id
    result['evolution'] = evo
    result['star_type'] = s['type'][star_id]
    result['star_temp'] = s['temp'][star_id]
    result['star_color'] = evo['color']
    result['star_luminosity'] = evo['luminosity']
    result['age_gyr'] = round(age, 3)
    result['birth_z'] = round(birth_z, 6)
    result['ms_lifetime'] = round(t_ms, 4)
    result['remaining_ms'] = round(max(t_ms - age, 0), 4)
    result['total_lifetime'] = round(t_ms * 1.3, 4)
    return jsonify(result)

@app.route('/api/evolution/<int:star_id>')
def api_evolution(star_id):
    """Get full H-R evolution track for a star."""
    cache_id, cache = _resolve_cache()
    if cache is None:
        if cache_id is not None:
            return jsonify({'error': 'Unknown cache_id. Run galaxy again.'}), 404
        return jsonify({'error': 'Run galaxy first'}), 400

    s = cache['stars']
    if star_id < 0 or star_id >= len(s['mass']):
        return jsonify({'error': 'Invalid star ID'}), 404
    mass = s['mass'][star_id]
    birth = s['birth'][star_id]
    birth_z = float(s.get('birth_z', [0.02] * len(s['mass']))[star_id])
    try:
        t_max = _parse_query_float('t_max', default=DEFAULT_VIEW_T_MAX, min_value=0.1, max_value=DEFAULT_VIEW_T_MAX)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    age_max = max(t_max - birth, 0.0)
    track = hr_track(mass, age_max=age_max, metallicity_z=birth_z)
    # Add absolute time
    for pt in track:
        pt['t'] = round(birth + pt['age'], 5)
    return jsonify({
        'cache_id': cache_id,
        'star_id': star_id, 'mass': round(mass, 4),
        'birth': round(birth, 3),
        'birth_z': round(birth_z, 6),
        'ms_lifetime': round(_ms_lifetime(mass, metallicity_z=birth_z), 4),
        'track': track
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
