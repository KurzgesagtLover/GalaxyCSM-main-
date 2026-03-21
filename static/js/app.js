/* ============================================================
   Galaxy CSM — Enhanced Three.js + H-R Diagram + Multi-Window
   ============================================================ */
let scene, camera, renderer, starPoints, raycaster, mouse;
const DEFAULT_GCE_T_MAX = 20.0;
const DEFAULT_VIEW_T_MAX = 10000.0;
const DEFAULT_STAR_COUNT = 25000;
let galaxyData = null, tMax = DEFAULT_VIEW_T_MAX, gceTMax = DEFAULT_GCE_T_MAX;
let galaxyCacheId = null;
let currentTime = 13.8, playing = false, playSpeed = 0.05;
let selectedStarId = -1, selectedPlanetIdx = -1;
let cachedStarData = null, cachedTrack = null;
let _detailTimer = null;
let _galaxyRequestToken = 0;

function withCacheQuery(query = '') {
    if (!galaxyCacheId) return query;
    const cachePart = `cache_id=${encodeURIComponent(galaxyCacheId)}`;
    return query ? `${query}&${cachePart}` : cachePart;
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch (err) {
        if (!response.ok) {
            throw new Error(`Request failed (${response.status})`);
        }
        throw new Error('Invalid JSON response');
    }
    if (!response.ok) {
        throw new Error(payload?.error || `Request failed (${response.status})`);
    }
    return payload;
}

function disposeStarField() {
    if (!starPoints) return;
    scene.remove(starPoints);
    starPoints.geometry?.dispose();
    starPoints.material?.dispose();
    starPoints = null;
    _disposeHeatmap();
}

// Log-scale time mapping
// 0~300: 0~20 Gyr (linear)
// 300~1000: 20~10000 Gyr (exponential)
function sliderToTime(v) {
    if (v <= 300) return (v / 300) * 20.0;
    const k = Math.log(10000.0 / 20.0) / 700.0;
    return 20.0 * Math.exp(k * (v - 300));
}

function timeToSlider(t) {
    if (t <= 20.0) return (t / 20.0) * 300;
    const k = Math.log(10000.0 / 20.0) / 700.0;
    return 300 + Math.log(t / 20.0) / k;
}

/* ---- Utility: Human-readable mass ---- */
function fmtMass(kg) {
    if (kg == null) return '—';
    const abs = Math.abs(kg);
    if (abs >= 1e24) return (kg / 1e24).toFixed(2) + ' Yg';
    if (abs >= 1e21) return (kg / 1e21).toFixed(2) + ' Zg';
    if (abs >= 1e18) return (kg / 1e18).toFixed(2) + ' Eg';
    if (abs >= 1e15) return (kg / 1e15).toFixed(2) + ' Pg';
    if (abs >= 1e12) return (kg / 1e12).toFixed(2) + ' Tg';
    if (abs >= 1e9) return (kg / 1e9).toFixed(2) + ' Gt';
    if (abs >= 1e6) return (kg / 1e6).toFixed(2) + ' Mt';
    return kg.toExponential(2) + ' kg';
}

/* ---- Shaders ---- */
const VS = `
  attribute float aSize; attribute float aBirth; attribute float aDeath; attribute vec3 aColor;
  uniform float uTime;
  varying vec3 vColor; varying float vAlpha;
  void main(){
    vColor = aColor;
    float alive = step(aBirth, uTime) * step(uTime, aDeath);
    float fadeIn = smoothstep(aBirth - 0.02, aBirth + 0.08, uTime);
    float fadeOut = 1.0 - smoothstep(aDeath - 0.3, aDeath, uTime);
    vAlpha = alive * fadeIn * fadeOut;
    // Fix 1px rendering bug for filtered stars
    if (aSize < 0.01) { vAlpha = 0.0; } 
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (180.0 / -mv.z) * max(vAlpha, 0.05);
    gl_Position = projectionMatrix * mv;
  }`;
const FS = `
  varying vec3 vColor; varying float vAlpha;
  void main(){
    if(vAlpha < 0.01) discard;
    float d = length(gl_PointCoord - 0.5);
    if(d > 0.5) discard;
    float glow = exp(-d * 5.5);
    gl_FragColor = vec4(vColor * (0.5 + 0.5 * glow), vAlpha * glow);
  }`;

/* ---- Init ---- */
document.addEventListener('DOMContentLoaded', () => {
    setupThree();
    init2DParams();
    _buildParamForms();
});

function setupThree() {
    const w = document.getElementById('canvasWrap');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(55, w.clientWidth / w.clientHeight, 0.1, 250);
    camera.position.set(0, 16, 24);
    camera.lookAt(0, 0, 0);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w.clientWidth, w.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0a0a0a, 1);
    w.appendChild(renderer.domElement);
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.15;
    mouse = new THREE.Vector2();

    let drag = false, px = 0, py = 0, rY = 0, rX = -0.55, dist = 28;
    const updCam = () => {
        camera.position.set(dist * Math.sin(rY) * Math.cos(rX), dist * Math.sin(-rX), dist * Math.cos(rY) * Math.cos(rX));
        camera.lookAt(0, 0, 0);
    };
    w.addEventListener('pointerdown', e => { drag = true; px = e.clientX; py = e.clientY; });
    w.addEventListener('pointermove', e => {
        if (!drag) return;
        rY += (e.clientX - px) * 0.004; rX += (e.clientY - py) * 0.004;
        rX = Math.max(-1.4, Math.min(1.4, rX)); px = e.clientX; py = e.clientY; updCam();
    });
    w.addEventListener('pointerup', () => drag = false);
    w.addEventListener('wheel', e => { dist = Math.max(4, Math.min(90, dist + e.deltaY * 0.02)); updCam(); });
    w.addEventListener('click', onStarClick);
    updCam();
    window.addEventListener('resize', () => {
        camera.aspect = w.clientWidth / w.clientHeight;
        camera.updateProjectionMatrix(); renderer.setSize(w.clientWidth, w.clientHeight);
    });
    document.getElementById('timeSlider').addEventListener('input', e => {
        currentTime = sliderToTime(parseFloat(e.target.value));
        document.getElementById('timeLabel').textContent = currentTime.toFixed(2) + ' Gyr';
        updateUniforms();
        onTimeChange();
    });
    document.getElementById('speedSlider').addEventListener('input', e => {
        // Exponential scale: 0 -> 10^{-4} (~0.0001 Gyr/frame), 30 -> 10^{2} (100 Gyr/frame)
        const v = parseFloat(e.target.value);
        playSpeed = Math.pow(10, (v - 20) / 5.0);
    });
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    if (playing && galaxyData) {
        currentTime += playSpeed;
        if (currentTime > tMax) { currentTime = tMax; playing = false; document.getElementById('btnPlay').textContent = '▶'; }
        document.getElementById('timeSlider').value = timeToSlider(currentTime);
        document.getElementById('timeLabel').textContent = currentTime.toFixed(2) + ' Gyr';
        updateUniforms();
        onTimeChange();
    }
    renderer.render(scene, camera);
}

function updateUniforms() {
    if (starPoints?.material?.uniforms) starPoints.material.uniforms.uTime.value = currentTime;
}

/* ---- Real-time update on time change ---- */
function onTimeChange() {
    if (selectedStarId < 0 || !cachedStarData) return;
    // Client-side interpolation from cached track
    updatePanelFromCache();
    updateHRMarker();
    // Debounced full server re-fetch for planet differentiation
    clearTimeout(_detailTimer);
    _detailTimer = setTimeout(() => loadStarDetail(selectedStarId, true), 300);

    // Also update map modes if radiating or GHZ
    updateColorsAndFilters();
}

function switchGraphMode(mode) {
    currentGraphMode = mode;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.style.color = 'var(--dim)';
        btn.style.borderBottom = 'none';
    });
    const activeBtn = document.getElementById(`tab_${mode}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.style.color = 'var(--acc)';
        activeBtn.style.borderBottom = '2px solid var(--acc)';
    }

    // Update axes labels based on mode
    let xName = '', yName = '';
    if (mode === 'r_process') { xName = 'NSM Yields'; yName = 'NSM Frequency'; }
    if (mode === 's_process') { xName = 'AGB Yields'; yName = 'AGB Frequency'; }
    if (mode === 'ia_sn') { xName = 'Ia SN Yields'; yName = 'Ia SN Frequency'; }
    if (mode === 'galaxy') { xName = 'SFE (Star Formation)'; yName = 'Outflow (Gas Escape)'; }

    document.getElementById('lblX').textContent = xName;
    document.getElementById('lblY').textContent = yName;

    // Show/hide collapsar sub-params
    const collDiv = document.getElementById('collapsarParams');
    if (collDiv) collDiv.style.display = mode === 'r_process' ? 'block' : 'none';

    syncDotToMode();
}

function init2DParams() {
    const area = document.getElementById('paramArea');
    let isDragging = false;

    area.addEventListener('mousedown', e => { isDragging = true; updateFromDot(e); });
    window.addEventListener('mousemove', e => { if (isDragging) updateFromDot(e); });
    window.addEventListener('mouseup', () => { isDragging = false; });
    switchGraphMode('r_process');
}

function valueToFrac(p) {
    if (!p.log) return (p.val - p.min) / (p.max - p.min);
    return (Math.log10(p.val) - Math.log10(p.min)) / (Math.log10(p.max) - Math.log10(p.min));
}

function fracToValue(p, f) {
    if (!p.log) return p.min + f * (p.max - p.min);
    return Math.pow(10, Math.log10(p.min) + f * (Math.log10(p.max) - Math.log10(p.min)));
}

function syncDotToMode() {
    const px = graphModes[currentGraphMode].x;
    const py = graphModes[currentGraphMode].y;
    const fx = valueToFrac(px);
    const fy = valueToFrac(py);
    const dot = document.getElementById('paramDot');
    dot.style.left = (fx * 100) + '%';
    dot.style.top = ((1 - fy) * 100) + '%';

    updateActiveValues(px, py);
}

function updateActiveValues(px, py) {
    const formatOut = (p) => (p.log && p.val < 0.01) ? p.val.toExponential(2) : p.val.toFixed(2);
    document.getElementById('valX').textContent = formatOut(px) + px.unit;
    document.getElementById('valY').textContent = formatOut(py) + py.unit;
}

function updateFromDot(e) {
    const area = document.getElementById('paramArea');
    const rect = area.getBoundingClientRect();
    let nx = (e.clientX - rect.left) / rect.width;
    let ny = (e.clientY - rect.top) / rect.height;
    nx = Math.max(0, Math.min(1, nx));
    ny = Math.max(0, Math.min(1, ny));

    const px = graphModes[currentGraphMode].x;
    const py = graphModes[currentGraphMode].y;

    px.val = fracToValue(px, nx);
    py.val = fracToValue(py, 1 - ny);

    syncDotToMode();
}

/* ---- Complete parameter definitions ---- */
const GEN_PARAMS = [
    { section: '기본 설정' },
    { id: 'stellar_model', label: '항성 진화 엔진', type: 'select', options: [
        { value: 'auto', text: 'auto (정밀 + fallback)' },
        { value: 'precise', text: 'precise (정밀 보간)' },
        { value: 'heuristic', text: 'heuristic (휴리스틱)' }
    ], default: 'auto' },
    { id: 'n_stars', label: '생성할 별 개수', type: 'int', min: 100, max: 200000, step: 1000, default: 25000 },
    { id: 'imf', label: '초기 질량 함수 (IMF)', type: 'select', options: [
        { value: 'kroupa', text: 'Kroupa (2001)' },
        { value: 'salpeter', text: 'Salpeter (1955)' },
        { value: 'chabrier', text: 'Chabrier (2003)' }
    ], default: 'kroupa' },

    { section: '시간 범위' },
    { id: 't_max', label: 'GCE 계산 시간 (Gyr)', type: 'float', min: 0.1, max: 100, step: 0.5, default: 20.0 },
    { id: 'view_t_max', label: '관측 시간 상한 (Gyr)', type: 'float', min: 1, max: 10000, step: 100, default: 10000 },

    { section: '은하 공간 구조' },
    { id: 'r_min', label: '최소 반경 (kpc)', type: 'float', min: 0.1, max: 5, step: 0.1, default: 0.5 },
    { id: 'r_max', label: '최대 반경 (kpc)', type: 'float', min: 5, max: 50, step: 1, default: 20.0 },
    { id: 'dr', label: '반경 해상도 Δr (kpc)', type: 'float', min: 0.1, max: 5, step: 0.1, default: 1.0 },

    { section: '별 형성 (Kennicutt-Schmidt)' },
    { id: 'sfr_efficiency', label: '별 형성 효율 (SFE)', type: 'float', min: 0.01, max: 0.5, step: 0.01, default: 0.08 },
    { id: 'sfr_exponent', label: 'K-S 법칙 지수', type: 'float', min: 1.0, max: 2.5, step: 0.1, default: 1.4 },

    { section: '가스 유입 (이중 지수)' },
    { id: 'infall_tau_thick', label: 'Thick disk τ (Gyr)', type: 'float', min: 0.1, max: 5, step: 0.1, default: 1.0 },
    { id: 'infall_tau_thin', label: 'Thin disk τ (Gyr)', type: 'float', min: 1, max: 20, step: 0.5, default: 7.0 },
    { id: 'infall_sigma_thick0', label: 'Thick disk Σ₀ (M☉/pc²)', type: 'float', min: 10, max: 200, step: 5, default: 55.0 },
    { id: 'infall_sigma_thin0', label: 'Thin disk Σ₀ (M☉/pc²)', type: 'float', min: 50, max: 1000, step: 10, default: 320.0 },
    { id: 'infall_rd', label: 'Disk scale length (kpc)', type: 'float', min: 1, max: 10, step: 0.5, default: 3.5 },

    { section: '유출 (Outflow)' },
    { id: 'outflow_eta', label: 'Mass-loading factor η', type: 'float', min: 0, max: 5, step: 0.1, default: 0.3 },

    { section: '원소 수율 배율' },
    { id: 'yield_r_multiplier', label: 'r-process (NSM) 수율', type: 'float', min: 0.1, max: 10, step: 0.1, default: 1.0, unit: '×' },
    { id: 'yield_s_multiplier', label: 's-process (AGB) 수율', type: 'float', min: 0.1, max: 10, step: 0.1, default: 1.0, unit: '×' },
    { id: 'yield_ia_multiplier', label: 'Type Ia SN 수율', type: 'float', min: 0.1, max: 10, step: 0.1, default: 1.0, unit: '×' },
    { id: 'agb_frequency_multiplier', label: 'AGB 빈도 배율', type: 'float', min: 0.1, max: 10, step: 0.1, default: 1.0, unit: '×' },

    { section: 'Type Ia 초신성' },
    { id: 'ia_N_per_Msun', label: 'Ia 발생률 (/M☉)', type: 'float', min: 0.0005, max: 0.005, step: 0.0001, default: 0.002 },
    { id: 'ia_t_min', label: 'Ia 최소 지연 (Gyr)', type: 'float', min: 0.01, max: 1.0, step: 0.01, default: 0.15 },
    { id: 'ia_dtd_slope', label: 'Ia DTD 기울기', type: 'float', min: -2, max: 0, step: 0.1, default: -1.0 },

    { section: '중성자별 합병 (NSM)' },
    { id: 'nsm_N_per_Msun', label: 'NSM 발생률 (/M☉)', type: 'float', min: 1e-6, max: 1e-3, step: 1e-6, default: 3e-5 },
    { id: 'nsm_t_min', label: 'NSM 최소 지연 (Gyr)', type: 'float', min: 0.001, max: 0.5, step: 0.001, default: 0.01 },
    { id: 'nsm_dtd_slope', label: 'NSM DTD 기울기', type: 'float', min: -2, max: 0, step: 0.1, default: -1.0 },
    { id: 'nsm_ejecta', label: 'NSM 방출 질량 (M☉)', type: 'float', min: 0.001, max: 0.1, step: 0.001, default: 0.03 },

    { section: 'Collapsar / Jet-SNe' },
    { id: 'collapsar_frac', label: 'CCSNe 중 비율', type: 'float', min: 0, max: 0.1, step: 0.005, default: 0.01 },
    { id: 'collapsar_ejecta', label: '방출 질량 (M☉)', type: 'float', min: 0.01, max: 0.2, step: 0.01, default: 0.05 },
];

function _groupParamsBySections() {
    const groups = [];
    let cur = null;
    GEN_PARAMS.forEach(p => {
        if (p.section) {
            cur = { title: p.section, params: [] };
            groups.push(cur);
        } else if (cur) {
            cur.params.push(p);
        }
    });
    return groups;
}

function _renderInputRow(p, prefix) {
    const row = document.createElement('label');
    row.className = 'gen-opt-row';

    const lbl = document.createElement('span');
    lbl.textContent = p.label;
    row.appendChild(lbl);

    if (p.type === 'select') {
        const sel = document.createElement('select');
        sel.id = prefix + p.id;
        sel.className = 'gen-opt-select';
        sel.style.width = '150px';
        p.options.forEach(o => {
            const opt = document.createElement('option');
            opt.value = o.value;
            opt.textContent = o.text;
            if (o.value === p.default) opt.selected = true;
            sel.appendChild(opt);
        });
        row.appendChild(sel);
    } else {
        const inp = document.createElement('input');
        inp.type = 'number';
        inp.id = prefix + p.id;
        inp.className = 'gen-opt-num';
        inp.min = p.min;
        inp.max = p.max;
        inp.step = p.step;
        inp.value = p.default;
        if (p.unit) {
            const wrap = document.createElement('span');
            wrap.style.cssText = 'display:flex;align-items:center;gap:3px';
            wrap.appendChild(inp);
            const u = document.createElement('span');
            u.style.cssText = 'font-size:10px;color:var(--dim);width:16px';
            u.textContent = p.unit;
            wrap.appendChild(u);
            row.appendChild(wrap);
        } else {
            row.appendChild(inp);
        }
    }
    return row;
}

function _renderGrid(container, prefix) {
    container.innerHTML = '';
    const groups = _groupParamsBySections();
    groups.forEach(g => {
        const card = document.createElement('div');
        card.className = 'setup-card';

        const title = document.createElement('div');
        title.className = 'setup-card-title';
        title.textContent = g.title;
        card.appendChild(title);

        g.params.forEach(p => card.appendChild(_renderInputRow(p, prefix)));
        container.appendChild(card);
    });
}

function _buildParamForms() {
    _renderGrid(document.getElementById('setupGrid'), 'opt_');
}

function _syncGrids(from, to) {
    GEN_PARAMS.forEach(p => {
        if (p.section) return;
        const src = document.getElementById(from + p.id);
        const dst = document.getElementById(to + p.id);
        if (src && dst) dst.value = src.value;
    });
}

function onSetupGenerate() {
    document.getElementById('setupScreen').classList.add('hidden');
    _renderGrid(document.getElementById('genOptsGrid'), 'ropt_');
    _syncGrids('opt_', 'ropt_');
    _doGalaxyGeneration(collectGenOpts('opt_'));
}

function onRegenerate() {
    document.getElementById('genOptsOverlay').classList.add('hidden');
    _doGalaxyGeneration(collectGenOpts('ropt_'));
}

function collectGenOpts(prefix) {
    prefix = prefix || 'opt_';
    const result = {};
    GEN_PARAMS.forEach(p => {
        if (p.section) return;
        const el = document.getElementById(prefix + p.id);
        if (!el) return;
        if (p.type === 'select') {
            result[p.id] = el.value;
        } else if (p.type === 'int') {
            result[p.id] = parseInt(el.value) || p.default;
        } else {
            result[p.id] = parseFloat(el.value) || p.default;
        }
    });
    return result;
}

function apply2DParams() {
    const opts = collectGenOpts('ropt_');
    if (typeof graphModes !== 'undefined') {
        Object.values(graphModes).forEach(mode => {
            opts[mode.x.id] = mode.x.val;
            opts[mode.y.id] = mode.y.val;
        });
    }
    const cfEl = document.getElementById('collapsar_frac');
    if (cfEl) opts.collapsar_frac = parseFloat(cfEl.value);
    const ceEl = document.getElementById('collapsar_ejecta');
    if (ceEl) opts.collapsar_ejecta = parseFloat(ceEl.value);
    _doGalaxyGeneration(opts);
}

function _doGalaxyGeneration(params) {
    document.getElementById('setupScreen')?.classList.add('hidden');
    document.getElementById('statusTxt').textContent = 'Generating...';
    document.getElementById('overlay').classList.add('show');
    const requestToken = ++_galaxyRequestToken;

    fetchJson('/api/galaxy', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    }).then(data => {
        if (requestToken !== _galaxyRequestToken) return;
        galaxyData = data;
        galaxyCacheId = galaxyData.cache_id || null;
        gceTMax = galaxyData.t_max || DEFAULT_GCE_T_MAX;
        tMax = galaxyData.view_t_max || tMax || DEFAULT_VIEW_T_MAX;
        currentTime = Math.min(currentTime, tMax);
        buildStarField();
        document.getElementById('starCount').textContent = galaxyData.n_stars.toLocaleString() + ' stars';
        document.getElementById('statusTxt').textContent = 'Ready (' + galaxyData.elapsed + 's)';
        document.getElementById('timeSlider').value = timeToSlider(currentTime);
        document.getElementById('timeLabel').textContent = currentTime.toFixed(2) + ' Gyr';

        if (!document.getElementById('gceWindow').classList.contains('hidden')) openWindow('gce');
    }).catch(e => {
        if (requestToken !== _galaxyRequestToken) return;
        document.getElementById('statusTxt').textContent = e.message || 'Error';
        console.error(e);
    }).finally(() => {
        if (requestToken === _galaxyRequestToken) {
            document.getElementById('overlay').classList.remove('show');
        }
    });
}

function togglePlay() {
    playing = !playing;
    document.getElementById('btnPlay').textContent = playing ? '⏸' : '▶';
    if (playing && currentTime >= tMax) currentTime = 0;
}

let origColor, origSize;

/* ---- Heatmap Field Overlay ---- */
let heatmapMesh = null, _hmCanvas = null, _hmCtx = null, _hmTexture = null;
const HM_SIZE = 512;
const HM_RADIUS_KPC = 22;

function _disposeHeatmap() {
    if (!heatmapMesh) return;
    scene.remove(heatmapMesh);
    heatmapMesh.geometry?.dispose();
    heatmapMesh.material?.dispose();
    if (_hmTexture) _hmTexture.dispose();
    heatmapMesh = null; _hmTexture = null;
}

function _buildHeatmapDisk() {
    _disposeHeatmap();
    _hmCanvas = document.createElement('canvas');
    _hmCanvas.width = HM_SIZE; _hmCanvas.height = HM_SIZE;
    _hmCtx = _hmCanvas.getContext('2d');

    _hmTexture = new THREE.CanvasTexture(_hmCanvas);
    _hmTexture.minFilter = THREE.LinearFilter;
    _hmTexture.magFilter = THREE.LinearFilter;

    const geo = new THREE.PlaneGeometry(HM_RADIUS_KPC * 2, HM_RADIUS_KPC * 2);
    const mat = new THREE.MeshBasicMaterial({
        map: _hmTexture, transparent: true,
        depthWrite: false, side: THREE.DoubleSide,
    });
    heatmapMesh = new THREE.Mesh(geo, mat);
    heatmapMesh.rotation.x = -Math.PI / 2;
    heatmapMesh.position.y = -0.08;
    heatmapMesh.renderOrder = -1;
    heatmapMesh.visible = false;
    scene.add(heatmapMesh);
}

function _lerpZone(vals, radii, distKpc) {
    const n = radii.length;
    if (distKpc <= radii[0]) return vals[0];
    if (distKpc >= radii[n - 1]) return vals[n - 1];
    for (let j = 1; j < n; j++) {
        if (distKpc <= radii[j]) {
            const t = (distKpc - radii[j - 1]) / (radii[j] - radii[j - 1]);
            return vals[j - 1] + t * (vals[j] - vals[j - 1]);
        }
    }
    return vals[n - 1];
}

function _updateHeatmapTexture() {
    if (!heatmapMesh || !galaxyData) return;
    const mode = document.getElementById('mapModeSelect').value;
    if (mode === 'spectrum') { heatmapMesh.visible = false; return; }
    heatmapMesh.visible = true;

    const gce = galaxyData.gce;
    const radii = gce.radius;
    const nZ = radii.length;
    const rMax = radii[nZ - 1];

    let it = 0;
    for (let i = 0; i < gce.time.length; i++) { if (gce.time[i] <= currentTime) it = i; }

    const zv = new Float32Array(nZ);
    const zv2 = (mode === 'ghz') ? new Float32Array(nZ) : null;

    for (let iz = 0; iz < nZ; iz++) {
        if (mode === 'radiation') {
            zv[iz] = gce.sn2_rate[iz][it];
        } else if (mode === 'metallicity') {
            zv[iz] = Math.log10(Math.max(gce.metallicity[iz][it], 1e-10) / 0.0134);
        } else if (mode === 'ghz') {
            const met = Math.log10(Math.max(gce.metallicity[iz][it], 1e-10) / 0.0134);
            const rad = gce.sn2_rate[iz][it];
            zv[iz] = Math.max(0, Math.min(1, (met + 1.0) / 0.8))
                   * Math.max(0, Math.min(1, (0.3 - rad) / 0.25));
        }
    }

    const W = HM_SIZE, H = HM_SIZE, cx = W / 2, cy = H / 2;
    const imgData = _hmCtx.createImageData(W, H);
    const d = imgData.data;

    for (let py = 0; py < H; py++) {
        for (let px = 0; px < W; px++) {
            const dx = px - cx, dy = py - cy;
            const distPx = Math.sqrt(dx * dx + dy * dy);
            const distKpc = (distPx / cx) * HM_RADIUS_KPC;

            const val = _lerpZone(zv, radii, distKpc);
            const edge = 1.0 - Math.pow(Math.min(distKpc / (rMax + 3), 1.0), 3.5);

            let r, g, b, a;
            if (mode === 'radiation') {
                const I = Math.pow(Math.max(0, Math.min(val, 1.0)), 0.55);
                r = (0.10 + I * 0.90) * 255;
                g = I * 0.95 * 255;
                b = (0.30 + I * 0.70) * 255;
                a = (0.08 + I * 0.38) * edge * 255;
            } else if (mode === 'metallicity') {
                const f = Math.max(0, Math.min(1, (val + 1.0) / 1.5));
                r = f * 255;
                g = (0.12 + 0.15 * (1 - Math.abs(f - 0.5) * 2)) * 255;
                b = (1.0 - f) * 255;
                a = (0.08 + 0.32 * (0.3 + 0.7 * f)) * edge * 255;
            } else {
                const v = Math.max(0, Math.min(1, val));
                r = (1 - v) * 55;
                g = v * 210 + (1 - v) * 28;
                b = v * 70 + (1 - v) * 28;
                a = (0.05 + 0.28 * v) * edge * 255;
            }

            const idx = (py * W + px) * 4;
            d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = a;
        }
    }
    _hmCtx.putImageData(imgData, 0, 0);
    _hmTexture.needsUpdate = true;
}

function buildStarField() {
    const s = galaxyData.stars, n = s.x.length;
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const sz = new Float32Array(n), bth = new Float32Array(n), dth = new Float32Array(n);

    // Store original sizes and colors for filtering and map mode restoring
    origColor = new Float32Array(n * 3);
    origSize = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        pos[i * 3] = s.x[i]; pos[i * 3 + 1] = s.z[i]; pos[i * 3 + 2] = s.y[i];
        const hx = s.color[i];
        const r = parseInt(hx.slice(1, 3), 16) / 255;
        const g_col = parseInt(hx.slice(3, 5), 16) / 255;
        const b = parseInt(hx.slice(5, 7), 16) / 255;
        col[i * 3] = r; col[i * 3 + 1] = g_col; col[i * 3 + 2] = b;
        origColor[i * 3] = r; origColor[i * 3 + 1] = g_col; origColor[i * 3 + 2] = b;

        sz[i] = s.size[i];
        origSize[i] = s.size[i];

        bth[i] = s.birth[i]; dth[i] = s.death[i];
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(sz, 1));
    g.setAttribute('aBirth', new THREE.BufferAttribute(bth, 1));
    g.setAttribute('aDeath', new THREE.BufferAttribute(dth, 1));

    const mat = new THREE.ShaderMaterial({
        vertexShader: VS, fragmentShader: FS,
        uniforms: { uTime: { value: currentTime } },
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    disposeStarField();
    starPoints = new THREE.Points(g, mat);
    scene.add(starPoints);

    _buildHeatmapDisk();
    updateColorsAndFilters(); // initial state
}

/* ---- Map Modes & Filtering ---- */
function updateColorsAndFilters() {
    if (!starPoints || !galaxyData) return;

    const mode = document.getElementById('mapModeSelect').value;
    const s = galaxyData.stars;
    const gce = galaxyData.gce;
    const n = s.x.length;

    const colAttr = starPoints.geometry.attributes.aColor;
    const szAttr = starPoints.geometry.attributes.aSize;

    // Get filter states
    const selSp = Array.from(document.querySelectorAll('.flt-sp:checked')).map(el => el.value);
    const selPh = Array.from(document.querySelectorAll('.flt-ph:checked')).map(el => el.value);
    const fp = document.getElementById('flt_has_p').checked;
    const fr = document.getElementById('flt_has_rocky').checked;
    const fg = document.getElementById('flt_has_gas').checked;
    const fhz = document.getElementById('flt_has_hz').checked;

    // For radiation mode, get time index
    let it = 0;
    if (mode === 'radiation' || mode === 'ghz') {
        const tGrid = gce.time;
        for (let i = 0; i < tGrid.length; i++) {
            if (tGrid[i] <= currentTime) it = i;
        }
    }

    let aliveCount = 0;
    for (let i = 0; i < n; i++) {
        // --- 1. Map Mode Coloring ---
        let r = origColor[i * 3], g_col = origColor[i * 3 + 1], b = origColor[i * 3 + 2];
        const met = s.met[i];
        const rad = (mode === 'radiation' || mode === 'ghz') ? gce.sn2_rate[s.r_zone[i]][it] : 0;

        if (mode === 'radiation') {
            // High rad -> bright cyan/white, Low rad -> dark purple
            const intensity = Math.pow(Math.min(rad, 1.0), 0.6); // gamma correction for visual glow
            r = 0.2 + intensity * 0.8;
            g_col = 0.0 + intensity * 1.0;
            b = 0.5 + intensity * 0.5;
        } else if (mode === 'metallicity') {
            // [Fe/H]: -1.0 (blue) to +0.5 (red)
            const f = Math.max(0, Math.min(1, (met + 1.0) / 1.5));
            r = f;
            g_col = 0.2;
            b = 1.0 - f;
        } else if (mode === 'ghz') {
            // GHZ: sufficient metallicity (met > -0.5) and low radiation
            const inGHZ = (met > -0.5) && (rad < 0.15);
            if (inGHZ) { r = 0; g_col = 1; b = 0; } // Green
            else { r = 0.3; g_col = 0.3; b = 0.3; } // Gray
        }

        colAttr.array[i * 3] = r; colAttr.array[i * 3 + 1] = g_col; colAttr.array[i * 3 + 2] = b;

        // --- 2. Filtering ---
        let visible = true;

        const age = Math.max(currentTime - s.birth[i], 0);
        const t_ms = s.ms_lifetime[i];
        const usingCurrentPhase = Array.isArray(s.phase_bucket) && Math.abs(currentTime - (galaxyData?.t_max ?? currentTime)) < 0.05;
        let phase = 'dead';
        if (currentTime < s.birth[i]) phase = 'unborn';  // not yet born → invisible
        else if (usingCurrentPhase) phase = s.phase_bucket[i] || 'dead';
        else if (age < t_ms * 0.05) phase = 'pre-MS';
        else if (age < t_ms) phase = 'MS';
        else if (age < t_ms * 1.25) phase = 'giant';

        // Apply remnant visual overrides if mode is spectrum and the star is dead
        if (phase === 'dead' && mode === 'spectrum') {
            const m = s.mass[i];
            if (m < 8) {
                // White Dwarf (blue-white, fading to dark grey if Black Dwarf)
                if (age > t_ms * 1.25 + 10.0) { // Approx 10 Gyr cooling = Black Dwarf
                    r = 0.1; g_col = 0.1; b = 0.1;
                } else {
                    r = 0.8; g_col = 0.9; b = 1.0;
                }
            } else if (m < 25) {
                // Neutron Star (faint blue)
                r = 0.5; g_col = 0.8; b = 1.0;
            } else {
                // Black Hole (black/invisible)
                r = 0.0; g_col = 0.0; b = 0.0;
            }
        }

        colAttr.array[i * 3] = r; colAttr.array[i * 3 + 1] = g_col; colAttr.array[i * 3 + 2] = b;

        if (!selPh.includes(phase)) visible = false;

        // Strict ZAMS Base Spectral Filtering
        if (!selSp.includes(s.type[i])) visible = false;

        // Planet Filters
        if (fp && s.has_planets[i] === 0) visible = false;
        if (fr && s.has_rocky[i] === 0) visible = false;
        if (fg && s.has_gas[i] === 0) visible = false;
        if (fhz && s.has_hz[i] === 0) visible = false;

        if (visible) {
            if (phase === 'dead') {
                const m = s.mass[i];
                if (m < 8) szAttr.array[i] = origSize[i] * 0.3;      // WD smaller
                else if (m < 25) szAttr.array[i] = origSize[i] * 0.1; // NS much smaller
                else szAttr.array[i] = 0.0;                           // BH invisible in regular view
            } else {
                szAttr.array[i] = origSize[i];
            }
            if (currentTime >= s.birth[i] && currentTime <= s.death[i]) aliveCount++;
        } else {
            szAttr.array[i] = 0; // hide
        }
    }

    colAttr.needsUpdate = true;
    szAttr.needsUpdate = true;
    document.getElementById('aliveCount').textContent = aliveCount.toLocaleString() + ' visible';

    _updateHeatmapTexture();
}

// Add event listeners to filters to trigger update
document.querySelectorAll('.flt-sp, .flt-ph, #flt_has_p, #flt_has_rocky, #flt_has_gas, #flt_has_hz').forEach(el => {
    el.addEventListener('change', updateColorsAndFilters);
});

/* ---- Star Click ---- */
function onStarClick(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    // Scale hitbox threshold dynamically based on visible star density
    const aliveText = document.getElementById('aliveCount').textContent;
    const visibleCount = parseInt(aliveText.replace(/[^0-9]/g, '')) || 500000;

    if (visibleCount < 500) raycaster.params.Points.threshold = 1.5;
    else if (visibleCount < 5000) raycaster.params.Points.threshold = 0.8;
    else if (visibleCount < 50000) raycaster.params.Points.threshold = 0.3;
    else raycaster.params.Points.threshold = 0.15;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(starPoints);

    // Only accept hits from stars that are structurally visible and alive
    const validHit = hits.find(h => {
        const id = h.index;
        const b = galaxyData.stars.birth[id], d = galaxyData.stars.death[id];
        const sz = starPoints.geometry.attributes.aSize.array[id];
        return sz > 0.01 && currentTime >= b && currentTime <= d;
    });

    if (validHit) {
        loadStarDetail(validHit.index);
    }
}

async 