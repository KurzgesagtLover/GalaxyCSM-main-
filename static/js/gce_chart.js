/* ============================================================
   GCE Chart — Plotly-based chemical evolution visualisation
   ============================================================ */

const GCE_ELEMENT_COLORS = {
    H:  '#bbbbbb', He: '#e8e8e8',
    C:  '#ff6b6b', N:  '#51cf66', O:  '#339af0', P:  '#f783ac', S:  '#ffd43b',
    Fe: '#ff922b', Zn: '#a9e34b', Mn: '#845ef7', Co: '#f06595', Cu: '#20c997',
    Ni: '#e599f7', Mo: '#94d82d', Se: '#66d9e8', V:  '#ffe066', W:  '#b197fc',
    Mg: '#74c0fc', Eu: '#ff6b6b', Ba: '#38d9a9',
    Si: '#ffa94d', Al: '#63e6be',
};

const GCE_DEFAULT_ELEMENTS = ['Fe', 'O', 'C', 'N', 'Mg', 'Eu', 'Ba', 'Si'];

let _gceMode = 'xh';
let _gceActiveElements = new Set(GCE_DEFAULT_ELEMENTS);
let _gceInitialised = false;

const _GCE_PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(22,22,22,0.85)',
    font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 11 },
    margin: { l: 56, r: 16, t: 30, b: 44 },
    legend: {
        font: { size: 9 },
        bgcolor: 'rgba(0,0,0,0.4)',
        bordercolor: 'rgba(255,255,255,0.08)',
        borderwidth: 1,
        orientation: 'v',
        x: 1.02, y: 1,
    },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
};

/* ---- Initialisation ---- */

function initGCEChart() {
    if (!galaxyData?.gce) return;
    const gce = galaxyData.gce;

    _buildZoneSelect(gce.radius);
    _buildElementToggles(gce.elements);
    _bindGCEEvents();
    _gceInitialised = true;
    plotGCE(_gceMode);
}

function _buildZoneSelect(radii) {
    const sel = document.getElementById('gceZoneSelect');
    sel.innerHTML = '';

    const optSolar = document.createElement('option');
    optSolar.value = 'solar';
    optSolar.textContent = '태양 위치 (~8 kpc)';
    sel.appendChild(optSolar);

    const optAvg = document.createElement('option');
    optAvg.value = 'avg';
    optAvg.textContent = '전체 평균';
    sel.appendChild(optAvg);

    radii.forEach((r, i) => {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = r.toFixed(1) + ' kpc';
        sel.appendChild(o);
    });

    const solarIdx = _findSolarZone(radii);
    sel.value = 'solar';
}

function _findSolarZone(radii) {
    let best = 0, bestDist = Infinity;
    radii.forEach((r, i) => {
        const d = Math.abs(r - 8.0);
        if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
}

function _buildElementToggles(elements) {
    const bar = document.getElementById('gceElementBar');
    bar.innerHTML = '';
    elements.forEach(el => {
        if (el === 'H' || el === 'He') return;
        const btn = document.createElement('button');
        btn.className = 'gce-el-toggle ' + (_gceActiveElements.has(el) ? 'on' : 'off');
        btn.dataset.el = el;
        const dot = document.createElement('span');
        dot.className = 'el-dot';
        dot.style.background = GCE_ELEMENT_COLORS[el] || '#888';
        btn.appendChild(dot);
        btn.appendChild(document.createTextNode(el));
        btn.addEventListener('click', () => {
            if (_gceActiveElements.has(el)) {
                _gceActiveElements.delete(el);
                btn.className = 'gce-el-toggle off';
            } else {
                _gceActiveElements.add(el);
                btn.className = 'gce-el-toggle on';
            }
            plotGCE(_gceMode);
        });
        bar.appendChild(btn);
    });
}

function _bindGCEEvents() {
    document.querySelectorAll('#gceTabs .gce-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#gceTabs .gce-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _gceMode = btn.dataset.mode;
            const elBar = document.getElementById('gceElementBar');
            elBar.style.display = (_gceMode === 'mass_budget') ? 'none' : 'flex';
            plotGCE(_gceMode);
        });
    });

    document.getElementById('gceZoneSelect').addEventListener('change', () => plotGCE(_gceMode));
}

/* ---- Zone index resolution ---- */

function _resolveZoneIdx(gce) {
    const sel = document.getElementById('gceZoneSelect');
    const v = sel.value;
    if (v === 'solar') return _findSolarZone(gce.radius);
    if (v === 'avg') return -1;
    return parseInt(v);
}

function _avgOverZones(arrOfArr) {
    const nr = arrOfArr.length;
    if (nr === 0) return [];
    const nt = arrOfArr[0].length;
    const out = new Array(nt).fill(0);
    for (let i = 0; i < nr; i++)
        for (let j = 0; j < nt; j++)
            out[j] += arrOfArr[i][j];
    for (let j = 0; j < nt; j++) out[j] /= nr;
    return out;
}

function _getZoneData(data2d, zoneIdx) {
    if (zoneIdx < 0) return _avgOverZones(data2d);
    return data2d[zoneIdx];
}

/* ---- Plotting ---- */

function plotGCE(mode) {
    if (!galaxyData?.gce) return;
    const gce = galaxyData.gce;
    const zoneIdx = _resolveZoneIdx(gce);

    switch (mode) {
        case 'xh':        _plotXH(gce, zoneIdx); break;
        case 'mass_frac':  _plotMassFrac(gce, zoneIdx); break;
        case 'feh_radial': _plotFeHRadial(gce); break;
        case 'mass_budget':_plotMassBudget(gce, zoneIdx); break;
        case 'xfe_feh':    _plotXFeFe(gce, zoneIdx); break;
    }
}

function _timeCursorShape(gce) {
    if (typeof currentTime === 'undefined') return [];
    const t = currentTime;
    const tArr = gce.time;
    if (t < tArr[0] || t > tArr[tArr.length - 1]) return [];
    return [{
        type: 'line', x0: t, x1: t, y0: 0, y1: 1, yref: 'paper',
        line: { color: 'rgba(108,180,238,0.5)', width: 1, dash: 'dot' },
    }];
}

/* -- [X/H] evolution -- */
function _plotXH(gce, zoneIdx) {
    const traces = [];
    gce.elements.forEach(el => {
        if (el === 'H' || el === 'He') return;
        if (!_gceActiveElements.has(el)) return;
        if (!gce.XH[el]) return;
        const y = _getZoneData(gce.XH[el], zoneIdx);
        traces.push({
            x: gce.time, y,
            name: `[${el}/H]`,
            mode: 'lines',
            line: { color: GCE_ELEMENT_COLORS[el] || '#888', width: 1.5 },
        });
    });

    const layout = {
        ..._GCE_PLOTLY_LAYOUT_BASE,
        title: { text: '[X/H] 시간 진화', font: { size: 13, color: '#aaa' } },
        xaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.xaxis, title: '시간 (Gyr)' },
        yaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.yaxis, title: '[X/H] (dex)', range: [-5, 1.5] },
        shapes: _timeCursorShape(gce),
    };

    Plotly.react('gcePlot', traces, layout, { responsive: true, displayModeBar: false });
}

/* -- Mass fractions -- */
function _plotMassFrac(gce, zoneIdx) {
    const traces = [];
    gce.elements.forEach(el => {
        if (el === 'H' || el === 'He') return;
        if (!_gceActiveElements.has(el)) return;
        if (!gce.mass_fractions[el]) return;
        const y = _getZoneData(gce.mass_fractions[el], zoneIdx);
        traces.push({
            x: gce.time, y,
            name: el,
            mode: 'lines',
            line: { color: GCE_ELEMENT_COLORS[el] || '#888', width: 1.5 },
        });
    });

    const layout = {
        ..._GCE_PLOTLY_LAYOUT_BASE,
        title: { text: '원소 질량 분율 X_i(t)', font: { size: 13, color: '#aaa' } },
        xaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.xaxis, title: '시간 (Gyr)' },
        yaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.yaxis, title: 'X_i (질량 분율)', type: 'log' },
        shapes: _timeCursorShape(gce),
    };

    Plotly.react('gcePlot', traces, layout, { responsive: true, displayModeBar: false });
}

/* -- [Fe/H] by radius -- */
function _plotFeHRadial(gce) {
    if (!gce.XH.Fe) return;
    const traces = [];
    const nr = gce.radius.length;

    const colorScale = (i) => {
        const f = nr > 1 ? i / (nr - 1) : 0.5;
        const r = Math.round(60 + 195 * (1 - f));
        const g = Math.round(100 + 100 * f);
        const b = Math.round(60 + 195 * f);
        return `rgb(${r},${g},${b})`;
    };

    for (let ir = 0; ir < nr; ir++) {
        traces.push({
            x: gce.time,
            y: gce.XH.Fe[ir],
            name: gce.radius[ir].toFixed(1) + ' kpc',
            mode: 'lines',
            line: { color: colorScale(ir), width: 1.2 },
        });
    }

    const layout = {
        ..._GCE_PLOTLY_LAYOUT_BASE,
        title: { text: '[Fe/H] 반경별 시간 진화', font: { size: 13, color: '#aaa' } },
        xaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.xaxis, title: '시간 (Gyr)' },
        yaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.yaxis, title: '[Fe/H] (dex)', range: [-5, 1.0] },
        shapes: _timeCursorShape(gce),
    };

    Plotly.react('gcePlot', traces, layout, { responsive: true, displayModeBar: false });
}

/* -- Gas / Stellar mass & SFR -- */
function _plotMassBudget(gce, zoneIdx) {
    const tArr = gce.time;
    const gas = _getZoneData(gce.gas_mass, zoneIdx);
    const star = _getZoneData(gce.stellar_mass, zoneIdx);
    const sfr = _getZoneData(gce.sfr, zoneIdx);

    const traces = [
        { x: tArr, y: gas,  name: '가스 질량', mode: 'lines', line: { color: '#339af0', width: 2 }, yaxis: 'y' },
        { x: tArr, y: star, name: '항성 질량', mode: 'lines', line: { color: '#ff922b', width: 2 }, yaxis: 'y' },
        { x: tArr, y: sfr,  name: 'SFR',       mode: 'lines', line: { color: '#51cf66', width: 1.5, dash: 'dot' }, yaxis: 'y2' },
    ];

    const layout = {
        ..._GCE_PLOTLY_LAYOUT_BASE,
        title: { text: '가스·항성 질량 & SFR', font: { size: 13, color: '#aaa' } },
        xaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.xaxis, title: '시간 (Gyr)' },
        yaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.yaxis, title: 'Σ (M☉ pc⁻²)', side: 'left' },
        yaxis2: {
            title: 'SFR (M☉ pc⁻² Gyr⁻¹)',
            overlaying: 'y', side: 'right',
            gridcolor: 'rgba(255,255,255,0.03)',
            titlefont: { color: '#51cf66' },
            tickfont: { color: '#51cf66' },
        },
        shapes: _timeCursorShape(gce),
        legend: { ..._GCE_PLOTLY_LAYOUT_BASE.legend, x: 0.02, y: 0.98, xanchor: 'left' },
    };

    Plotly.react('gcePlot', traces, layout, { responsive: true, displayModeBar: false });
}

/* -- [X/Fe] vs [Fe/H] -- */
function _plotXFeFe(gce, zoneIdx) {
    if (!gce.XH.Fe) return;
    const feH = _getZoneData(gce.XH.Fe, zoneIdx);
    const traces = [];

    gce.elements.forEach(el => {
        if (el === 'H' || el === 'He' || el === 'Fe') return;
        if (!_gceActiveElements.has(el)) return;
        if (!gce.XH[el]) return;
        const xh = _getZoneData(gce.XH[el], zoneIdx);
        const xfe = xh.map((v, i) => v - feH[i]);
        traces.push({
            x: feH, y: xfe,
            name: `[${el}/Fe]`,
            mode: 'lines',
            line: { color: GCE_ELEMENT_COLORS[el] || '#888', width: 1.5 },
        });
    });

    const layout = {
        ..._GCE_PLOTLY_LAYOUT_BASE,
        title: { text: '[X/Fe] vs [Fe/H]', font: { size: 13, color: '#aaa' } },
        xaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.xaxis, title: '[Fe/H] (dex)', range: [-5, 1.0] },
        yaxis: { ..._GCE_PLOTLY_LAYOUT_BASE.yaxis, title: '[X/Fe] (dex)', range: [-2, 3] },
    };

    Plotly.react('gcePlot', traces, layout, { responsive: true, displayModeBar: false });
}

/* ---- Time cursor update (call from onTimeChange) ---- */
function updateGCETimeCursor() {
    if (!_gceInitialised) return;
    if (document.getElementById('gceWindow').classList.contains('hidden')) return;
    if (_gceMode === 'xfe_feh') return;

    const plotDiv = document.getElementById('gcePlot');
    if (!plotDiv?.layout?.shapes) return;

    const gce = galaxyData?.gce;
    if (!gce) return;

    Plotly.relayout('gcePlot', { shapes: _timeCursorShape(gce) });
}
