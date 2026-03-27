/* ============================================================
   Validation & Observational Comparison UI
   ============================================================ */
let _validationData = null;
let _validationLoading = false;
let _validationTab = 'summary';

function loadValidation(forceRefresh) {
    if (_validationLoading) return;
    if (_validationData && !forceRefresh) {
        _renderValidation();
        return;
    }
    _validationLoading = true;
    _validationData = null;
    _renderValidationPlaceholder();

    fetchJson('/api/validation')
        .then(data => {
            _validationData = data;
            _renderValidation();
        })
        .catch(err => {
            document.getElementById('validationContent').innerHTML =
                `<div class="val-placeholder"><p style="color:var(--acc3)">검증 실행 실패</p><p class="val-sub">${err.message}</p></div>`;
        })
        .finally(() => { _validationLoading = false; });
}

function _renderValidationPlaceholder() {
    document.getElementById('validationContent').innerHTML = `
        <div class="val-placeholder">
            <div class="val-spinner"></div>
            <p>검증 데이터를 불러오는 중…</p>
            <p class="val-sub">공개 기준값 벤치마크 + VizieR 관측 카탈로그 대조 실행 중</p>
        </div>`;
}

function switchValidationTab(tab) {
    _validationTab = tab;
    document.querySelectorAll('.vtab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.vtab-panel').forEach(p => {
        p.classList.toggle('hidden', p.dataset.tab !== tab);
    });
    if (tab === 'observational' && _validationData) {
        requestAnimationFrame(() => _plotObservationalChart());
    }
}

/* ---- Main render ---- */
function _renderValidation() {
    if (!_validationData) return;
    const bench = _validationData.benchmark_validation;
    const obs = _validationData.observational_validation;
    const s = bench.summary;

    const passRate = (s.pass_rate * 100).toFixed(0);
    const passColor = s.failed === 0 ? 'var(--ok)' : 'var(--acc3)';

    let html = `
    <div class="vtab-bar">
        <button class="vtab-btn active" data-tab="summary" onclick="switchValidationTab('summary')">종합 요약</button>
        <button class="vtab-btn" data-tab="benchmarks" onclick="switchValidationTab('benchmarks')">내부 벤치마크</button>
        <button class="vtab-btn" data-tab="observational" onclick="switchValidationTab('observational')">관측 대조</button>
    </div>`;

    /* ---- Summary Tab ---- */
    html += `<div class="vtab-panel" data-tab="summary">`;
    html += _renderSummaryTab(bench, obs, passRate, passColor);
    html += `</div>`;

    /* ---- Benchmarks Tab ---- */
    html += `<div class="vtab-panel hidden" data-tab="benchmarks">`;
    html += _renderBenchmarksTab(bench);
    html += `</div>`;

    /* ---- Observational Tab ---- */
    html += `<div class="vtab-panel hidden" data-tab="observational">`;
    html += _renderObservationalTab(obs);
    html += `</div>`;

    document.getElementById('validationContent').innerHTML = html;
}

/* ---- Summary Tab ---- */
function _renderSummaryTab(bench, obs, passRate, passColor) {
    const s = bench.summary;
    const comp = obs.comparison;
    const slopeStatus = comp.slope_status;
    const slopeColor = slopeStatus === 'good' ? 'var(--ok)' : slopeStatus === 'acceptable' ? 'var(--warn)' : 'var(--acc3)';

    return `
    <div class="val-summary-grid">
        <div class="val-card">
            <div class="val-card-title">내부 벤치마크</div>
            <div class="val-big" style="color:${passColor}">${s.passed}/${s.total}</div>
            <div class="val-label">통과율 ${passRate}%</div>
            <div class="val-domain-row">
                ${Object.entries(s.domains).map(([d, v]) =>
                    `<span class="val-domain-chip"><span class="val-domain-name">${d}</span> ${v.passed}/${v.total}</span>`
                ).join('')}
            </div>
        </div>
        <div class="val-card">
            <div class="val-card-title">관측 대조 (Open Cluster)</div>
            <div class="val-big" style="color:${slopeColor}">${slopeStatus.toUpperCase()}</div>
            <div class="val-label">${obs.selection.n_clusters}개 클러스터 비교</div>
            <div class="val-detail-rows">
                <div class="val-row"><span class="lbl">기울기 차이</span><span class="val">${_fv(comp.slope_delta_dex_per_kpc, 4, true)} dex/kpc</span></div>
                <div class="val-row"><span class="lbl">σ offset</span><span class="val">${_fv(comp.slope_sigma_offset, 2)}σ</span></div>
                <div class="val-row"><span class="lbl">RMSE</span><span class="val">${_fv(comp.rmse_dex, 4)} dex</span></div>
                <div class="val-row"><span class="lbl">χ²/ν</span><span class="val">${_fv(comp.reduced_chi2, 2)}</span></div>
            </div>
        </div>
        <div class="val-card">
            <div class="val-card-title">태양권 비교</div>
            <div class="val-detail-rows" style="margin-top:8px">
                <div class="val-row"><span class="lbl">모델 [Fe/H]</span><span class="val">${_fv(obs.simulation.solar_feh, 4, true)} dex</span></div>
                <div class="val-row"><span class="lbl">관측 [Fe/H]</span><span class="val">${obs.observation.solar_feh != null ? _fv(obs.observation.solar_feh, 4, true) + ' dex' : '—'}</span></div>
                <div class="val-row"><span class="lbl">관측 산포</span><span class="val">${obs.observation.solar_scatter_dex != null ? _fv(obs.observation.solar_scatter_dex, 4) + ' dex' : '—'}</span></div>
                <div class="val-row"><span class="lbl">클러스터 수</span><span class="val">${obs.observation.solar_cluster_count}</span></div>
            </div>
        </div>
        <div class="val-card">
            <div class="val-card-title">최대 편차 메트릭</div>
            <div class="val-detail-rows" style="margin-top:8px">
                ${s.worst_metrics.slice(0, 4).map(m => `
                    <div class="val-row">
                        <span class="lbl">${m.label}</span>
                        <span class="val" style="color:${m.passed ? 'var(--ok)' : 'var(--acc3)'}">
                            ${_fv(m.actual, 4)} <span style="color:var(--dim);font-size:9px">vs ${_fv(m.observed, 4)}</span>
                        </span>
                    </div>`).join('')}
            </div>
        </div>
    </div>
    <div class="val-note">
        기준: IAU 2015 Resolution B3 (태양 상수) · JPL Planetary Physical Parameters (지구) · Asplund+2009 (Z☉) · Zhang+2024 Open Clusters (VizieR J/A+A/692/A212)
    </div>`;
}

/* ---- Benchmarks Tab ---- */
function _renderBenchmarksTab(bench) {
    const results = bench.results;
    let html = `<table class="val-table">
        <thead><tr>
            <th>지표</th><th>시뮬레이션</th><th>관측/기준값</th><th>편차</th><th>허용오차</th><th>상대%</th><th>상태</th>
        </tr></thead><tbody>`;

    for (const r of results) {
        const statusClass = r.passed ? 'val-pass' : 'val-fail';
        const statusText = r.passed ? 'PASS' : 'FAIL';
        const relPct = r.rel_diff != null ? (r.rel_diff * 100).toFixed(1) + '%' : '—';
        html += `<tr>
            <td class="val-metric-name">${r.label}<br><span class="val-unit">${r.unit}</span></td>
            <td class="mono">${_fv(r.actual, 4)}</td>
            <td class="mono">${_fv(r.observed, 4)}</td>
            <td class="mono">${_fv(r.signed_diff, 4, true)}</td>
            <td class="mono">±${_fv(r.tolerance.value, 4)}</td>
            <td class="mono">${relPct}</td>
            <td><span class="val-status ${statusClass}">${statusText}</span></td>
        </tr>`;
    }
    html += `</tbody></table>`;

    html += `<div class="val-bar-section"><div class="val-section-title">정규화 오차 (허용오차 대비)</div>`;
    for (const r of results) {
        const pct = Math.min(r.normalized_error * 100, 100);
        const color = r.normalized_error < 0.5 ? 'var(--ok)' : r.normalized_error < 0.8 ? 'var(--warn)' : 'var(--acc3)';
        html += `<div class="val-bar-row">
            <span class="val-bar-label">${r.id.split('.')[1]}</span>
            <div class="val-bar-track"><div class="val-bar-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="val-bar-pct">${(r.normalized_error * 100).toFixed(0)}%</span>
        </div>`;
    }
    html += `</div>`;
    return html;
}

/* ---- Observational Tab ---- */
function _renderObservationalTab(obs) {
    const sim = obs.simulation;
    const observed = obs.observation;
    const comp = obs.comparison;
    const profile = obs.profile.binned_profile;

    let html = `
    <div class="val-obs-header">
        <div class="val-obs-info">
            <span class="val-obs-catalog">${obs.source.catalog_id}</span>
            <span class="val-obs-desc">${obs.source.label} · ${obs.selection.n_clusters}개 클러스터 · ${obs.selection.r_min_kpc}–${obs.selection.r_max_kpc} kpc</span>
        </div>
    </div>
    <div class="val-obs-stats">
        <div class="val-obs-stat"><span class="lbl">관측 기울기</span><span class="val mono">${_fv(observed.slope_dex_per_kpc, 4)} ± ${_fv(observed.slope_err_dex_per_kpc, 4)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">모델 기울기</span><span class="val mono">${_fv(sim.sampled_slope_dex_per_kpc, 4)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">평균 |잔차|</span><span class="val mono">${_fv(comp.mean_abs_residual_dex, 4)} dex</span></div>
        <div class="val-obs-stat"><span class="lbl">중앙 |잔차|</span><span class="val mono">${_fv(comp.median_abs_residual_dex, 4)} dex</span></div>
    </div>
    <div id="valObsPlot" style="width:100%;height:320px;margin-top:8px"></div>
    <div id="valResidualPlot" style="width:100%;height:180px;margin-top:12px"></div>
    <div class="val-section-title" style="margin-top:16px">1 kpc 빈 프로파일</div>
    <table class="val-table val-table-sm">
        <thead><tr><th>반경 (kpc)</th><th>클러스터 수</th><th>관측 [Fe/H]</th><th>모델 [Fe/H]</th><th>Δ</th></tr></thead>
        <tbody>`;

    for (const p of profile) {
        const delta = p.simulated_feh - p.observed_weighted_mean_feh;
        html += `<tr>
            <td>${p.r_lo_kpc.toFixed(0)}–${p.r_hi_kpc.toFixed(0)}</td>
            <td>${p.count}</td>
            <td class="mono">${_fv(p.observed_weighted_mean_feh, 4, true)}</td>
            <td class="mono">${_fv(p.simulated_feh, 4, true)}</td>
            <td class="mono" style="color:${Math.abs(delta) < 0.05 ? 'var(--ok)' : Math.abs(delta) < 0.1 ? 'var(--warn)' : 'var(--acc3)'}">${_fv(delta, 4, true)}</td>
        </tr>`;
    }
    html += `</tbody></table>`;
    return html;
}

/* ---- Plotly chart for observational comparison ---- */
function _plotObservationalChart() {
    if (!_validationData) return;
    const obs = _validationData.observational_validation;
    const profile = obs.profile.binned_profile;
    if (!profile || profile.length === 0) return;

    const plotEl = document.getElementById('valObsPlot');
    if (!plotEl) return;

    const xObs = profile.map(p => (p.r_lo_kpc + p.r_hi_kpc) / 2);
    const yObs = profile.map(p => p.observed_weighted_mean_feh);
    const yObsErr = profile.map(p => p.observed_std_feh);
    const ySim = profile.map(p => p.simulated_feh);

    const simFull = _validationData.benchmark_validation?.actuals?.galaxy?.context;
    const xFull = simFull?.radii_kpc || xObs;
    const yFull = simFull?.final_feh_profile || ySim;

    const simSlope = obs.simulation.sampled_slope_dex_per_kpc;
    const simInt = obs.simulation.sampled_intercept_dex;
    const obsSlope = obs.observation.slope_dex_per_kpc;
    const obsInt = obs.observation.intercept_dex;
    const rMin = obs.selection.r_min_kpc;
    const rMax = obs.selection.r_max_kpc;
    const xFit = [rMin, rMax];

    const traces = [];

    if (xFull.length > 2) {
        traces.push({
            x: xFull, y: yFull,
            mode: 'lines', name: '모델 프로파일 (전체)',
            line: { color: '#6cb4ee', width: 2.5 },
        });
    }

    traces.push(
        {
            x: xObs, y: yObs,
            error_y: { type: 'data', array: yObsErr, visible: true, color: 'rgba(250,204,21,0.4)' },
            mode: 'markers', name: '관측 (빈 평균)',
            marker: { color: '#facc15', size: 10, symbol: 'diamond', line: { width: 1, color: '#fff' } },
        },
        {
            x: xObs, y: ySim,
            mode: 'markers', name: '모델 (빈 중심)',
            marker: { color: '#6cb4ee', size: 8, symbol: 'circle', line: { width: 1, color: '#fff' } },
        },
        {
            x: xFit, y: xFit.map(x => obsSlope * x + obsInt),
            mode: 'lines', name: `관측 기울기 (${obsSlope.toFixed(3)} dex/kpc)`,
            line: { color: 'rgba(250,204,21,0.5)', width: 1.5, dash: 'dash' },
        },
        {
            x: xFit, y: xFit.map(x => simSlope * x + simInt),
            mode: 'lines', name: `모델 기울기 (${simSlope.toFixed(3)} dex/kpc)`,
            line: { color: 'rgba(108,180,238,0.5)', width: 1.5, dash: 'dot' },
        },
    );

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0.15)',
        font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 11 },
        margin: { l: 55, r: 20, t: 30, b: 45 },
        xaxis: {
            title: 'R<sub>GC</sub> (kpc)', color: '#7a7a7a',
            gridcolor: 'rgba(255,255,255,0.06)', zeroline: false,
        },
        yaxis: {
            title: '[Fe/H] (dex)', color: '#7a7a7a',
            gridcolor: 'rgba(255,255,255,0.06)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.15)',
        },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.4)', font: { size: 10 } },
        showlegend: true,
    };

    Plotly.newPlot(plotEl, traces, layout, { responsive: true, displayModeBar: false });

    const residualEl = document.getElementById('valResidualPlot');
    if (residualEl && profile.length > 0) {
        const residuals = profile.map(p => p.simulated_feh - p.observed_weighted_mean_feh);
        const labels = profile.map(p => `${p.r_lo_kpc.toFixed(0)}–${p.r_hi_kpc.toFixed(0)}`);
        const colors = residuals.map(r => Math.abs(r) < 0.05 ? 'rgba(74,222,128,0.7)' : Math.abs(r) < 0.1 ? 'rgba(250,204,21,0.7)' : 'rgba(247,112,98,0.7)');

        Plotly.newPlot(residualEl, [{
            x: labels, y: residuals, type: 'bar', marker: { color: colors },
            text: residuals.map(r => (r >= 0 ? '+' : '') + r.toFixed(3)),
            textposition: 'outside', textfont: { size: 10, color: '#d4d4d4' },
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.15)',
            font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 10 },
            margin: { l: 45, r: 15, t: 24, b: 35 },
            yaxis: { title: 'Δ[Fe/H]', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.25)' },
            xaxis: { title: 'kpc', color: '#7a7a7a' },
            title: { text: '빈별 잔차 (모델 − 관측)', font: { size: 11, color: '#7a7a7a' }, x: 0.5 },
            showlegend: false,
        }, { responsive: true, displayModeBar: false });
    }
}

/* ---- Helpers ---- */
function _fv(val, digits, sign) {
    if (val == null || !isFinite(val)) return '—';
    const n = Number(val);
    const abs = Math.abs(n);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return n.toExponential(digits);
    const s = n.toFixed(digits);
    return sign && n > 0 ? '+' + s : s;
}
