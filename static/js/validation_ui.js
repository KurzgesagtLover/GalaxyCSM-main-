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
            <p class="val-sub">공개 기준값 벤치마크 + VizieR 관측 카탈로그 + 운동학 진단 실행 중</p>
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
    if (tab === 'kinematics' && _validationData) {
        requestAnimationFrame(() => _plotKinematicCharts());
    }
}

/* ---- Main render ---- */
function _renderValidation() {
    if (!_validationData) return;
    const bench = _validationData.benchmark_validation;
    const obs = _validationData.observational_validation;
    const kin = _validationData.stellar_kinematic_validation;
    const s = bench.summary;

    const passRate = (s.pass_rate * 100).toFixed(0);
    const passColor = s.failed === 0 ? 'var(--ok)' : 'var(--acc3)';

    let html = `
    <div class="vtab-bar">
        <button class="vtab-btn active" data-tab="summary" onclick="switchValidationTab('summary')">종합 요약</button>
        <button class="vtab-btn" data-tab="benchmarks" onclick="switchValidationTab('benchmarks')">내부 벤치마크</button>
        <button class="vtab-btn" data-tab="observational" onclick="switchValidationTab('observational')">관측 대조</button>
        <button class="vtab-btn" data-tab="kinematics" onclick="switchValidationTab('kinematics')">운동학</button>
    </div>`;

    /* ---- Summary Tab ---- */
    html += `<div class="vtab-panel" data-tab="summary">`;
    html += _renderSummaryTab(bench, obs, kin, passRate, passColor);
    html += `</div>`;

    /* ---- Benchmarks Tab ---- */
    html += `<div class="vtab-panel hidden" data-tab="benchmarks">`;
    html += _renderBenchmarksTab(bench);
    html += `</div>`;

    /* ---- Observational Tab ---- */
    html += `<div class="vtab-panel hidden" data-tab="observational">`;
    html += _renderObservationalTab(obs);
    html += `</div>`;

    /* ---- Kinematics Tab ---- */
    html += `<div class="vtab-panel hidden" data-tab="kinematics">`;
    html += _renderKinematicsTab(kin);
    html += `</div>`;

    document.getElementById('validationContent').innerHTML = html;
}

/* ---- Summary Tab ---- */
function _renderSummaryTab(bench, obs, kin, passRate, passColor) {
    const s = bench.summary;
    const comp = obs.comparison;
    const slopeStatus = comp.slope_status;
    const slopeColor = slopeStatus === 'good' ? 'var(--ok)' : slopeStatus === 'acceptable' ? 'var(--warn)' : 'var(--acc3)';
    const kinOverall = kin?.overall || {};
    const kinAvr = kin?.avr || {};
    const kinFeh = kin?.feh_guiding_radius || {};
    const kinColor = _statusColor(kinOverall.status);

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
            <div class="val-card-title">운동학 진단</div>
            <div class="val-big" style="color:${kinColor}">${String(kinOverall.status || 'unknown').toUpperCase()}</div>
            <div class="val-label">${kinOverall.passed_checks ?? 0}/${kinOverall.total_checks ?? 0} checks</div>
            <div class="val-detail-rows">
                <div class="val-row"><span class="lbl">AVR</span><span class="val">${String(kinAvr.status || 'unknown').toUpperCase()}</span></div>
                <div class="val-row"><span class="lbl">corr(나이, σR)</span><span class="val">${_fv(kinAvr.sigma_R_age_correlation, 3)}</span></div>
                <div class="val-row"><span class="lbl">corr(나이, e)</span><span class="val">${_fv(kin?.eccentricity?.age_correlation, 3)}</span></div>
                <div class="val-row"><span class="lbl">Young [Fe/H]-Rg</span><span class="val">${_fv(kinFeh?.young?.slope_dex_per_kpc, 4, true)} dex/kpc</span></div>
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
    const migration = obs.migration_model || {};
    const migrationLabel = migration.enabled
        ? `+${_fv(migration.mean_birth_offset_kpc, 2)} kpc / σ ${_fv(migration.sigma_kpc, 2)} kpc`
        : '비활성';

    let html = `
    <div class="val-obs-header">
        <div class="val-obs-info">
            <span class="val-obs-catalog">${obs.source.catalog_id}</span>
            <span class="val-obs-desc">${obs.source.label} · ${obs.selection.n_clusters}개 클러스터 · ${obs.selection.r_min_kpc}–${obs.selection.r_max_kpc} kpc</span>
        </div>
    </div>
    <div class="val-obs-stats">
        <div class="val-obs-stat"><span class="lbl">관측 기울기</span><span class="val mono">${_fv(observed.slope_dex_per_kpc, 4)} ± ${_fv(observed.slope_err_dex_per_kpc, 4)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">모델 기울기 (보정)</span><span class="val mono">${_fv(sim.sampled_slope_dex_per_kpc, 4)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">모델 기울기 (원본)</span><span class="val mono">${_fv(sim.sampled_gas_slope_dex_per_kpc, 4)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">평균 |잔차|</span><span class="val mono">${_fv(comp.mean_abs_residual_dex, 4)} dex</span></div>
        <div class="val-obs-stat"><span class="lbl">평균 |잔차| (원본)</span><span class="val mono">${_fv(comp.gas_only_mean_abs_residual_dex, 4)} dex</span></div>
        <div class="val-obs-stat"><span class="lbl">Migration 커널</span><span class="val mono">${migrationLabel}</span></div>
        <div class="val-obs-stat"><span class="lbl">잔차 개선</span><span class="val mono">${_fv(comp.mean_abs_residual_gain_dex, 4)} dex</span></div>
    </div>
    <div id="valObsPlot" style="width:100%;height:320px;margin-top:8px"></div>
    <div id="valResidualPlot" style="width:100%;height:180px;margin-top:12px"></div>
    <div class="val-section-title" style="margin-top:16px">1 kpc 빈 프로파일</div>
    <table class="val-table val-table-sm">
        <thead><tr><th>반경 (kpc)</th><th>클러스터 수</th><th>관측 [Fe/H]</th><th>모델 [Fe/H] (보정)</th><th>모델 [Fe/H] (원본)</th><th>Δ</th></tr></thead>
        <tbody>`;

    for (const p of profile) {
        const delta = p.simulated_feh - p.observed_weighted_mean_feh;
        html += `<tr>
            <td>${p.r_lo_kpc.toFixed(0)}–${p.r_hi_kpc.toFixed(0)}</td>
            <td>${p.count}</td>
            <td class="mono">${_fv(p.observed_weighted_mean_feh, 4, true)}</td>
            <td class="mono">${_fv(p.simulated_feh, 4, true)}</td>
            <td class="mono">${_fv(p.simulated_feh_raw ?? p.simulated_feh, 4, true)}</td>
            <td class="mono" style="color:${Math.abs(delta) < 0.05 ? 'var(--ok)' : Math.abs(delta) < 0.1 ? 'var(--warn)' : 'var(--acc3)'}">${_fv(delta, 4, true)}</td>
        </tr>`;
    }
    html += `</tbody></table>`;
    return html;
}

/* ---- Kinematics Tab ---- */
function _renderKinematicsTab(kin) {
    if (!kin) {
        return `<div class="val-placeholder"><p class="val-sub">운동학 진단 데이터가 없습니다.</p></div>`;
    }
    const overall = kin.overall || {};
    const avr = kin.avr || {};
    const ecc = kin.eccentricity || {};
    const fehRg = kin.feh_guiding_radius || {};
    const bins = kin.age_binned_stats || [];
    const selection = kin.selection || {};

    let html = `
    <div class="val-obs-header">
        <div class="val-obs-info">
            <span class="val-obs-catalog">Synthetic Disk Kinematics</span>
            <span class="val-obs-desc">${selection.alive_stars ?? 0}개 현재 생존성 · 태양권 ${selection.solar_annulus_kpc ? selection.solar_annulus_kpc[0] : '—'}–${selection.solar_annulus_kpc ? selection.solar_annulus_kpc[1] : '—'} kpc</span>
        </div>
    </div>
    <div class="val-obs-stats">
        <div class="val-obs-stat"><span class="lbl">전체 상태</span><span class="val mono" style="color:${_statusColor(overall.status)}">${String(overall.status || 'unknown').toUpperCase()}</span></div>
        <div class="val-obs-stat"><span class="lbl">통과 체크</span><span class="val mono">${overall.passed_checks ?? 0}/${overall.total_checks ?? 0}</span></div>
        <div class="val-obs-stat"><span class="lbl">corr(나이, σR)</span><span class="val mono">${_fv(avr.sigma_R_age_correlation, 3)}</span></div>
        <div class="val-obs-stat"><span class="lbl">corr(나이, σz)</span><span class="val mono">${_fv(avr.sigma_z_age_correlation, 3)}</span></div>
        <div class="val-obs-stat"><span class="lbl">corr(나이, e)</span><span class="val mono">${_fv(ecc.age_correlation, 3)}</span></div>
        <div class="val-obs-stat"><span class="lbl">Young [Fe/H]-Rg</span><span class="val mono">${_fv(fehRg?.young?.slope_dex_per_kpc, 4, true)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">Old [Fe/H]-Rg</span><span class="val mono">${_fv(fehRg?.old?.slope_dex_per_kpc, 4, true)} dex/kpc</span></div>
        <div class="val-obs-stat"><span class="lbl">Old 산포 > Young</span><span class="val mono">${fehRg?.diagnostics?.old_scatter_exceeds_young ? 'YES' : 'NO'}</span></div>
    </div>
    <div id="valAvrPlot" style="width:100%;height:280px;margin-top:8px"></div>
    <div id="valEccPlot" style="width:100%;height:250px;margin-top:12px"></div>
    <div id="valFehRgPlot" style="width:100%;height:320px;margin-top:12px"></div>
    <div class="val-section-title" style="margin-top:16px">연령 빈 운동학 요약</div>
    <table class="val-table val-table-sm">
        <thead><tr><th>나이 빈 (Gyr)</th><th>N</th><th>σR</th><th>σz</th><th>e<sub>med</sub></th><th>상태</th></tr></thead>
        <tbody>`;

    for (const b of bins) {
        const binOk = !!(b.sigma_r_in_reference && b.sigma_z_in_reference && b.eccentricity_in_reference);
        html += `<tr>
            <td>${_fv(b.age_lo_gyr, 1)}–${_fv(b.age_hi_gyr, 1)}</td>
            <td>${b.count}</td>
            <td class="mono">${_fv(b.sigma_r_measured_km_s, 1)} <span style="color:var(--dim)">[${_fv(b.sigma_r_reference_range_km_s?.[0], 0)}–${_fv(b.sigma_r_reference_range_km_s?.[1], 0)}]</span></td>
            <td class="mono">${_fv(b.sigma_z_measured_km_s, 1)} <span style="color:var(--dim)">[${_fv(b.sigma_z_reference_range_km_s?.[0], 0)}–${_fv(b.sigma_z_reference_range_km_s?.[1], 0)}]</span></td>
            <td class="mono">${_fv(b.median_eccentricity, 3)} <span style="color:var(--dim)">[${_fv(b.eccentricity_reference_range?.[0], 2)}–${_fv(b.eccentricity_reference_range?.[1], 2)}]</span></td>
            <td><span class="val-status ${binOk ? 'val-pass' : 'val-fail'}">${binOk ? 'PASS' : 'CHECK'}</span></td>
        </tr>`;
    }
    html += `</tbody></table>
    <div class="val-note">이 탭은 외부 카탈로그 likelihood가 아니라, Milky Way 원반의 넓은 경험적 범위와 비교하는 chemo-dynamical sanity check입니다.</div>`;
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
    const ySimRaw = profile.map(p => p.simulated_feh_raw ?? p.simulated_feh);

    const simFull = _validationData.benchmark_validation?.actuals?.galaxy?.context;
    const xFull = obs.simulation?.radii_kpc || simFull?.radii_kpc || xObs;
    const yFullRaw = obs.simulation?.gas_profile_feh || simFull?.final_feh_profile || ySimRaw;
    const yFullMigrated = obs.simulation?.migration_adjusted_profile_feh || ySim;

    const simSlope = obs.simulation.sampled_slope_dex_per_kpc;
    const simInt = obs.simulation.sampled_intercept_dex;
    const gasSlope = obs.simulation.sampled_gas_slope_dex_per_kpc ?? simSlope;
    const gasInt = obs.simulation.sampled_gas_intercept_dex ?? simInt;
    const obsSlope = obs.observation.slope_dex_per_kpc;
    const obsInt = obs.observation.intercept_dex;
    const rMin = obs.selection.r_min_kpc;
    const rMax = obs.selection.r_max_kpc;
    const xFit = [rMin, rMax];

    const traces = [];

    if (xFull.length > 2) {
        traces.push({
            x: xFull, y: yFullRaw,
            mode: 'lines', name: '모델 가스장 (원본)',
            line: { color: 'rgba(108,180,238,0.45)', width: 1.8, dash: 'dash' },
        });
        traces.push({
            x: xFull, y: yFullMigrated,
            mode: 'lines', name: '모델 프로파일 (이동 보정)',
            line: { color: '#6cb4ee', width: 2.6 },
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
            mode: 'markers', name: '모델 (빈 중심, 보정)',
            marker: { color: '#6cb4ee', size: 8, symbol: 'circle', line: { width: 1, color: '#fff' } },
        },
        {
            x: xFit, y: xFit.map(x => obsSlope * x + obsInt),
            mode: 'lines', name: `관측 기울기 (${obsSlope.toFixed(3)} dex/kpc)`,
            line: { color: 'rgba(250,204,21,0.5)', width: 1.5, dash: 'dash' },
        },
        {
            x: xFit, y: xFit.map(x => gasSlope * x + gasInt),
            mode: 'lines', name: `모델 기울기 원본 (${gasSlope.toFixed(3)} dex/kpc)`,
            line: { color: 'rgba(108,180,238,0.35)', width: 1.2, dash: 'dot' },
        },
        {
            x: xFit, y: xFit.map(x => simSlope * x + simInt),
            mode: 'lines', name: `모델 기울기 보정 (${simSlope.toFixed(3)} dex/kpc)`,
            line: { color: 'rgba(108,180,238,0.65)', width: 1.5, dash: 'dot' },
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

function _plotKinematicCharts() {
    const kin = _validationData?.stellar_kinematic_validation;
    if (!kin) return;

    const bins = kin.age_binned_stats || [];
    const x = bins.map(b => b.mean_age_gyr ?? ((b.age_lo_gyr + b.age_hi_gyr) / 2));

    const avrEl = document.getElementById('valAvrPlot');
    if (avrEl && bins.length > 0) {
        const sigmaR = bins.map(b => b.sigma_r_measured_km_s);
        const sigmaZ = bins.map(b => b.sigma_z_measured_km_s);
        const sigmaRLo = bins.map(b => b.sigma_r_reference_range_km_s?.[0] ?? null);
        const sigmaRHi = bins.map(b => b.sigma_r_reference_range_km_s?.[1] ?? null);
        const sigmaZLo = bins.map(b => b.sigma_z_reference_range_km_s?.[0] ?? null);
        const sigmaZHi = bins.map(b => b.sigma_z_reference_range_km_s?.[1] ?? null);

        Plotly.newPlot(avrEl, [
            { x, y: sigmaRHi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
            { x, y: sigmaRLo, mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(108,180,238,0.10)', line: { width: 0 }, name: 'σR 참조 범위' },
            { x, y: sigmaZHi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
            { x, y: sigmaZLo, mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(250,204,21,0.10)', line: { width: 0 }, name: 'σz 참조 범위' },
            { x, y: sigmaR, mode: 'lines+markers', name: 'σR 측정', line: { color: '#6cb4ee', width: 2.2 }, marker: { size: 8 } },
            { x, y: sigmaZ, mode: 'lines+markers', name: 'σz 측정', line: { color: '#facc15', width: 2.0 }, marker: { size: 8, symbol: 'diamond' } },
        ], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.15)',
            font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 11 },
            margin: { l: 55, r: 20, t: 28, b: 42 },
            xaxis: { title: '나이 (Gyr)', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: '속도분산 (km/s)', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.35)', font: { size: 10 } },
            title: { text: 'Age-Velocity Dispersion Relation', font: { size: 12, color: '#7a7a7a' }, x: 0.5 },
        }, { responsive: true, displayModeBar: false });
    }

    const eccEl = document.getElementById('valEccPlot');
    if (eccEl && bins.length > 0) {
        const eccMed = bins.map(b => b.median_eccentricity);
        const eccLo = bins.map(b => b.eccentricity_reference_range?.[0] ?? null);
        const eccHi = bins.map(b => b.eccentricity_reference_range?.[1] ?? null);

        Plotly.newPlot(eccEl, [
            { x, y: eccHi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
            { x, y: eccLo, mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(244,114,182,0.10)', line: { width: 0 }, name: 'e 참조 범위' },
            { x, y: eccMed, mode: 'lines+markers', name: '중앙 e', line: { color: '#f472b6', width: 2.2 }, marker: { size: 8 } },
        ], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.15)',
            font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 11 },
            margin: { l: 55, r: 20, t: 28, b: 42 },
            xaxis: { title: '나이 (Gyr)', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: '궤도 이심률 e', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.35)', font: { size: 10 } },
            title: { text: 'Age-Eccentricity Relation', font: { size: 12, color: '#7a7a7a' }, x: 0.5 },
        }, { responsive: true, displayModeBar: false });
    }

    const fehRgEl = document.getElementById('valFehRgPlot');
    const raw = kin.raw_samples || {};
    const young = raw.young_feh_guiding_radius || {};
    const old = raw.old_feh_guiding_radius || {};
    if (fehRgEl) {
        const traces = [];
        const youngX = young.guiding_radius_kpc || [];
        const youngY = young.feh_dex || [];
        const oldX = old.guiding_radius_kpc || [];
        const oldY = old.feh_dex || [];

        if (youngX.length > 0) {
            traces.push({
                x: youngX, y: youngY, mode: 'markers', name: '젊은 별',
                marker: { color: 'rgba(108,180,238,0.55)', size: 6, symbol: 'circle' },
            });
        }
        if (oldX.length > 0) {
            traces.push({
                x: oldX, y: oldY, mode: 'markers', name: '오래된 별',
                marker: { color: 'rgba(250,204,21,0.45)', size: 6, symbol: 'diamond' },
            });
        }

        const xAll = youngX.concat(oldX).filter(v => Number.isFinite(v));
        const xMin = xAll.length ? Math.min(...xAll) : 4;
        const xMax = xAll.length ? Math.max(...xAll) : 12;
        const youngFit = kin.feh_guiding_radius?.young || {};
        const oldFit = kin.feh_guiding_radius?.old || {};

        if (youngFit.slope_dex_per_kpc != null && youngFit.intercept_dex != null) {
            traces.push({
                x: [xMin, xMax],
                y: [xMin, xMax].map(xVal => youngFit.slope_dex_per_kpc * xVal + youngFit.intercept_dex),
                mode: 'lines', name: `젊은 기울기 (${youngFit.slope_dex_per_kpc.toFixed(3)})`,
                line: { color: '#6cb4ee', width: 2.0 },
            });
        }
        if (oldFit.slope_dex_per_kpc != null && oldFit.intercept_dex != null) {
            traces.push({
                x: [xMin, xMax],
                y: [xMin, xMax].map(xVal => oldFit.slope_dex_per_kpc * xVal + oldFit.intercept_dex),
                mode: 'lines', name: `오래된 기울기 (${oldFit.slope_dex_per_kpc.toFixed(3)})`,
                line: { color: '#facc15', width: 2.0, dash: 'dash' },
            });
        }

        Plotly.newPlot(fehRgEl, traces, {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.15)',
            font: { family: 'Outfit, sans-serif', color: '#d4d4d4', size: 11 },
            margin: { l: 55, r: 20, t: 28, b: 42 },
            xaxis: { title: 'Guiding Radius Rg (kpc)', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: '[Fe/H] (dex)', color: '#7a7a7a', gridcolor: 'rgba(255,255,255,0.06)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.15)' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.35)', font: { size: 10 } },
            title: { text: '[Fe/H] vs Guiding Radius (Young vs Old)', font: { size: 12, color: '#7a7a7a' }, x: 0.5 },
        }, { responsive: true, displayModeBar: false });
    }
}

/* ---- Helpers ---- */
function _statusColor(status) {
    if (status === 'good') return 'var(--ok)';
    if (status === 'mixed' || status === 'acceptable') return 'var(--warn)';
    if (status === 'tension' || status === 'fail') return 'var(--acc3)';
    return 'var(--dim)';
}

function _fv(val, digits, sign) {
    if (val == null || !isFinite(val)) return '—';
    const n = Number(val);
    const abs = Math.abs(n);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return n.toExponential(digits);
    const s = n.toFixed(digits);
    return sign && n > 0 ? '+' + s : s;
}
