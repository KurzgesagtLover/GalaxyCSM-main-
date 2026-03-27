/* ---- Star Detail And Planet Windows ---- */
let _detailRequestToken = 0;
let _detailAbortController = null;

function _interpLinear(a, b, t) {
    return a + (b - a) * t;
}

function _interpPositive(a, b, t) {
    if (!(a > 0) || !(b > 0)) return _interpLinear(a, b, t);
    return Math.exp(_interpLinear(Math.log(a), Math.log(b), t));
}

function sampleTrackAtAge(track, age) {
    if (!track || track.length === 0) return null;
    if (track.length === 1 || age <= track[0].age) return track[0];
    const last = track[track.length - 1];
    if (age >= last.age) return last;

    let hi = 1;
    while (hi < track.length && track[hi].age < age) hi++;
    const lo = track[hi - 1];
    const up = track[hi];
    const span = Math.max(up.age - lo.age, 1e-12);
    const t = Math.max(0, Math.min(1, (age - lo.age) / span));
    const phaseSource = t < 0.5 ? lo : up;

    return {
        ...phaseSource,
        age,
        T_eff: _interpPositive(lo.T_eff || 0, up.T_eff || 0, t),
        luminosity: _interpPositive(lo.luminosity || 0, up.luminosity || 0, t),
        radius: _interpPositive(lo.radius || 0, up.radius || 0, t),
        abs_mag: (lo.abs_mag !== undefined && up.abs_mag !== undefined)
            ? _interpLinear(lo.abs_mag, up.abs_mag, t)
            : phaseSource.abs_mag,
        log_g: (lo.log_g !== undefined && up.log_g !== undefined)
            ? _interpLinear(lo.log_g, up.log_g, t)
            : phaseSource.log_g,
        flare_activity: (lo.flare_activity !== undefined && up.flare_activity !== undefined)
            ? _interpLinear(lo.flare_activity, up.flare_activity, t)
            : phaseSource.flare_activity,
    };
}

function updatePanelFromCache() {
    if (!cachedStarData) return;
    const d = cachedStarData;
    const age = Math.max(currentTime - d.birth_time, 0);
    const evo = sampleTrackAtAge(cachedTrack, age);
    // Update DOM elements directly
    const el = document.getElementById('detailContent');
    if (!el) return;
    const upd = (id, val) => { const e = el.querySelector(`[data-field="${id}"]`); if (e) e.textContent = val; };
    upd('age', fmtFixedTrunc(age, 3) + ' Gyr');
    upd('remaining_ms', fmtFixedTrunc(Math.max(d.ms_lifetime - age, 0), 4) + ' Gyr');
    upd('elapsed_pct', d.ms_lifetime > 0 ? fmtPercentTrunc(age / d.ms_lifetime, 1) : '?');
    if (evo) {
        // Flare calculation
        let displayLuminosity = evo.luminosity || 0;
        let isFlaring = false;
        let isBlackDwarf = evo.phase === 'Black Dwarf';

        if (evo.flare_activity !== undefined && evo.flare_activity > 0 && !isBlackDwarf) {
            // Random chance of flare based on activity level
            if (Math.random() < (evo.flare_activity * 0.05)) {
                isFlaring = true;
                const spike = 1.0 + Math.random() * (evo.flare_activity * 2.0); // Up to many times brighter
                displayLuminosity *= spike;
            }
        }

        upd('phase', evo.phase_kr || evo.phase);
        upd('T_eff', fmtAutoSci(evo.T_eff || 0, { fixed: 1, exp: 3, large: 1e4 }) + ' K');

        // Show flare tag if active
        const lSp = el.querySelector(`[data-field="luminosity"]`);
        if (lSp) {
            if (isFlaring) {
                lSp.innerHTML = `<span style="color:#f44;font-weight:bold">🔥 ${fmtExpTrunc(displayLuminosity, 3)} L☉</span>`;
            } else {
                lSp.textContent = fmtExpTrunc(displayLuminosity, 3) + ' L☉';
            }
        }

        upd('radius', fmtAutoSci(evo.radius || 0, { fixed: 4, exp: 3, small: 1e-2, large: 1e3 }) + ' R☉');
        upd('spectral', evo.spectral_class || '?');
        upd('abs_mag', evo.abs_mag !== undefined ? fmtFixedTrunc(evo.abs_mag, 3) : '?');
        upd('log_g', evo.log_g !== undefined ? fmtFixedTrunc(evo.log_g, 3) : '?');
        upd('star_name', (evo.spectral_class || d.star_type) + ' #' + d.star_id);
        // Update phase tag
        const tag = el.querySelector('.phase-tag');
        if (tag) { tag.textContent = evo.phase_kr || evo.phase; }
        // Update star icon glow
        const icon = el.querySelector('.star-icon');
        if (icon && evo.color) {
            let glowColor = isFlaring ? '#ffbb66' : evo.color;
            let glowSize = isFlaring ? '48px' : '24px';
            if (isBlackDwarf) {
                glowColor = '#111';
                glowSize = '0px';
            }
            icon.style.background = `radial-gradient(circle,${glowColor},transparent 70%)`;
            icon.style.boxShadow = `0 0 ${glowSize} ${glowColor}`;
            icon.style.transition = 'box-shadow 0.1s, background 0.1s';
        }
    }
}

/* ---- Comprehensive Uncertainty Controls (4 Tabs) ---- */
const graphModes = {
    'r_process': { x: { id: 'yield_r_multiplier', min: 0.1, max: 10.0, val: 1.1, log: true, unit: 'x' }, y: { id: 'nsm_N_per_Msun', min: 1e-5, max: 1e-4, val: 3e-5, log: true, unit: '/M☉' } },
    's_process': { x: { id: 'yield_s_multiplier', min: 0.1, max: 10.0, val: 1.0, log: true, unit: 'x' }, y: { id: 'agb_frequency_multiplier', min: 0.1, max: 10.0, val: 1.0, log: true, unit: 'x' } },
    'ia_sn': { x: { id: 'yield_ia_multiplier', min: 0.1, max: 10.0, val: 1.0, log: true, unit: 'x' }, y: { id: 'ia_N_per_Msun', min: 0.5e-3, max: 5.0e-3, val: 2.0e-3, log: true, unit: '/M☉' } },
    'galaxy': { x: { id: 'sfr_efficiency', min: 0.01, max: 0.30, val: 0.08, log: false, unit: '' }, y: { id: 'outflow_eta', min: 0.0, max: 2.5, val: 1.1, log: false, unit: '' } }
};

let currentGraphMode = 'r_process';

async function loadStarDetail(id, isRefresh = false) {
    const requestToken = ++_detailRequestToken;
    _detailAbortController?.abort();
    _detailAbortController = new AbortController();
    const { signal } = _detailAbortController;
    selectedStarId = id;
    if (!isRefresh) {
        selectedPlanetIdx = -1;
        document.getElementById('planetWindow').classList.add('hidden');
    }
    const panel = document.getElementById('detailPanel');
    const el = document.getElementById('detailContent');
    panel.classList.remove('hidden');
    if (!isRefresh) el.innerHTML = '<p style="color:var(--dim);padding:20px">Loading...</p>';
    try {
        let qp = `t=${currentTime.toFixed(3)}`;
        const esiR = document.getElementById('esi_w_radius');
        if (esiR) qp += `&w_radius=${esiR.value}&w_density=${document.getElementById('esi_w_density').value}&w_escape=${document.getElementById('esi_w_escape').value}&w_temp=${document.getElementById('esi_w_temp').value}`;
        const lvEl = document.getElementById('lv_frac');
        if (lvEl) qp += `&lv_frac=${lvEl.value}`;
        const dpEl = document.getElementById('disprop_scale');
        if (dpEl) qp += `&disprop_scale=${dpEl.value}`;
        const data = await fetchJson(`/api/star/${id}?${withCacheQuery(qp)}`, { signal });
        if (requestToken !== _detailRequestToken || selectedStarId !== id) return;
        cachedStarData = data;
        renderStarDetail(data, el);
        // Auto-refresh planet detail window if open
        if (selectedPlanetIdx >= 0 && !document.getElementById('planetWindow').classList.contains('hidden')) {
            selectPlanet(id, selectedPlanetIdx);
        }
        // Auto-load H-R track for real-time updates
        if (!cachedTrack || !isRefresh) {
            const d2 = await fetchJson(`/api/evolution/${id}?${withCacheQuery(`t_max=${tMax}`)}`, { signal });
            if (requestToken !== _detailRequestToken || selectedStarId !== id) return;
            cachedTrack = d2.track;
        }
    } catch (e) {
        if (e.name === 'AbortError') return;
        console.error("Star load error:", e);
        if (requestToken !== _detailRequestToken) return;
        if (!isRefresh) el.innerHTML = '<p style="color:var(--acc3)">Error loading star</p>';
    }
}

function renderStarDetail(d, el) {
    const phaseCls = {
        'MS': 'MS', 'pre-MS': 'MS', 'disk': 'MS', 'subgiant': 'RGB', 'RGB': 'RGB', 'HB': 'RGB', 'AGB': 'RGB',
        'post-AGB': 'RGB', 'PN': 'WD', 'WD': 'WD', 'blue_dwarf': 'MS',
        'BSG': 'BSG', 'RSG': 'RSG', 'YSG': 'RSG', 'LBV': 'LBV', 'WR': 'WR',
        'hypergiant': 'LBV', 'SN': 'SN', 'NS': 'NS', 'BH': 'BH'
    };
    const evo = d.evolution || {};
    const pc = phaseCls[evo.phase] || 'MS';
    const pn = { rocky: '암석', hot_rocky: '뜨거운 암석', gas_giant: '가스 거대', mini_neptune: '미니 해왕성' };

    let h = `
  <div class="star-header">
    <div class="star-icon" style="background:radial-gradient(circle,${evo.color || '#fff'},transparent 70%);
      box-shadow:0 0 24px ${evo.color || '#fff'}"></div>
    <div class="star-info">
      <h3 data-field="star_name">${evo.spectral_class || d.star_type} #${d.star_id}</h3>
      <p><span class="phase-tag phase-${pc}">${evo.phase_kr || evo.phase}</span></p>
    </div>
  </div>

  <div class="section"><div class="section-title">항성 기본 정보</div>
    <div class="info-grid">
      <div class="info-row"><span class="lbl">질량</span><span class="val">${fmtAutoSci(d.star_mass, { fixed: 4, exp: 3, small: 1e-2, large: 1e3 })} M☉</span></div>
      <div class="info-row"><span class="lbl">스펙트럼</span><span class="val" data-field="spectral">${evo.spectral_class || '?'}</span></div>
      <div class="info-row"><span class="lbl">나이</span><span class="val" data-field="age">${fmtFixedTrunc(d.age_gyr, 3)} Gyr</span></div>
      <div class="info-row"><span class="lbl">탄생</span><span class="val">${fmtFixedTrunc(d.birth_time, 3)} Gyr</span></div>
      <div class="info-row"><span class="lbl">금속성 Z</span><span class="val">${fmtExpTrunc(d.metallicity, 3)}</span></div>
      <div class="info-row"><span class="lbl">위치</span><span class="val">${fmtFixedTrunc(d.radius_kpc, 3)} kpc</span></div>
    </div>
  </div>

  <div class="section"><div class="section-title">현재 물리량</div>
    <div class="info-grid">
      <div class="info-row"><span class="lbl">유효온도 T<sub>eff</sub></span><span class="val" data-field="T_eff">${evo.T_eff != null ? fmtAutoSci(evo.T_eff, { fixed: 1, exp: 3, large: 1e4 }) : '?'} K</span></div>
      <div class="info-row"><span class="lbl">광도 L</span><span class="val" data-field="luminosity">${evo.luminosity != null ? fmtExpTrunc(evo.luminosity, 3) : '?'} L☉</span></div>
      <div class="info-row"><span class="lbl">반경 R</span><span class="val" data-field="radius">${evo.radius != null ? fmtAutoSci(evo.radius, { fixed: 4, exp: 3, small: 1e-2, large: 1e3 }) : '?'} R☉</span></div>
      <div class="info-row"><span class="lbl">절대등급 M<sub>V</sub></span><span class="val" data-field="abs_mag">${evo.abs_mag != null ? fmtFixedTrunc(evo.abs_mag, 3) : '?'}</span></div>
      <div class="info-row"><span class="lbl">표면중력 log g</span><span class="val" data-field="log_g">${evo.log_g != null ? fmtFixedTrunc(evo.log_g, 3) : '?'}</span></div>
      <div class="info-row"><span class="lbl">현재 위상</span><span class="val" data-field="phase">${evo.phase_kr || evo.phase}</span></div>
    </div>
  </div>

  <div class="section"><div class="section-title">진화 타임라인</div>
    <div class="info-grid">
      <div class="info-row"><span class="lbl">주계열 수명</span><span class="val">${d.ms_lifetime != null ? fmtFixedTrunc(d.ms_lifetime, 4) : '?'} Gyr</span></div>
      <div class="info-row"><span class="lbl">주계열 잔존</span><span class="val" data-field="remaining_ms">${d.remaining_ms != null ? fmtFixedTrunc(d.remaining_ms, 4) : '?'} Gyr</span></div>
      <div class="info-row"><span class="lbl">총 수명</span><span class="val">${d.total_lifetime != null ? fmtFixedTrunc(d.total_lifetime, 4) : '?'} Gyr</span></div>
      <div class="info-row"><span class="lbl">경과 비율</span><span class="val" data-field="elapsed_pct">${d.ms_lifetime > 0 ? fmtPercentTrunc(d.age_gyr / d.ms_lifetime, 1) : '?'}</span></div>
    </div>
    ${_renderEvolutionBar(d, evo)}
    <button class="hdr-btn" style="margin-top:8px;width:100%" onclick="loadHR(${d.star_id})">
      ✦ H-R 진화 궤적 보기
    </button>
  </div>`;

    // Stellar composition
    if (d.stellar_composition) {
        h += `<div class="section"><div class="section-title">항성 조성 (탄생시 ISM)</div>
      <div style="display:flex;flex-wrap:wrap;gap:2px">`;
        for (const [el, v] of Object.entries(d.stellar_composition)) {
            if (v < 1e-10) continue;
            h += `<div class="comp-chip"><span class="comp-el">${el}</span><span class="comp-val">${fmtExpTrunc(v, 2)}</span></div>`;
        }
        h += '</div></div>';
    }
    // ============= Planet section (left column) =============
    let ph = '';

    // Asteroid Belt — Compact clickable card
    if (d.asteroid_belt) {
        const ab = d.asteroid_belt;
        const comp = ab.composition || {};
        ph += `<div class="section"><div class="section-title">소행성대</div>
      <div class="planet-card" style="cursor:pointer" onclick='selectBelt(${d.star_id})'>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-weight:600;font-size:12px">🪨 소행성대</span>
          <span style="font-size:10px;color:var(--dim)">${ab.ain_au} – ${ab.aout_au} AU</span>
        </div>
        <div class="info-grid" style="margin-top:3px">
          <div class="info-row"><span class="lbl">질량</span><span class="val" style="color:#ffa">${ab.final_mass_earth} M⊕</span></div>
          <div class="info-row"><span class="lbl">조성 S/C/M</span><span class="val">${(comp.S_type_frac * 100 || 0).toFixed(0)}/${(comp.C_type_frac * 100 || 0).toFixed(0)}/${(comp.M_type_frac * 100 || 0).toFixed(0)}%</span></div>
          <div class="info-row"><span class="lbl">충돌활성도</span><span class="val">${ab.collision_index ?? '-'}</span></div>
        </div>
        <div style="text-align:center;font-size:10px;color:var(--acc);margin-top:4px">▶ 상세 보기</div>
      </div>
    </div>`;
    }

    // Planets
    const formedCount = d.planets.filter(p => p.formed !== false).length;
    if (d.n_planets > 0) {
        ph += `<div class="section"><div class="section-title">행성계 (${formedCount}/${d.n_planets})</div>`;
        d.planets.forEach((p, i) => {
            const unformed = p.formed === false;
            const dimStyle = unformed ? 'opacity:0.35;pointer-events:none' : '';
            let hzTag = '';
            if (!unformed) {
                if (p.is_habitable) hzTag = '<span class="hz-tag" style="background:#28f">🌊 거주가능</span>';
                else if (p.in_hz_dynamic) hzTag = '<span class="hz-tag" style="background:#555">HZ 궤도</span>';
            }
            ph += `<div class="planet-card" id="pc${i}" style="${dimStyle}" onclick='selectPlanet(${d.star_id},${i})'>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-weight:600;font-size:12px">${pn[p.type] || p.type} ${i + 1}</span>
          <span style="display:flex;gap:4px">
            ${unformed ? '<span style="font-size:9px;color:var(--acc3)">미생성</span>' : ''}
            ${hzTag}
            ${p.magnetic && p.magnetic.dynamo_active ? `<span style="font-size:9px;color:#7df">🧲 ${p.magnetic.B_surface_uT.toFixed(1)}μT</span>` : ''}
            ${p.differentiation ? `<span style="font-size:9px;color:var(--dim)">${(p.differentiation.diff_progress * 100).toFixed(0)}% 분화</span>` : ''}
          </span>
        </div>
        <div class="info-grid" style="margin-top:3px">
          <div class="info-row"><span class="lbl">질량</span><span class="val">${p.mass_earth} M⊕</span></div>
          <div class="info-row"><span class="lbl">궤도</span><span class="val">${p.semi_major_au} AU</span></div>
          ${p.T_eq ? `<div class="info-row"><span class="lbl">T<sub>eq</sub></span><span class="val">${p.T_eq} K</span></div>` : ''}
        </div>`;
            if (p.has_moon_system) {
                const mc = p.moon_count;
                const _chKr = { giant_impact: '거대충돌', binary_terrestrial: '쌍행성', debris_chain: '파편열' };
                const _chTag = (p.moon_system && p.moon_system.formation_channel && _chKr[p.moon_system.formation_channel])
                    ? `<span style="font-size:8px;color:var(--dim);margin-right:4px">${_chKr[p.moon_system.formation_channel]}</span>` : '';
                ph += `<div style="text-align:right; margin-top:4px;">
                    <button onclick="event.stopPropagation(); selectMoon(${d.star_id}, ${i})" style="font-size:10px; padding:2px 8px; border-radius:4px; background:rgba(255,255,255,0.1); border:1px solid var(--dim); color:var(--brt); cursor:pointer;">
                        🌙 ${_chTag}위성 ${mc}개 보기
                    </button>
                </div>`;
            }
            if (p.differentiation) {
                const df = p.differentiation;
                ph += `<div class="diff-bar"><div class="diff-core" style="width:${df.core_frac * 100}%"></div>
          <div class="diff-mantle" style="width:${df.mantle_frac * 100}%"></div>
          <div class="diff-crust" style="width:${df.crust_frac * 100}%"></div></div>
          <div class="diff-legend"><span class="lg-core">핵 ${(df.core_frac * 100).toFixed(1)}%</span>
          <span class="lg-mantle">맨틀 ${(df.mantle_frac * 100).toFixed(1)}%</span>
          <span class="lg-crust">지각 ${(df.crust_frac * 100).toFixed(1)}%</span></div>`;
            }
            ph += '</div>';
        });
        ph += '</div>';
    }

    // Render to DOM
    el.innerHTML = h;
    const pSection = document.getElementById('planetSection');
    if (pSection) {
        pSection.innerHTML = ph;
    } else {
        // Fallback for cached HTML without planetSection
        el.innerHTML += ph;
    }

    // Auto-refresh open sub-windows
    if (!document.getElementById('planetWindow').classList.contains('hidden') && selectedPlanetIdx >= 0 && d.planets[selectedPlanetIdx]) {
        selectPlanet(d.star_id, selectedPlanetIdx);
    }
    if (!document.getElementById('beltWindow').classList.contains('hidden') && d.asteroid_belt) {
        selectBelt(d.star_id);
    }
    if (!document.getElementById('diskWindow').classList.contains('hidden') && d.disk) {
        selectDisk(d.star_id);
    }
    if (!document.getElementById('moonWindow').classList.contains('hidden') && selectedMoonPlanetIdx >= 0 && d.planets[selectedMoonPlanetIdx]) {
        selectMoon(d.star_id, selectedMoonPlanetIdx);
    }
}

/* ---- Evolution Phase Bar ---- */

function _renderEvolutionBar(d, evo) {
    const dk = d.disk || {};
    const diskLife = dk.lifetime_myr ? dk.lifetime_myr / 1000 : 0.003; // Gyr
    const msLife = d.ms_lifetime || 10;
    const totalLife = d.total_lifetime || msLife * 1.3;
    const age = d.age_gyr || 0;

    // Phase widths as % of total lifetime
    const diskPct = Math.min(diskLife / totalLife * 100, 15); // cap visual width
    const preMSPct = Math.min(diskLife * 0.5 / totalLife * 100, 10);
    const msPct = Math.min(msLife / totalLife * 100, 90 - diskPct - preMSPct);
    const postMSPct = Math.max(100 - diskPct - preMSPct - msPct, 5);

    const currentPct = Math.min(age / totalLife * 100, 100);
    const curPhase = evo.phase || 'MS';

    const isInDisk = curPhase === 'disk';
    const isInPreMS = curPhase === 'pre-MS';
    const isInMS = curPhase === 'MS' || curPhase === 'blue_dwarf';

    return `
    <div style="margin-top:10px;font-size:10px;color:var(--dim)">
      <div style="display:flex;height:22px;border-radius:6px;overflow:hidden;position:relative;cursor:pointer">
        <div onclick="selectDisk(${d.star_id})"
          style="width:${diskPct}%;background:${isInDisk ? '#f90' : '#553300'};display:flex;align-items:center;justify-content:center;transition:all .2s"
          title="원시원반 (${(diskLife * 1000).toFixed(1)} Myr)">
          <span style="font-size:8px;white-space:nowrap">🌌 원반</span>
        </div>
        <div style="width:${preMSPct}%;background:${isInPreMS ? '#c66' : '#442222'};display:flex;align-items:center;justify-content:center"
          title="전주계열 (${(diskLife * 500).toFixed(0)} Myr)">
          <span style="font-size:8px">T Tau</span>
        </div>
        <div style="width:${msPct}%;background:${isInMS ? '#369' : '#1a2a3a'};display:flex;align-items:center;justify-content:center"
          title="주계열 (${msLife.toFixed(2)} Gyr)">
          <span style="font-size:8px">주계열</span>
        </div>
        <div style="width:${postMSPct}%;background:${!isInDisk && !isInPreMS && !isInMS ? '#633' : '#2a1a1a'};display:flex;align-items:center;justify-content:center"
          title="후주계열">
          <span style="font-size:8px">후기</span>
        </div>
        <div style="position:absolute;left:${currentPct}%;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 6px #fff;pointer-events:none"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:2px;font-size:9px">
        <span>0</span>
        <span>${(totalLife).toFixed(1)} Gyr</span>
      </div>
    </div>`;
}

/* ---- Disk Detail Window ---- */

function selectDisk(starId) {
    const d = cachedStarData;
    if (!d || !d.disk) return;
    const dk = d.disk;
    const win = document.getElementById('diskWindow');
    win.classList.remove('hidden');
    const el = document.getElementById('diskContent');
    const evo = d.evolution || {};
    const isActive = evo.phase === 'disk';

    el.innerHTML = `
    <div style="padding:2px 0 8px">
      ${isActive
            ? '<div style="padding:6px 12px;background:rgba(255,153,0,0.15);border:1px solid #f90;border-radius:8px;text-align:center;margin-bottom:8px"><span style="color:#f90;font-weight:600">🌌 현재 원시원반 단계</span></div>'
            : '<div style="padding:4px 12px;background:rgba(100,100,100,0.15);border-radius:8px;text-align:center;margin-bottom:8px;font-size:11px;color:var(--dim)">원반 소산 (과거 단계)</div>'
        }
      <div class="info-grid">
        <div class="info-row"><span class="lbl">디스크 질량</span><span class="val">${dk.disk_mass_Mearth?.toLocaleString()} M⊕ (${dk.disk_mass_Msun} M☉)</span></div>
        <div class="info-row"><span class="lbl">먼지/가스 비</span><span class="val">${dk.dust_to_gas}</span></div>
        <div class="info-row"><span class="lbl">반경</span><span class="val">${dk.r_disk_au} AU</span></div>
        <div class="info-row"><span class="lbl">눈선 (Snow line)</span><span class="val">${dk.snow_line_au} AU</span></div>
        <div class="info-row"><span class="lbl">표면밀도 Σ₀</span><span class="val">${dk.sigma_0_g_cm2} g/cm² (slope ${dk.sigma_slope})</span></div>
        <div class="info-row"><span class="lbl">수명</span><span class="val">${dk.lifetime_myr} Myr</span></div>
        <div class="info-row"><span class="lbl">외부 FUV</span><span class="val">${dk.F_UV_G0} G₀</span></div>
        <div class="info-row"><span class="lbl">총 고체 질량</span><span class="val">${dk.total_solid_mass_earth} M⊕</span></div>
      </div>
    </div>`;
}

/* ---- Asteroid Belt Detail Window ---- */

function selectBelt(starId) {
    const d = cachedStarData;
    if (!d || !d.asteroid_belt) return;
    const ab = d.asteroid_belt;
    const sv = ab.survival || {};
    const sd = ab.seed || {};
    const facs = sv.factors || {};
    const comp = ab.composition || {};
    const szd = ab.size_dist || {};
    const ecc = ab.ecc_a || [];
    const inc = ab.inc_a_deg || [];
    const meanE = ecc.length ? (ecc.reduce((a, b) => a + b, 0) / ecc.length).toFixed(3) : '-';
    const meanI = inc.length ? (inc.reduce((a, b) => a + b, 0) / inc.length).toFixed(1) : '-';
    const res = ab.resonances || [];

    const win = document.getElementById('beltWindow');
    win.classList.remove('hidden');
    const el = document.getElementById('beltContent');

    // Tab structure
    let h = `<div style="display:flex;gap:4px;margin-bottom:8px">
      <button class="hdr-btn belt-tab" onclick="switchBeltTab('bt1')" id="beltTabBtn1" style="flex:1;font-size:10px">기본 정보</button>
      <button class="hdr-btn belt-tab" onclick="switchBeltTab('bt2')" id="beltTabBtn2" style="flex:1;font-size:10px">크기분포</button>
      <button class="hdr-btn belt-tab" onclick="switchBeltTab('bt3')" id="beltTabBtn3" style="flex:1;font-size:10px">궤도역학</button>
    </div>`;

    // TAB 1 — 기본 정보
    h += `<div id="bt1" class="belt-tab-content">
      <div class="info-grid">
        <div class="info-row"><span class="lbl">최종 질량</span><span class="val" style="color:#ffa">${ab.final_mass_earth} M⊕</span></div>
        <div class="info-row"><span class="lbl">중심 a₀</span><span class="val">${ab.a0_au} AU</span></div>
        <div class="info-row"><span class="lbl">경계</span><span class="val">${ab.ain_au} – ${ab.aout_au} AU (Δ${ab.belt_width_au} AU)</span></div>
        <div class="info-row"><span class="lbl">씨앗 → 잔존</span><span class="val">${sd.seed_mass_earth} → ${ab.final_mass_earth} M⊕ (×${sv.survival_fraction})</span></div>
        <div class="info-row"><span class="lbl">S / C / M</span><span class="val">${(comp.S_type_frac * 100 || 0).toFixed(0)}% / ${(comp.C_type_frac * 100 || 0).toFixed(0)}% / ${(comp.M_type_frac * 100 || 0).toFixed(0)}%</span></div>
        <div class="info-row"><span class="lbl">충돌 활성도</span><span class="val">${ab.collision_index ?? '-'}</span></div>
        <div class="info-row"><span class="lbl">평균 잔존률</span><span class="val">${ab.survived_frac_mean ?? '-'}</span></div>
      </div>
      <div style="margin-top:8px;font-size:10px;color:var(--dim)">
        <div style="font-weight:600;margin-bottom:3px">생존 인자:</div>
        <div>거대행성: ${facs.giant_presence ?? '-'} | 이주: ${facs.giant_migration ?? '-'} | 공명: ${facs.resonance_scanning ?? '-'}</div>
        <div>배아: ${facs.embryo_scattering ?? '-'} | 불안정: ${facs.late_instability ?? '-'} (${facs.instability_occurred ? '발생' : '미발생'}) | 충돌: ${facs.collisional_grinding ?? '-'}</div>
      </div>
    </div>`;

    // TAB 2 — 크기분포
    h += `<div id="bt2" class="belt-tab-content" style="display:none">
      <div class="info-grid">
        <div class="info-row"><span class="lbl">D<sub>max</sub></span><span class="val">${szd.D_max_km ?? '-'} km</span></div>
        <div class="info-row"><span class="lbl">멱법칙 기울기 q</span><span class="val">${szd.q_slope ?? '-'}</span></div>
      </div>`;

    // Size distribution mini-chart
    if (szd.D_km && szd.N_gt_D) {
        h += '<div style="margin-top:8px;font-size:9px;color:var(--dim)"><div style="font-weight:600;margin-bottom:3px">누적 크기분포 N(>D):</div>';
        const maxN = Math.max(...szd.N_gt_D.map(n => Math.log10(Math.max(n, 1))));
        szd.D_km.forEach((D, i) => {
            if (i % 4 !== 0) return;
            const logN = Math.log10(Math.max(szd.N_gt_D[i], 1));
            const barW = Math.min(logN / maxN * 120, 120);
            h += `<div style="display:flex;gap:4px;align-items:center;margin:1px 0">
              <span style="width:55px;text-align:right">${D < 1 ? D.toFixed(3) : D.toFixed(0)} km</span>
              <div style="background:rgba(108,180,238,0.45);height:5px;width:${barW}px;border-radius:2px"></div>
              <span>${szd.N_gt_D[i] < 1000 ? szd.N_gt_D[i].toFixed(0) : szd.N_gt_D[i].toExponential(1)}</span>
            </div>`;
        });
        h += '</div>';
    }
    h += '</div>';

    // TAB 3 — 궤도역학
    h += `<div id="bt3" class="belt-tab-content" style="display:none">
      <div class="info-grid">
        <div class="info-row"><span class="lbl">평균 이심률 e</span><span class="val">${meanE}</span></div>
        <div class="info-row"><span class="lbl">평균 경사각 i</span><span class="val">${meanI}°</span></div>
      </div>`;

    if (res.length > 0) {
        h += `<div style="margin-top:8px;font-size:10px;color:var(--dim)">
          <div style="font-weight:600;margin-bottom:3px">공명/커크우드 간격 (Kirkwood Gaps):</div>`;
        res.forEach(r => {
            const barW = Math.min(r.depth * 100, 100);
            h += `<div style="display:flex;gap:4px;align-items:center;margin:2px 0">
              <span style="width:28px;font-weight:600">${r.ratio}</span>
              <span style="width:55px">${r.a_au} AU</span>
              <div style="background:linear-gradient(90deg,rgba(255,100,100,${r.depth}),transparent);height:8px;width:${barW}px;border-radius:2px"></div>
              <span>${(r.depth * 100).toFixed(0)}%</span>
            </div>`;
        });
        h += '</div>';
    }
    h += '</div>';

    el.innerHTML = h;
    switchBeltTab('bt1');
}

function switchBeltTab(id) {
    document.querySelectorAll('.belt-tab-content').forEach(c => c.style.display = 'none');
    document.querySelectorAll('.belt-tab').forEach(b => b.style.opacity = '0.5');
    const tab = document.getElementById(id);
    if (tab) tab.style.display = '';
    // Highlight active button
    const idx = id.replace('bt', '');
    const btn = document.getElementById('beltTabBtn' + idx);
    if (btn) btn.style.opacity = '1';
}

/* ---- Moon Detail Window ---- */
let selectedMoonPlanetIdx = -1;
function selectMoon(starId, idx) {
    selectedMoonPlanetIdx = idx;
    const planet = cachedStarData.planets[idx];
    if (!planet || !planet.moon_system) return;
    const sys = planet.moon_system;

    const win = document.getElementById('moonWindow');
    win.classList.remove('hidden');
    const el = document.getElementById('moonContent');

    const pn = { rocky: '암석 행성', hot_rocky: '뜨거운 암석', gas_giant: '가스 거대행성', mini_neptune: '미니 해왕성' };
    const planetName = `${pn[planet.type] || planet.type} ${idx + 1}`;
    const isGasType = planet.type === 'gas_giant' || planet.type === 'mini_neptune';

    let h = `<div style="margin-bottom:10px;font-size:11px;color:var(--dim)">
        <strong style="color:var(--brt)">${planetName}의 위성계</strong>
    </div>`;

    if (isGasType) {
        h += _renderGasGiantMoonTab(sys, planet);
    } else {
        h += _renderRockyMoonTab(sys, planet);
    }

    el.innerHTML = h;
}

/* ---- Planet Detail Window ---- */

function selectPlanet(starId, idx) {
    selectedPlanetIdx = idx;
    const planet = cachedStarData.planets[idx];
    if (!planet) return;

    document.querySelectorAll('.planet-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('pc' + idx)?.classList.add('selected');
    const win = document.getElementById('planetWindow');
    win.classList.remove('hidden');
    const el = document.getElementById('planetContent');
    const pn = { rocky: '암석 행성', hot_rocky: '뜨거운 암석', gas_giant: '가스 거대행성', mini_neptune: '미니 해왕성' };

    // --- UNFORMED ---
    if (planet.formed === false) {
        el.innerHTML = '<p style="color:var(--acc3)">⏳ 원시행성 원반 단계 — 아직 미생성</p>';
        return;
    }

    const pp = planet.physical;
    const atm = planet.atmosphere;
    const df = planet.differentiation;
    const th = planet.thermal;
    const cv = planet.core_viscosity;
    const mg = planet.magnetic;

    // TAB 1 — 물리 특성
    const t1 = pp ? `<div class="info-grid">
      <div class="info-row"><span class="lbl">적도 지름</span><span class="val">${(pp.R_eq_km * 2).toLocaleString()} km</span></div>
      <div class="info-row"><span class="lbl">평균 지름</span><span class="val">${(pp.R_mean_km * 2).toLocaleString()} km</span></div>
      <div class="info-row"><span class="lbl">극지방 지름</span><span class="val">${(pp.R_pol_km * 2).toLocaleString()} km</span></div>
      <div class="info-row"><span class="lbl">편평도</span><span class="val">${pp.oblateness}</span></div>
      <div class="info-row"><span class="lbl">질량</span><span class="val">${planet.mass_earth} M⊕</span></div>
      <div class="info-row"><span class="lbl">밀도</span><span class="val">${pp.density_kg_m3?.toLocaleString()} kg/m³</span></div>
      <div class="info-row"><span class="lbl">일사량</span><span class="val">${pp.insolation_rel} S⊕</span></div>
      <div class="info-row"><span class="lbl">유효 온도</span><span class="val">${pp.T_eq_K} K</span></div>
      <div class="info-row"><span class="lbl">ESI</span><span class="val" style="color:${pp.ESI > 0.8 ? '#7f7' : pp.ESI > 0.5 ? '#ff7' : '#f77'}">${pp.ESI}</span></div>
      <div class="info-row"><span class="lbl">본드 / 기하적 반사율</span><span class="val">${pp.albedo_bond} / ${pp.albedo_geom}</span></div>
      <div class="info-row"><span class="lbl">자전주기</span><span class="val">${pp.rotation_period_hr} h</span></div>
      <div class="info-row"><span class="lbl">태양시</span><span class="val">${pp.solar_day_hr} h</span></div>
      <div class="info-row"><span class="lbl">자전축 기울기</span><span class="val">${planet.axial_tilt_deg}°</span></div>
      <div class="info-row"><span class="lbl">이심률</span><span class="val">${planet.eccentricity}</span></div>
      <div class="info-row"><span class="lbl">중력 (적도/평균/극)</span><span class="val">${pp.g_eq_m_s2} / ${pp.g_mean_m_s2} / ${pp.g_pol_m_s2} m/s²</span></div>
      <div class="info-row"><span class="lbl">원형궤도 / 탈출 속도</span><span class="val">${pp.v_circ_km_s} / ${pp.v_esc_km_s} km/s</span></div>
      <div class="info-row"><span class="lbl">나이</span><span class="val">${pp.age_gyr} Gyr</span></div>
      <div class="info-row"><span class="lbl">조석 가열</span><span class="val">${pp.tidal_heating_TW} TW</span></div>
    </div>` : '<p style="color:var(--dim)">데이터 없음</p>';

    // TAB 2 — 대기
    let t2 = '<p style="color:var(--dim)">데이터 없음</p>';
    if (atm) {
        t2 = `<div class="info-grid">
      <div class="info-row"><span class="lbl">유효 온도</span><span class="val">${atm.T_eq_K} K</span></div>
      <div class="info-row"><span class="lbl">표면 기온</span><span class="val">${atm.surface_temp_K} K</span></div>
      <div class="info-row"><span class="lbl">온실효과</span><span class="val" style="color:${atm.greenhouse_K > 100 ? '#f77' : '#7df'}">+${atm.greenhouse_K} K</span></div>
      <div class="info-row"><span class="lbl">대기압</span><span class="val">${atm.surface_pressure_atm != null ? atm.surface_pressure_atm + ' atm' : '—'}</span></div>
      <div class="info-row"><span class="lbl">공기밀도</span><span class="val">${atm.air_density_kg_m3} kg/m³</span></div>
      <div class="info-row"><span class="lbl">스케일 높이</span><span class="val">${atm.scale_height_km} km</span></div>
      <div class="info-row"><span class="lbl">대류권계면</span><span class="val">${atm.tropopause_km} km</span></div>
      <div class="info-row"><span class="lbl">균질권 고도</span><span class="val">${atm.homosphere_km} km</span></div>
      <div class="info-row"><span class="lbl">음속</span><span class="val">${atm.sound_speed_m_s} m/s</span></div>
      <div class="info-row"><span class="lbl">외기권 온도</span><span class="val">${atm.exosphere_temp_K} K</span></div>
      <div class="info-row"><span class="lbl">평균 몰 질량</span><span class="val">${atm.mean_mol_mass} g/mol</span></div>
      <div class="info-row"><span class="lbl">대기 질량</span><span class="val">${atm.atm_mass_kg != null ? fmtMass(atm.atm_mass_kg) : '—'}</span></div>
      <div class="info-row"><span class="lbl">상대 질량</span><span class="val">${atm.atm_relative_mass != null ? Number(atm.atm_relative_mass).toExponential(2) : '—'}</span></div>
    </div>`;
        const ph = atm.photolysis;
        if (ph && ph.species) {
            t2 += `<div style="margin:6px 0;padding:6px;border-radius:6px;background:rgba(255,255,255,0.03)">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
            <span style="background:#546;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700">${ph.stellar_uv_class}</span>
            <span style="font-size:10px;color:var(--dim)">F_UV=${ph.F_UV_total} erg/cm²/s</span>
          </div>
          <table class="el-table" style="font-size:9px"><thead><tr>
            <th>분자</th><th>τ 상층</th><th>τ 중층</th><th>τ 적분</th>
          </tr></thead><tbody>`;
            for (const [sp, v] of Object.entries(ph.species)) {
                const fmt = x => x > 1e6 ? '∞' : x > 100 ? x.toFixed(0) + 'yr' : x > 1 ? x.toFixed(1) + 'yr' : x > 0.01 ? (x * 365).toFixed(0) + 'd' : (x * 365 * 24).toFixed(0) + 'h';
                t2 += `<tr><td class="el-name">${sp}</td>
                  <td>${fmt(v.tau_upper_yr)}</td><td>${fmt(v.tau_mid_yr)}</td><td>${fmt(v.tau_column_yr)}</td></tr>`;
            }
            t2 += '</tbody></table></div>';
        }
    }

    // TAB 3 — 조성
    let t3 = '<p style="color:var(--dim)">데이터 없음</p>';
    if (atm?.composition && Object.keys(atm.composition).length > 0) {
        t3 = `<table class="el-table"><thead><tr><th>분자</th><th>부피%</th><th>분압 (atm)</th><th>비율</th></tr></thead><tbody>`;
        for (const [mol, data] of Object.entries(atm.composition)) {
            const pct = data.pct;
            const barW = Math.min(pct, 100);
            const color = mol === 'N2' ? '#48f' : mol === 'O2' ? '#4f8' : mol === 'CO2' ? '#fa4' :
                mol === 'H2O' ? '#4df' : mol === 'CH4' ? '#f84' : mol === 'H2' ? '#aaf' :
                    mol === 'He' ? '#ffa' : mol === 'Ar' ? '#a8f' : mol === 'SO2' ? '#f44' : '#888';
            t3 += `<tr><td class="el-name">${mol}</td>
              <td>${pct < 0.01 ? pct.toExponential(2) : pct.toFixed(3)}%</td>
              <td>${data.partial_P_atm < 0.001 ? data.partial_P_atm.toExponential(2) : data.partial_P_atm.toFixed(4)}</td>
              <td class="bar-cell"><div style="background:${color};height:10px;border-radius:2px;width:${Math.max(barW * 0.8, 2).toFixed(0)}px"></div></td></tr>`;
        }
        t3 += '</tbody></table>';
    }

    // TAB 4 — 내부구조
    let t4 = '<p style="color:var(--dim)">암석 행성 데이터 없음</p>';
    if (df) {
        const allEls = Object.keys(df.core);
        let elTable = `<table class="el-table"><thead><tr><th>원소</th><th>핵</th><th>맨틀</th><th>지각</th><th>분포</th></tr></thead><tbody>`;
        const formatPct = v => {
            const p = v * 100;
            if (p === 0) return '0.00%';
            if (Math.abs(p) < 0.01) return p.toExponential(2) + '%';
            return p.toFixed(2) + '%';
        };

        allEls.forEach(e => {
            const c = df.core[e] || 0, m = df.mantle[e] || 0, cr = df.crust[e] || 0;
            const mx = Math.max(c, m, cr, 0.001);
            elTable += `<tr><td class="el-name">${e}</td><td>${formatPct(c)}</td><td>${formatPct(m)}</td>
              <td>${formatPct(cr)}</td><td class="bar-cell"><div style="display:flex;gap:1px">
              <div class="el-bar core" style="width:${(c / mx * 28).toFixed(0)}px"></div>
              <div class="el-bar mantle" style="width:${(m / mx * 28).toFixed(0)}px"></div>
              <div class="el-bar crust" style="width:${(cr / mx * 28).toFixed(0)}px"></div>
              </div></td></tr>`;
        });
        elTable += '</tbody></table>';
        t4 = `<div class="info-grid" style="margin-bottom:8px">
          <div class="info-row"><span class="lbl">Fe/Si</span><span class="val">${df.fe_si}</span></div>
          <div class="info-row"><span class="lbl">산화도 ΔIW</span><span class="val">${df.oxidation_iw}</span></div>
          <div class="info-row"><span class="lbl">분화 진행</span><span class="val">${(df.diff_progress * 100).toFixed(1)}%</span></div>
        </div>`;
        // Late Veneer class info
        const lv = planet.late_veneer || df.late_veneer;
        if (lv) {
            const lvColors = { LV0: '#666', LV1: '#68a', LV2: '#4a8', LV3: '#c84', LV4: '#e44' };
            t4 += `<div style="margin:6px 0;padding:6px;border-radius:6px;background:rgba(255,255,255,0.03)">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
            <span style="background:${lvColors[lv.class] || '#888'};padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700">${lv.class}</span>
            <span style="font-size:11px;color:var(--dim)">Late Veneer: ${lv.label} (${lv.score})</span>
          </div>
          <div class="info-grid" style="font-size:10px">
            <div class="info-row"><span class="lbl">LV \uC9C8\uB7C9\uBE44</span><span class="val">${lv.lv_mass_frac}</span></div>
            <div class="info-row"><span class="lbl">\uBB3C \uC804\uB2EC</span><span class="val">${lv.water_frac}</span></div>
            <div class="info-row"><span class="lbl">HSE</span><span class="val">\u00D7${lv.hse_enrichment}</span></div>
            <div class="info-row"><span class="lbl">\uD0C4\uC18C\uC9C8</span><span class="val">${(lv.carbonaceous_frac * 100).toFixed(0)}%</span></div>
            <div class="info-row"><span class="lbl">\uD3ED\uACA9 \u03C4</span><span class="val">${lv.tail_myr} Myr</span></div>
          </div>
        </div>`;
        }
        t4 += `<div class="diff-bar" style="height:22px">
          <div class="diff-core" style="width:${df.core_frac * 100}%"></div>
          <div class="diff-mantle" style="width:${df.mantle_frac * 100}%"></div>
          <div class="diff-crust" style="width:${df.crust_frac * 100}%"></div>
        </div>
        <div class="diff-legend" style="margin-bottom:10px">
          <span class="lg-core">핵 ${(df.core_frac * 100).toFixed(1)}%</span>
          <span class="lg-mantle">맨틀 ${(df.mantle_frac * 100).toFixed(1)}%</span>
          <span class="lg-crust">지각 ${(df.crust_frac * 100).toFixed(1)}%</span>
        </div>${elTable}`;
    }

    // TAB 5 — 핵 & 자기장
    let t5 = '';
    if (th) {
        t5 += `<div class="section-title" style="margin-bottom:6px">🔥 핵 열진화</div>
        <div class="info-grid" style="margin-bottom:12px">
          <div class="info-row"><span class="lbl">핵 온도</span><span class="val">${th.T_core?.toLocaleString()} K</span></div>
          <div class="info-row"><span class="lbl">CMB 온도</span><span class="val">${th.T_cmb?.toLocaleString()} K</span></div>
          <div class="info-row"><span class="lbl">고상선</span><span class="val">${th.T_solidus?.toLocaleString()} K</span></div>
          <div class="info-row"><span class="lbl">CMB 열류</span><span class="val">${th.Q_cmb_TW} TW</span></div>
          <div class="info-row"><span class="lbl">방사성 열</span><span class="val">${th.P_radio_TW} TW</span></div>
          <div class="info-row"><span class="lbl">내핵 비율</span><span class="val">${(th.inner_core_frac * 100).toFixed(1)}%</span></div>
          <div class="info-row"><span class="lbl">핵 반경</span><span class="val">${th.R_core_km} km</span></div>
          <div class="info-row"><span class="lbl">행성 반경</span><span class="val">${th.R_planet_km} km</span></div>
          <div class="info-row"><span class="lbl">외핵 상태</span><span class="val">${th.core_liquid ? '🔴 액체' : '⚪ 고체'}</span></div>
        </div>`;
    }
    if (cv) {
        t5 += `<div class="info-grid" style="margin-bottom:12px">
          <div class="info-row"><span class="lbl">핵 점성</span><span class="val">${cv.eta_Pa_s.toExponential(2)} Pa·s</span></div>
          <div class="info-row"><span class="lbl">핵 상태</span><span class="val">${cv.state === 'liquid' ? '액체' : '고체'}</span></div>
        </div>`;
    }
    if (mg) {
        t5 += `<div class="section-title" style="margin-bottom:6px">🧲 자기장</div>
        <div class="info-grid">
          <div class="info-row"><span class="lbl">표면 자기장</span><span class="val">${mg.B_surface_uT} μT</span></div>
          <div class="info-row"><span class="lbl">다이나모</span><span class="val" style="color:${mg.dynamo_active ? '#7f7' : '#f77'}">${mg.dynamo_active ? '✅ 가동' : '❌ 정지'}</span></div>
          <div class="info-row"><span class="lbl">분류</span><span class="val">${mg.field_type}</span></div>
        </div>`;
    }
    if (!t5) t5 = '<p style="color:var(--dim)">암석 행성 데이터 없음</p>';

    const tabs = [
        { id: 'phy', label: '물리', content: t1 },
        { id: 'atm', label: '대기', content: t2 },
        { id: 'comp', label: '조성', content: t3 },
        { id: 'int', label: '내부구조', content: t4 },
        { id: 'core', label: '핵·자기장', content: t5 },
        { id: 'rad', label: '방사선', content: '' },
    ];

    // TAB 6 — 방사선 및 방어막 (Radiation & Defense)
    let t6 = '<p style="color:var(--dim)">방사선 데이터 없음</p>';
    if (planet.radiation) {
        const rad = planet.radiation;
        const fmtW = v => {
            if (v < 1e-4) return '~ 0.00';
            if (v > 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
            return v.toPrecision(3);
        };
        const shielded = (rad.transmittance.x_euv < 0.1 && rad.transmittance.uv < 0.3);
        const safe_txt = shielded ? '<span style="color:#7f7">안전 (강력한 차폐)</span>' : '<span style="color:#f77">위험 (피폭 심각)</span>';

        t6 = `
        <div class="info-grid" style="margin-bottom:12px">
          <div class="info-row"><span class="lbl">자기권계면</span><span class="val">${rad.magnetopause_r} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">태양풍 방어</span><span class="val">${rad.mag_shielding_pct}%</span></div>
          <div class="info-row"><span class="lbl">생명체 지표</span><span class="val">${safe_txt}</span></div>
        </div>
        <table class="el-table">
            <thead>
                <tr><th>파장 종류</th><th>우주 입사량</th><th>지표 도달량</th><th>투과율</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td class="el-name" style="color:#b266ff">X-선 / 극자외선</td>
                    <td>${fmtW(rad.incident_W_m2.x_euv)}</td>
                    <td>${fmtW(rad.surface_W_m2.x_euv)}</td>
                    <td>${(rad.transmittance.x_euv * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                    <td class="el-name" style="color:#6688ff">자외선 (UV)</td>
                    <td>${fmtW(rad.incident_W_m2.uv)}</td>
                    <td>${fmtW(rad.surface_W_m2.uv)}</td>
                    <td>${(rad.transmittance.uv * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                    <td class="el-name" style="color:#ffff66">가시광선 (VIS)</td>
                    <td>${fmtW(rad.incident_W_m2.vis)}</td>
                    <td>${fmtW(rad.surface_W_m2.vis)}</td>
                    <td>${(rad.transmittance.vis * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                    <td class="el-name" style="color:#ff6666">적외선 (IR)</td>
                    <td>${fmtW(rad.incident_W_m2.ir)}</td>
                    <td>${fmtW(rad.surface_W_m2.ir)}</td>
                    <td>${(rad.transmittance.ir * 100).toFixed(1)}%</td>
                </tr>
            </tbody>
        </table>
        <div style="font-size:10px; color:var(--dim); margin-top:10px;">
           * 단위: W/m² (수직 입사면적 기준). 자기장과 대기($O_3, CO_2$)가 방패 역할을 수행합니다.
        </div>`;
    }
    tabs[5].content = t6;

    // TAB 7 — 위성계
    let t7 = '<p style="color:var(--dim)">위성 데이터 없음</p>';
    const ms = planet.moon_system;
    if (ms) {
        const isGasType = planet.type === 'gas_giant' || planet.type === 'mini_neptune';
        if (isGasType) {
            t7 = _renderGasGiantMoonTab(ms, planet);
        } else {
            t7 = _renderRockyMoonTab(ms, planet);
        }
    }
    tabs.push({ id: 'moon', label: '위성계', content: t7 });

    const header = `<div style="margin-bottom:6px;font-size:11px;color:var(--dim)">
      <strong style="color:var(--brt)">${pn[planet.type] || planet.type} ${idx + 1}</strong>
      · ${planet.mass_earth} M⊕ · ${planet.semi_major_au} AU
      ${planet.in_hz ? '<span class="hz-tag">HZ</span>' : ''}
    </div>`;

    const tabBar = `<div class="ptab-bar">${tabs.map(t =>
        `<button class="ptab-btn${t.id === 'phy' ? ' active' : ''}" onclick="switchPlanetTab('${t.id}')" id="ptab-${t.id}">${t.label}</button>`
    ).join('')}</div>`;

    const tabPanels = tabs.map(t =>
        `<div class="ptab-panel${t.id === 'phy' ? '' : ' hidden'}" id="ptab-panel-${t.id}">${t.content}</div>`
    ).join('');

    el.innerHTML = header + tabBar + tabPanels;
}

function switchPlanetTab(id) {
    document.querySelectorAll('.ptab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.ptab-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById('ptab-' + id)?.classList.add('active');
    document.getElementById('ptab-panel-' + id)?.classList.remove('hidden');
}

function _renderGasGiantMoonTab(sys, planet) {
    const sm = sys.summary || {};
    let h = `<div style="margin-bottom:8px;font-size:11px;color:var(--dim)">
      규칙위성 <strong style="color:var(--brt)">${sm.n_regular || 0}</strong>개
      · 불규칙위성 <strong style="color:var(--brt)">${sm.n_irregular || 0}</strong>개
      · 총질량 <strong style="color:var(--brt)">${sm.total_regular_mass_earth || 0}</strong> M⊕
      · 공명쌍 ${sm.resonant_chain_count || 0}
    </div>`;
    if (sys.cpd) {
        const c = sys.cpd;
        h += `<div class="info-grid" style="margin-bottom:10px">
          <div class="info-row"><span class="lbl">CPD 질량</span><span class="val">${c.mass_earth} M⊕</span></div>
          <div class="info-row"><span class="lbl">CPD 반경</span><span class="val">${c.outer_radius_rp} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">눈선</span><span class="val">${c.snow_line_rp} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">고체 질량</span><span class="val">${c.accessible_solid_mass_earth} M⊕</span></div>
          <div class="info-row"><span class="lbl">수명</span><span class="val">${c.dissipation_myr} Myr</span></div>
          <div class="info-row"><span class="lbl">호스트 체제</span><span class="val">${c.host_regime || '-'}</span></div>
        </div>`;
    }
    if (sys.regular_moons && sys.regular_moons.length > 0) {
        h += `<div class="section-title">규칙 위성 (${sys.regular_moons.length})</div>
        <table class="el-table" style="margin-bottom:10px">
        <thead><tr><th>이름</th><th>질량(M⊕)</th><th>궤도(R<sub>p</sub>)</th><th>유형</th></tr></thead><tbody>`;
        const famKr = { io_like: '이오형', europa_like: '유로파형', ganymede_like: '가니메데형', callisto_like: '칼리스토형', titan_like: '타이탄형' };
        const famClr = { io_like: '#fa0', europa_like: '#aaf', ganymede_like: '#8df', callisto_like: '#999', titan_like: '#fd4' };
        sys.regular_moons.forEach(m => {
            const fLabel = famKr[m.family] || m.family;
            const fColor = famClr[m.family] || 'var(--dim)';
            const resTag = m.resonance ? ` <span style="font-size:8px;color:var(--acc)">[${m.resonance.type}]</span>` : '';
            h += `<tr><td class="el-name">${m.name}${resTag}</td>
              <td>${m.mass_earth > 0.001 ? m.mass_earth.toFixed(4) : m.mass_earth.toExponential(2)}</td>
              <td>${m.semi_major_rp.toFixed(1)}</td>
              <td><span style="color:${fColor}">${fLabel}</span></td></tr>`;
        });
        h += '</tbody></table>';
    }
    if (sys.irregular_moons && sys.irregular_moons.length > 0) {
        h += `<div class="section-title">불규칙 위성 (${sys.irregular_moons.length})</div>
        <table class="el-table">
        <thead><tr><th>이름</th><th>질량(M⊕)</th><th>궤도(R<sub>Hill</sub>)</th><th>e</th><th>방향</th></tr></thead><tbody>`;
        sys.irregular_moons.forEach(m => {
            const dir = m.retrograde ? '<span style="color:#f77">역행</span>' : '순행';
            h += `<tr><td class="el-name">${m.name}</td>
              <td>${m.mass_earth.toExponential(2)}</td>
              <td>${m.semi_major_rhill.toFixed(3)}</td>
              <td>${m.eccentricity.toFixed(2)}</td>
              <td>${dir}</td></tr>`;
        });
        h += '</tbody></table>';
    }
    if ((!sys.regular_moons || sys.regular_moons.length === 0) && (!sys.irregular_moons || sys.irregular_moons.length === 0)) {
        h += '<p style="color:var(--dim);text-align:center;margin-top:8px">생성된 위성이 없습니다.</p>';
    }
    return h;
}

function _renderRockyMoonTab(sys, planet) {
    const channelKr = {
        giant_impact: '거대 충돌',
        binary_terrestrial: '쌍행성형 충돌',
        debris_chain: '파편열 형성',
        none: '위성 없음',
    };
    const channelClr = {
        giant_impact: '#4facfe',
        binary_terrestrial: '#f0a',
        debris_chain: '#fa4',
        none: 'var(--dim)',
    };
    const sm = sys.summary || {};
    const imp = sys.impact_state || {};
    const dd = sys.debris_disk || {};
    const ch = sys.formation_channel || 'none';

    let h = `<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <span style="background:${channelClr[ch] || 'var(--dim)'};padding:2px 10px;border-radius:10px;font-size:10px;font-weight:700;color:#000">
        ${channelKr[ch] || ch}
      </span>`;
    if (sm.binary_like) {
        h += `<span style="background:#f0a;padding:2px 8px;border-radius:10px;font-size:9px;font-weight:600;color:#000">쌍행성</span>`;
    }
    h += `</div>`;

    if (ch === 'none') {
        h += `<div class="info-grid" style="margin-bottom:10px">
          <div class="info-row"><span class="lbl">충돌 확률</span><span class="val">${((imp.giant_impact_prob || 0) * 100).toFixed(1)}%</span></div>
          <div class="info-row"><span class="lbl">Hill 반경</span><span class="val">${imp.hill_radius_rp || '-'} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">안정 한계</span><span class="val">${imp.stable_outer_rp || '-'} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">Roche 한계</span><span class="val">${imp.roche_limit_rp || '-'} R<sub>p</sub></span></div>
          <div class="info-row"><span class="lbl">생존 인자</span><span class="val">${imp.survival_factor || '-'}</span></div>
          <div class="info-row"><span class="lbl">항성 조석 패널티</span><span class="val">${imp.stellar_tide_penalty || '-'}</span></div>
        </div>
        <p style="color:var(--dim);text-align:center">조건에 따라 위성이 형성되지 않았습니다.</p>`;
        return h;
    }

    h += `<div class="info-grid" style="margin-bottom:10px">
      <div class="info-row"><span class="lbl">충돌 확률</span><span class="val">${((imp.giant_impact_prob || 0) * 100).toFixed(1)}%</span></div>
      <div class="info-row"><span class="lbl">임팩터 비율 γ</span><span class="val">${imp.impactor_mass_ratio || '-'}</span></div>
      <div class="info-row"><span class="lbl">충돌속도/탈출속도</span><span class="val">${imp.impact_velocity_over_escape || '-'}</span></div>
      <div class="info-row"><span class="lbl">충돌 각도</span><span class="val">${imp.impact_angle_deg || '-'}°</span></div>
      <div class="info-row"><span class="lbl">각운동량 지표</span><span class="val">${imp.angular_momentum_proxy || '-'}</span></div>
      <div class="info-row"><span class="lbl">Late Veneer 점수</span><span class="val">${imp.late_veneer_score || '-'}</span></div>
    </div>`;

    h += `<div class="section-title">파편 디스크</div>
    <div class="info-grid" style="margin-bottom:10px">
      <div class="info-row"><span class="lbl">디스크 질량</span><span class="val">${dd.disk_mass_earth || 0} M⊕</span></div>
      <div class="info-row"><span class="lbl">질량 분율</span><span class="val">${((dd.disk_mass_fraction || 0) * 100).toFixed(3)}%</span></div>
      <div class="info-row"><span class="lbl">규산염</span><span class="val">${((dd.silicate_fraction || 0) * 100).toFixed(1)}%</span></div>
      <div class="info-row"><span class="lbl">철</span><span class="val">${((dd.iron_fraction || 0) * 100).toFixed(1)}%</span></div>
      <div class="info-row"><span class="lbl">휘발성물질</span><span class="val">${((dd.volatile_fraction || 0) * 100).toFixed(1)}%</span></div>
    </div>`;

    const compBar = `<div style="display:flex;height:14px;border-radius:4px;overflow:hidden;margin-bottom:10px">
      <div style="width:${(dd.silicate_fraction || 0) * 100}%;background:#c8960f" title="규산염"></div>
      <div style="width:${(dd.iron_fraction || 0) * 100}%;background:#b83220" title="철"></div>
      <div style="width:${(dd.volatile_fraction || 0) * 100}%;background:#4facfe" title="휘발성"></div>
    </div>
    <div style="display:flex;gap:12px;font-size:9px;color:var(--dim);margin-bottom:10px">
      <span><span style="display:inline-block;width:7px;height:7px;border-radius:2px;background:#c8960f;vertical-align:middle;margin-right:2px"></span>규산염</span>
      <span><span style="display:inline-block;width:7px;height:7px;border-radius:2px;background:#b83220;vertical-align:middle;margin-right:2px"></span>철</span>
      <span><span style="display:inline-block;width:7px;height:7px;border-radius:2px;background:#4facfe;vertical-align:middle;margin-right:2px"></span>휘발성</span>
    </div>`;
    h += compBar;

    h += `<div class="section-title">위성 요약</div>
    <div class="info-grid" style="margin-bottom:10px">
      <div class="info-row"><span class="lbl">대형 위성</span><span class="val">${sm.n_major || 0}개</span></div>
      <div class="info-row"><span class="lbl">소형 위성</span><span class="val">${sm.n_minor || 0}개</span></div>
      <div class="info-row"><span class="lbl">최대 위성 질량</span><span class="val">${sm.largest_moon_mass_earth || 0} M⊕</span></div>
      <div class="info-row"><span class="lbl">총 위성 질량</span><span class="val">${sm.total_moon_mass_earth || 0} M⊕</span></div>
      <div class="info-row"><span class="lbl">위성/행성 질량비</span><span class="val">${sm.moon_to_planet_mass_ratio || 0}</span></div>
    </div>`;

    const famKrMoon = { moon_like: '달형', charon_like: '카론형', volatile_impact_moon: '휘발성충돌', phobos_like: '포보스형', deimos_like: '데이모스형' };
    const famClrMoon = { moon_like: '#ddd', charon_like: '#f0a', volatile_impact_moon: '#4df', phobos_like: '#a88', deimos_like: '#886' };

    if (sys.major_moons && sys.major_moons.length > 0) {
        h += `<div class="section-title">대형 위성</div>
        <table class="el-table" style="margin-bottom:8px">
        <thead><tr><th>이름</th><th>질량(M⊕)</th><th>궤도(R<sub>p</sub>)</th><th>주기(d)</th><th>유형</th></tr></thead><tbody>`;
        sys.major_moons.forEach(m => {
            const fLabel = famKrMoon[m.family] || m.family;
            const fColor = famClrMoon[m.family] || 'var(--dim)';
            h += `<tr><td class="el-name">${m.name}</td>
              <td>${m.mass_earth > 0.001 ? m.mass_earth.toFixed(4) : m.mass_earth.toExponential(2)}</td>
              <td>${m.semi_major_rp.toFixed(1)}</td>
              <td>${m.orbital_period_days.toFixed(2)}</td>
              <td><span style="color:${fColor}">${fLabel}</span></td></tr>`;
        });
        h += '</tbody></table>';
    }

    if (sys.minor_moons && sys.minor_moons.length > 0) {
        h += `<div class="section-title">소형 위성</div>
        <table class="el-table">
        <thead><tr><th>이름</th><th>질량(M⊕)</th><th>궤도(R<sub>p</sub>)</th><th>주기(d)</th><th>유형</th></tr></thead><tbody>`;
        sys.minor_moons.forEach(m => {
            const fLabel = famKrMoon[m.family] || m.family;
            const fColor = famClrMoon[m.family] || 'var(--dim)';
            h += `<tr><td class="el-name">${m.name}</td>
              <td>${m.mass_earth.toExponential(2)}</td>
              <td>${m.semi_major_rp.toFixed(1)}</td>
              <td>${m.orbital_period_days.toFixed(2)}</td>
              <td><span style="color:${fColor}">${fLabel}</span></td></tr>`;
        });
        h += '</tbody></table>';
    }

    if ((!sys.major_moons || sys.major_moons.length === 0) && (!sys.minor_moons || sys.minor_moons.length === 0)) {
        h += '<p style="color:var(--dim);text-align:center;margin-top:8px">위성이 형성되지 않았습니다.</p>';
    }
    return h;
}



/* ---- H-R Diagram with Labeled Regions ---- */
async
