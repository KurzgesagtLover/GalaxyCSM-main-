/* ---- H-R Diagram Helpers ---- */

function updateHRMarker() {
    if (!cachedTrack || !cachedStarData) return;
    const hrPlot = document.getElementById('hrPlot');
    if (!hrPlot || !hrPlot.data) return;
    const age = Math.max(currentTime - cachedStarData.birth_time, 0);
    let best = 0;
    for (let i = 0; i < cachedTrack.length; i++) {
        if (cachedTrack[i].age <= age) best = i;
    }
    const pt = cachedTrack[best];
    if (!pt || pt.T_eff <= 100 || pt.luminosity <= 1e-7) return;
    // Find the marker trace (the one with star-diamond symbol)
    const markerIdx = hrPlot.data.findIndex(t => t.marker && t.marker.symbol === 'star-diamond');
    if (markerIdx >= 0) {
        Plotly.restyle('hrPlot', {
            x: [[Math.log10(pt.T_eff)]],
            y: [[Math.log10(Math.max(pt.luminosity, 1e-7))]],
            'marker.color': [pt.color],
            text: [[pt.phase_kr]],
            name: `현재: ${pt.phase_kr}`,
        }, [markerIdx]);
    }
}

async function loadHR(starId) {
    const win = document.getElementById('hrWindow');
    win.classList.remove('hidden');
    try {
        let track;
        if (cachedTrack && selectedStarId === starId) {
            track = cachedTrack;
        } else {
            const r = await fetch(`/api/evolution/${starId}?${withCacheQuery(`t_max=${tMax}`)}`);
            const d = await r.json();
            cachedTrack = d.track;
            track = d.track;
        }
        const birth = cachedStarData?.birth_time || galaxyData.stars.birth[starId];
        const mass = cachedStarData?.star_mass || galaxyData.stars.mass[starId];
        plotHR({ track, birth, mass, star_id: starId });
    } catch (e) { console.error(e); }
}

function plotHR(data) {
    const tr = data.track.filter(p => p.luminosity > 1e-7 && p.T_eff > 100);
    const logT = tr.map(p => Math.log10(p.T_eff));
    const logL = tr.map(p => Math.log10(Math.max(p.luminosity, 1e-7)));
    const phases = tr.map(p => p.phase);
    const colors = tr.map(p => p.color);
    const labels = tr.map(p => `${p.phase_kr} (${p.age.toFixed(3)} Gyr)\nT=${p.T_eff}K L=${p.luminosity.toExponential(2)}`);

    // Current position
    const age = currentTime - data.birth;
    let curIdx = 0;
    for (let i = 0; i < tr.length; i++) { if (tr[i].age <= age) curIdx = i; }

    // Phase-colored segments
    const phaseColors = {
        'pre-MS': '#ffcc6f', 'MS': '#4facfe', 'subgiant': '#ffd2a1', 'RGB': '#ff8040',
        'HB': '#ffd060', 'AGB': '#ff4020', 'post-AGB': '#cc3010', 'PN': '#cc80ff',
        'WD': '#d0d8ff', 'blue_dwarf': '#60c0ff',
        'BSG': '#92b5ff', 'RSG': '#ff6030', 'YSG': '#ffcc40', 'LBV': '#60e0ff',
        'hypergiant': '#ff90e0', 'WR': '#a060ff', 'SN': '#ffffff', 'NS': '#8080ff', 'BH': '#444'
    };

    // Build colored trace segments
    const traces = [];
    let segStart = 0;
    for (let i = 1; i <= tr.length; i++) {
        if (i === tr.length || phases[i] !== phases[segStart]) {
            const x = logT.slice(segStart, i + (i < tr.length ? 1 : 0));
            const y = logL.slice(segStart, i + (i < tr.length ? 1 : 0));
            traces.push({
                x, y, mode: 'lines', line: { color: phaseColors[phases[segStart]] || '#888', width: 2.5 },
                name: tr[segStart].phase_kr || phases[segStart], showlegend: false,
                hovertext: labels.slice(segStart, i), hoverinfo: 'text',
            });
            segStart = i;
        }
    }

    // Current position marker
    traces.push({
        x: [logT[curIdx]], y: [logL[curIdx]], mode: 'markers+text',
        marker: {
            size: 14, color: colors[curIdx], line: { width: 2, color: '#fff' },
            symbol: 'star-diamond'
        },
        text: [tr[curIdx].phase_kr], textposition: 'top right',
        textfont: { color: '#fff', size: 11 },
        name: `현재: ${tr[curIdx].phase_kr}`, showlegend: true,
    });

    // H-R diagram background image
    const images = [{
        source: '/static/img/hr_background.png',
        xref: 'x', yref: 'y',
        x: 5.2, y: 7,
        sizex: -2.0, sizey: 12,
        sizing: 'stretch',
        layer: 'below',
        opacity: 0.8
    }];

    // Region labels as annotations
    const annotations = [
        { x: 4.1, y: -2.5, text: '백색왜성', showarrow: false, font: { color: 'rgba(208,216,255,0.4)', size: 10 } },
        { x: 4.15, y: 2, text: '주계열', showarrow: false, font: { color: 'rgba(79,172,254,0.4)', size: 10 }, textangle: -50 },
        { x: 3.5, y: 3, text: '적색거성', showarrow: false, font: { color: 'rgba(255,128,64,0.4)', size: 10 } },
        { x: 4.3, y: 5.5, text: '초거성', showarrow: false, font: { color: 'rgba(255,144,224,0.4)', size: 10 } },
        { x: 4.6, y: 5, text: 'LBV / WR', showarrow: false, font: { color: 'rgba(160,96,255,0.4)', size: 9 } },
        // Temperature scale
        { x: 4.7, y: -4.5, text: 'O B', showarrow: false, font: { color: 'rgba(146,181,255,0.5)', size: 9 } },
        { x: 4.0, y: -4.5, text: 'A F', showarrow: false, font: { color: 'rgba(248,247,255,0.5)', size: 9 } },
        { x: 3.7, y: -4.5, text: 'G K', showarrow: false, font: { color: 'rgba(255,210,161,0.5)', size: 9 } },
        { x: 3.5, y: -4.5, text: 'M', showarrow: false, font: { color: 'rgba(255,204,111,0.5)', size: 9 } },
    ];

    // Legend entries for phases
    const seenPhases = new Set();
    tr.forEach(p => {
        if (!seenPhases.has(p.phase)) {
            seenPhases.add(p.phase);
            traces.push({
                x: [null], y: [null], mode: 'markers',
                marker: { size: 8, color: phaseColors[p.phase] || '#888' },
                name: p.phase_kr, showlegend: true,
            });
        }
    });

    Plotly.newPlot('hrPlot', traces, {
        title: {
            text: `H-R 다이어그램 — ${data.mass} M☉ (#${data.star_id})`,
            font: { color: '#d0d8ee', size: 14, family: 'Outfit' }
        },
        xaxis: {
            title: 'log T<sub>eff</sub> (K)', autorange: false, range: [5.2, 3.2],
            color: '#6a78a0', gridcolor: 'rgba(100,140,255,0.06)'
        },
        yaxis: {
            title: 'log L/L☉', autorange: false, range: [-5, 7],
            color: '#6a78a0', gridcolor: 'rgba(100,140,255,0.06)'
        },
        paper_bgcolor: 'rgba(12,18,36,0.95)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Outfit', color: '#d0d8ee' },
        showlegend: true,
        legend: {
            font: { size: 9 }, bgcolor: 'rgba(12,18,36,0.7)',
            bordercolor: 'rgba(100,140,255,0.1)', borderwidth: 1, x: 0.01, y: 0.99
        },
        margin: { t: 40, b: 55, l: 55, r: 20 },
        images, annotations,
    }, { responsive: true });
}
