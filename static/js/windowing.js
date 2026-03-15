/* ---- Window Helpers ---- */

function openWindow(type) {
    if (type === 'hr') {
        document.getElementById('hrWindow').classList.remove('hidden');
        if (selectedStarId >= 0) loadHR(selectedStarId);
    } else if (type === 'gce') {
        document.getElementById('gceWindow').classList.remove('hidden');
    }
}

function togglePanelExpand() {
    const panel = document.getElementById('detailPanel');
    const btn = document.getElementById('expandToggle');
    panel.classList.toggle('expanded');
    if (panel.classList.contains('expanded')) {
        btn.textContent = '▶ 접기';
    } else {
        btn.textContent = '◀ 행성계';
    }
}

function closePanel(id) {
    const el = document.getElementById(id);
    el.classList.add('hidden');
    if (id === 'detailPanel') {
        el.classList.remove('expanded');
        const btn = document.getElementById('expandToggle');
        if (btn) btn.textContent = '◀ 행성계';
        selectedStarId = -1;
        cachedStarData = null;
        cachedTrack = null;
    }
}

/* ---- Draggable Windows ---- */

function startDrag(e, winId) {
    const win = document.getElementById(winId);
    const rect = win.getBoundingClientRect();
    const ox = e.clientX - rect.left, oy = e.clientY - rect.top;
    const move = ev => {
        win.style.left = (ev.clientX - ox) + 'px';
        win.style.top = (ev.clientY - oy) + 'px';
        win.style.right = 'auto'; win.style.bottom = 'auto';
    };
    const up = () => { document.removeEventListener('pointermove', move); document.removeEventListener('pointerup', up); };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
}
