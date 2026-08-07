// ─────────────────────────────────────────────────────
// AetherEdge Dashboard — Lifecycle-aware controller
// States: ready → running → finished
// ─────────────────────────────────────────────────────

let currentStatus  = "ready";
let alertTimeout   = null;
let frameInterval  = null;
let statsInterval  = null;
let eventsInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchEvents();
    statsInterval  = setInterval(fetchStats,  700);
    eventsInterval = setInterval(fetchEvents, 1400);
    initAlertStream();
    // Don't poll frames until running
});

// ── SSE stream ──
function initAlertStream() {
    if (!window.EventSource) return;
    const es = new EventSource('/api/alerts/stream');
    es.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'connected') return;
            if (data.type === 'finished') {
                handleFinished();
                return;
            }
            showAlert(data);
            prependRow(data, true);
        } catch (_) {}
    };
}

// ── Lifecycle controls ──
async function startAnalysis() {
    try {
        await fetch('/api/start', { method: 'POST' });
        setStatus('running');
        // MJPEG stream (src=/api/stream) updates itself — no JS polling needed
    } catch(e) { console.error(e); }
}

async function stopAnalysis() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        setStatus('ready');
    } catch(e) { console.error(e); }
}



async function resetAll() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        setStatus('ready');
        closeResults();
        // Clear UI
        setEl('kpiTotal', '0'); setEl('kpiCars', '0');
        setEl('kpiHeavy', '0'); setEl('kpiFps', '—');
        setBar('carsBar','carsPercent',0);
        setBar('heavyBar','heavyPercent',0);
        setBar('ttbBar','ttbPercent',0);
        setBar('progressBar','progressPct',0);
        setEl('videoTimecode','—');
        setEl('logCount','0 events');
        document.getElementById('eventTableBody').innerHTML =
            `<tr><td colspan="5" class="empty-state">Press Start Analysis to begin…</td></tr>`;
    } catch(e) { console.error(e); }
}

function handleFinished() {
    setStatus('finished');
    setTimeout(showResults, 800);
}

// ── Status visual updates ──
function setStatus(s) {
    currentStatus = s;
    const pill = document.getElementById('statusPill');
    const dot  = document.getElementById('statusDot');
    const txt  = document.getElementById('statusText');
    const startBtn = document.getElementById('startBtn');
    const stopBtn  = document.getElementById('stopBtn');
    const live     = document.getElementById('liveBadge');
    const ready    = document.getElementById('readyBadge');

    pill.className = `status-pill ${s}`;
    dot.className  = `pulse-dot ${s}`;

    if (s === 'ready') {
        txt.textContent = 'READY';
        startBtn.style.display = '';
        stopBtn.style.display  = 'none';
        live.style.display  = 'none';
        ready.style.display = '';
    } else if (s === 'running') {
        txt.textContent = 'RUNNING';
        startBtn.style.display = 'none';
        stopBtn.style.display  = '';
        live.style.display  = '';
        ready.style.display = 'none';
    } else if (s === 'finished') {
        txt.textContent = 'FINISHED';
        startBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Run Again';
        startBtn.style.display = '';
        stopBtn.style.display  = 'none';
        live.style.display  = 'none';
        ready.style.display = '';
    }
}

// ── Frame polling ──
function updateVideoFrame() {
    const img = document.getElementById('videoStream');
    if (img) img.src = '/api/frame?t=' + Date.now();
}

// ── Stats polling ──
async function fetchStats() {
    try {
        const r = await fetch('/api/stats');
        if (!r.ok) return;
        const d = await r.json();

        // Sync status from server (handles page refresh)
        if (d.status !== currentStatus) {
            if (d.status === 'running') {
                setStatus('running');
                // MJPEG stream self-updates, nothing to start
            } else if (d.status === 'finished' && currentStatus !== 'finished') {
                handleFinished();
                return;
            } else if (d.status === 'ready') {
                setStatus('ready');
            }
        }

        animateValue('kpiTotal', d.total_crossings || 0);
        animateValue('kpiCars',  d.cars || 0);
        setEl('kpiHeavy', (d.trucks||0)+(d.buses||0));

        if (d.status === 'running') {
            setEl('kpiFps', d.processing_fps ? d.processing_fps.toFixed(1) : '…');
        }

        const tot = Math.max(d.total_crossings||1, 1);
        setBar('carsBar',  'carsPercent',  Math.round((d.cars||0)/tot*100));
        setBar('heavyBar', 'heavyPercent', Math.round(((d.trucks||0)+(d.buses||0))/tot*100));
        setBar('ttbBar',   'ttbPercent',   Math.round((d.top_to_bottom||0)/tot*100));

        if (d.progress_pct !== undefined)
            setBar('progressBar','progressPct', Math.round(d.progress_pct));
        if (d.video_fps && d.current_frame) {
            const cur = (d.current_frame / d.video_fps).toFixed(1);
            setEl('videoTimecode', `${cur}s / ${d.video_duration||'—'}s`);
        }
        if (d.video_name) setEl('videoNameDisplay', d.video_name);

    } catch(_) {}
}

// ── Event log ──
async function fetchEvents() {
    try {
        const r = await fetch('/api/events');
        if (!r.ok) return;
        const events = await r.json();
        setEl('logCount', `${events.length} events`);
        const tbody = document.getElementById('eventTableBody');
        if (!tbody) return;
        if (events.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="empty-state">Press Start Analysis to begin…</td></tr>`;
            return;
        }
        const recent = [...events].reverse().slice(0, 25);
        tbody.innerHTML = recent.map(ev => buildRow(ev, false)).join('');
    } catch(_) {}
}

function buildRow(ev, isNew) {
    const dir = ev.direction === 'top_to_bottom' ? '↓' : '↑';
    const cls = ev.vehicle_class || 'car';
    return `<tr class="${isNew?'new-row':''}">
      <td>${ev.timestamp_seconds.toFixed(2)}s</td>
      <td>#${ev.track_id}</td>
      <td><span class="class-tag ${cls}">${cls.toUpperCase()}</span></td>
      <td>${dir}</td>
      <td style="color:#10b981">✓</td>
    </tr>`;
}

function prependRow(ev, isNew) {
    const tbody = document.getElementById('eventTableBody');
    if (!tbody) return;
    const empty = tbody.querySelector('.empty-state');
    if (empty) empty.parentElement.remove();
    const tr = document.createElement('tbody');
    tr.innerHTML = buildRow(ev, isNew);
    tbody.insertBefore(tr.firstElementChild, tbody.firstChild);
    while (tbody.rows.length > 30) tbody.deleteRow(tbody.rows.length-1);
}

// ── Alert banner ──
function showAlert(ev) {
    const banner = document.getElementById('alertBanner');
    const dir    = ev.direction === 'top_to_bottom' ? '↓' : '↑';
    document.getElementById('alertTitle').textContent  = '🚨 VEHICLE CROSSED LINE';
    document.getElementById('alertDetail').textContent =
        `#${ev.track_id} · ${(ev.vehicle_class||'').toUpperCase()} · ${dir} · ${ev.timestamp_seconds.toFixed(2)}s`;
    banner.style.display = 'flex';
    clearTimeout(alertTimeout);
    alertTimeout = setTimeout(() => { banner.style.display = 'none'; }, 4000);
}

// ── Results modal ──
async function showResults() {
    try {
        const r = await fetch('/api/stats');
        const d = await r.json();

        setEl('rTotal',    d.total_crossings || 0);
        setEl('rCars',     d.cars || 0);
        setEl('rHeavy',    (d.trucks||0)+(d.buses||0));
        setEl('rDuration', d.video_duration ? `${d.video_duration}s` : '—');
        setEl('rTTB',      d.top_to_bottom || 0);
        setEl('rBTT',      d.bottom_to_top || 0);
        setEl('resultsVideoName', d.video_name || '');

        document.getElementById('resultsBackdrop').style.display = 'flex';
    } catch(_) {}
}

function closeResults() {
    document.getElementById('resultsBackdrop').style.display = 'none';
}

async function runAgain() {
    closeResults();
    await resetAll();
    await startAnalysis();
}

// ── Upload ──
function triggerFileInput() {
    document.getElementById('videoFileInput').click();
}

async function uploadVideoFile() {
    const input = document.getElementById('videoFileInput');
    if (!input.files || !input.files.length) return;
    const file = input.files[0];
    const fd   = new FormData();
    fd.append('video', file);
    setStatus('ready');
    setEl('dropzoneLabel', 'Uploading…');
    try {
        const r = await fetch('/api/upload', { method:'POST', body:fd });
        const d = await r.json();
        if (d.status === 'ready') {
            setEl('videoNameDisplay', d.video_name);
            setEl('dropzoneLabel', d.video_name);
            closeResults();
        }
    } catch(e) { alert('Upload error: ' + e); }
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    document.getElementById('videoFileInput').files = dt.files;
    uploadVideoFile();
}

// ── Config ──
function updateConfLabel() {
    setEl('confVal', document.getElementById('confRange').value);
}

async function updateConfig() {
    updateConfLabel();
    const p = {
        conf_thresh:  parseFloat(document.getElementById('confRange').value),
        frame_skip:   parseInt(document.getElementById('speedMode').value),
        direction:    document.getElementById('directionSelect').value,
        line_coords: [
            [parseInt(document.getElementById('p1x').value)||220,
             parseInt(document.getElementById('p1y').value)||420],
            [parseInt(document.getElementById('p2x').value)||1080,
             parseInt(document.getElementById('p2y').value)||420],
        ],
    };
    try {
        await fetch('/api/config', { method:'POST',
            headers:{'Content-Type':'application/json'}, body:JSON.stringify(p) });
    } catch(_) {}
}

// ── Helpers ──
function animateValue(id, newVal) {
    const el = document.getElementById(id);
    if (!el) return;
    if (parseInt(el.textContent) !== newVal) {
        el.textContent = newVal;
        el.classList.add('bump');
        setTimeout(() => el.classList.remove('bump'), 200);
    }
}
function setEl(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}
function setBar(barId, pctId, pct) {
    const bar = document.getElementById(barId);
    const lbl = document.getElementById(pctId);
    if (bar) bar.style.width = `${pct}%`;
    if (lbl) lbl.textContent = `${pct}%`;
}
