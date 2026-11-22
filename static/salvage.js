const byId = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = value => `${(value * 100).toFixed(1)}%`;

function notice(message, error = false) {
    const node = byId('notice');
    node.hidden = false;
    node.classList.toggle('error', error);
    node.textContent = message;
    clearTimeout(notice.timer);
    notice.timer = setTimeout(() => { node.hidden = true; }, error ? 9000 : 4000);
}

function fourValues(text) {
    return String(text).split(/[\s,]+/).filter(Boolean).map(Number);
}

function readSettings(form) {
    const v = Object.fromEntries(new FormData(form));
    const settings = {
        view_count: Number(v.view_count), lpi: Number(v.lpi), phase: Number(v.phase),
        shear: Number(v.shear), order: v.order, latitude: Number(v.latitude),
        width_mm: Number(v.width_mm), height_mm: Number(v.height_mm),
        dpi: Number(v.dpi), mount: v.mount,
    };
    const crop = fourValues(v.crop);
    if (crop.length) {
        if (crop.length !== 4 || crop.some(n => !Number.isFinite(n))) {
            throw Error('Crop needs four numbers between 0 and 1.');
        }
        settings.crop = crop;
    }
    const damage = v.damage.split('\n').map(line => fourValues(line)).filter(parts => parts.length);
    for (const region of damage) {
        if (region.length !== 4 || region.some(n => !Number.isFinite(n))) {
            throw Error('Each damage rectangle needs four numbers between 0 and 1.');
        }
    }
    if (damage.length) settings.damage = damage;
    return { title: v.title, settings };
}

function classBar(record) {
    const rows = [
        ['read', record.read_fraction, 'status-read'],
        ['estimated', record.estimated_fraction, 'status-est'],
        ['missing', record.missing_fraction, 'status-miss'],
    ];
    return `<div class="class-bar" role="img" aria-label="read, estimated, missing proportions">
        ${rows.map(([name, frac, cls]) =>
            `<span class="${cls}" style="width:${frac * 100}%" title="${esc(name)} ${pct(frac)}"></span>`).join('')}
    </div>`;
}

function renderJob(jobId, report) {
    const base = `/api/salvage/${jobId}`;
    const views = report.views.map((r, i) => {
        return `<article class="view-card" data-view="${i}">
            <h3>View ${r.view} <small>slot ${r.slot}</small></h3>
            <div class="view-images">
                <figure><figcaption>Reconstruction</figcaption>
                    <img class="recon-image" data-view="${i}" src="${base}/asset/view-${r.view}-reconstructed"
                         alt="Reconstructed view ${r.view}"></figure>
                <figure><figcaption>Pixel status</figcaption>
                    <img src="${base}/asset/view-${r.view}-status" alt="Pixel provenance for view ${r.view}"></figure>
            </div>
            ${classBar(r)}
            <ul class="view-stats">
                <li class="status-read">Read ${pct(r.read_fraction)} (${r.read_pixels.toLocaleString()} px)</li>
                <li class="status-est">Estimated ${pct(r.estimated_fraction)} (${r.estimated_pixels.toLocaleString()} px)</li>
                <li class="status-miss">Missing ${pct(r.missing_fraction)} (${r.missing_pixels.toLocaleString()} px)</li>
            </ul>
            <p class="muted">Compact frame ${r.compact_size[0]} × ${r.compact_size[1]} ·
                ~${r.mean_read_columns_per_row} read columns/row · ${r.empty_rows} empty rows</p>
            <div class="row">
                <a class="button" href="${base}/asset/view-${r.view}-reconstructed" download>Reconstruction PNG</a>
                <a class="button" href="${base}/asset/view-${r.view}-evidence" download>Full-width evidence PNG</a>
            </div>
        </article>`;
    }).join('');
    return `<div class="report-head">
        <h2>${esc(report.input.filename)}</h2>
        <p class="warning"><strong>Reconstruction only — not a recovered camera original.</strong></p>
        <p class="muted">${esc(report.provenance)}</p>
        <div class="row">
            <a class="button orange" href="${base}/download" download>Download per-view results (ZIP)</a>
        </div>
        <div class="print-stage">
            <figure><figcaption>Print with crop / damage marks — click a view image to locate its source block</figcaption>
                <img id="print-image" src="${base}/asset/print-preview" alt="Submitted interlaced print"></figure>
            <figure><figcaption>Band assignment across the print</figcaption>
                <img src="${base}/asset/print-band-map" alt="Print band map"></figure>
        </div>
    </div>
    <div class="legend">
        <span class="status-read">● Read</span><span class="status-est">● Estimated</span>
        <span class="status-miss">● Missing</span>
    </div>
    <div class="view-grid">${views}</div>`;
}

async function locate(event) {
    const image = event.target.closest('.recon-image');
    if (!image) return;
    const rect = image.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    const jobId = byId('salvage-output').dataset.job;
    const view = Number(image.dataset.view) + 1;
    try {
        const response = await fetch(`/api/salvage/${jobId}/locate`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ view, x, y }),
        });
        const payload = await response.json();
        if (!response.ok) throw Error(payload.error || 'Location failed.');
        showLocation(payload.locate, jobId);
    } catch (error) {
        notice(error.message, true);
    }
}

function showLocation(loc, jobId) {
    const printImage = byId('print-image');
    const marker = byId('print-marker');
    if (marker) marker.remove();
    const [l, t, r, b] = loc.printed_box;
    const box = document.createElement('div');
    box.id = 'print-marker';
    box.style.left = `${l * 100}%`;
    box.style.top = `${t * 100}%`;
    box.style.width = `${(r - l) * 100}%`;
    box.style.height = `${(b - t) * 100}%`;
    printImage.parentElement.style.position = 'relative';
    printImage.parentElement.appendChild(box);
    printImage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const cls = { read: 'status-read', estimated: 'status-est', missing: 'status-miss' }[loc.status];
    const dialog = byId('salvage-dialog');
    dialog.innerHTML = `<header><h2>Print source block — view ${loc.view}</h2>
        <button type="button" data-close>Close</button></header>
        <p>This reconstructed pixel is <strong class="${cls}">${esc(loc.status)}</strong>.</p>
        <p class="muted">Compact point (${loc.compact_point[0]}, ${loc.compact_point[1]}) →
            printed column ${loc.printed_point[0]}, row ${loc.printed_point[1]} of
            ${esc(byId('salvage-output').dataset.filename)}.</p>
        <p class="muted">Read pixels come straight from that column; estimated pixels were interpolated;
            missing pixels have no printed source.</p>`;
    if (!dialog.open) dialog.showModal();
    dialog.querySelector('[data-close]').onclick = () => dialog.close();
}

async function loadJobs() {
    try {
        const data = await (await fetch('/api/salvage')).json();
        const node = byId('salvage-jobs');
        node.innerHTML = data.jobs.length ? data.jobs.map(job =>
            `<button type="button" class="job-row" data-job="${job.id}">
                <strong>${esc(job.title)}</strong>
                <span>${new Date(job.created).toLocaleString()} ·
                    ${job.summary.views.length} views · ${esc(job.summary.input.filename)}</span>
            </button>`).join('') : '<p class="muted">No salvage jobs yet.</p>';
    } catch (error) {
        byId('salvage-jobs').textContent = 'Could not load saved jobs.';
    }
}

byId('salvage-form').addEventListener('submit', async event => {
    event.preventDefault();
    const status = byId('salvage-status');
    const output = byId('salvage-output');
    let parsed;
    try {
        parsed = readSettings(event.target);
    } catch (error) {
        notice(error.message, true);
        return;
    }
    const body = new FormData();
    body.set('print', event.target.print.files[0]);
    body.set('title', parsed.title);
    body.set('settings', JSON.stringify(parsed.settings));
    status.textContent = 'Reconstructing… this can take a few seconds for large scans.';
    output.innerHTML = '';
    try {
        const response = await fetch('/api/salvage', { method: 'POST', body });
        const data = await response.json();
        if (!response.ok) throw Error(data.error || 'Reconstruction failed.');
        status.textContent = '';
        output.innerHTML = renderJob(data.job.id, data.job.report);
        output.dataset.job = data.job.id;
        output.dataset.filename = data.job.report.input.filename;
        notice('Reconstruction complete. Read, estimated and missing pixels are shown separately.');
        loadJobs();
    } catch (error) {
        status.textContent = '';
        notice(error.message, true);
    }
});

byId('salvage-jobs').addEventListener('click', async event => {
    const row = event.target.closest('.job-row');
    if (!row) return;
    const data = await (await fetch(`/api/salvage/${row.dataset.job}`)).json();
    const output = byId('salvage-output');
    output.innerHTML = renderJob(data.job.id, data.job.report);
    output.dataset.job = data.job.id;
    output.dataset.filename = data.job.report.input.filename;
    byId('salvage-status').textContent = '';
});

byId('salvage-output').addEventListener('click', locate);

loadJobs();
