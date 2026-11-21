// Two-view middle synthesis dialog: circle the ornament and mark one clear
// background corner on each capture, then preview the interpolated frame.
import {ask, button, esc, field, img, modal, notice, readout, select} from './ui.js';

export const midState = {
    open: false, side: 'left', mode: 'polygon',
    leftPoly: [], rightPoly: [], leftAnchor: null, rightAnchor: null,
    mid: 0.5, preview: null, busy: false,
};

function currentPoly() {
    return midState.side === 'left' ? midState.leftPoly : midState.rightPoly;
}
function setPoly(points) {
    if (midState.side === 'left') midState.leftPoly = points;
    else midState.rightPoly = points;
}
function currentAnchor() {
    return midState.side === 'left' ? midState.leftAnchor : midState.rightAnchor;
}
function setAnchor(point) {
    if (midState.side === 'left') midState.leftAnchor = point;
    else midState.rightAnchor = point;
}

function polygonSvg(points, color) {
    if (points.length < 2) return '';
    const stroke = points.map(p => p.map(v => (v * 100).toFixed(2)).join(',')).join(' ');
    const dots = points.map(([x, y]) =>
        `<circle cx="${(x * 100).toFixed(2)}" cy="${(y * 100).toFixed(2)}" r="1.1" fill="${color}"/>`).join('');
    return `<svg class="mark-layer" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polygon points="${stroke}" fill="${color}55" stroke="${color}" stroke-width="0.6"
        ${points.length >= 3 ? '' : 'fill-opacity="0"'} vector-effect="non-scaling-stroke"/>${dots}</svg>`;
}

function anchorSvg(point, color) {
    if (!point) return '';
    const [x, y] = point;
    const px = x * 100, py = y * 100;
    return `<svg class="mark-layer" viewBox="0 0 100 100" preserveAspectRatio="none">
        <line x1="${(px - 2.2).toFixed(2)}" y1="${py.toFixed(2)}" x2="${(px + 2.2).toFixed(2)}" y2="${py.toFixed(2)}" stroke="${color}" stroke-width="0.7" vector-effect="non-scaling-stroke"/>
        <line x1="${px.toFixed(2)}" y1="${(py - 2.2).toFixed(2)}" x2="${px.toFixed(2)}" y2="${(py + 2.2).toFixed(2)}" stroke="${color}" stroke-width="0.7" vector-effect="non-scaling-stroke"/>
        <circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="1.1" fill="${color}"/></svg>`;
}

function pane(context, side, viewRecord) {
    const poly = side === 'left' ? midState.leftPoly : midState.rightPoly;
    const anchor = side === 'left' ? midState.leftAnchor : midState.rightAnchor;
    const color = side === 'left' ? '#ff9d3c' : '#7fdcff';
    const active = midState.side === side;
    return `<div class="mid-pane ${active ? 'active' : ''}" data-mid-pane="${side}">
        <header><b>${esc(viewRecord.title)}</b><small>${side === 'left' ? 'Left capture' : 'Right capture'}</small></header>
        <div class="mid-canvas" data-mid-canvas="${side}">
            ${img(`${context.api()}/views/${viewRecord.id}/preview`, viewRecord.title)}
            ${polygonSvg(poly, color)}${anchorSvg(anchor, '#ffe14d')}
        </div>
        <div class="row tight">
            ${button('Circle ornament', 'mid-mode', `polygon:${side}`)}
            ${button('Mark corner', 'mid-mode', `anchor:${side}`)}
            ${button('Undo', 'mid-undo', side)}${button('Clear', 'mid-clear', side)}
        </div>
        <small class="muted">${poly.length} outline points · ${anchor ? 'corner marked' : 'corner missing'}</small>
    </div>`;
}

export function openMidview(context) {
    const views = context.state.card.views;
    if (views.length < 2) {
        notice('Import both windowsill captures before synthesizing a middle view.', true);
        return;
    }
    midState.open = true;
    midState.busy = false;
    midState.preview = null;
    if (!midState.leftView) {
        midState.leftView = views[0].id;
        midState.rightView = views[Math.min(1, views.length - 1)].id;
    }
    renderMidview(context);
}

function chosen(context, id) {
    return context.state.card.views.find(v => v.id === id) || context.state.card.views[0];
}

function renderMidview(context) {
    const left = chosen(context, midState.leftView);
    const right = chosen(context, midState.rightView);
    midState.leftView = left.id;
    midState.rightView = right.id;
    const choices = context.state.card.views.map(v => [v.id, v.title]);
    const p = midState.preview;
    modal('Synthesize middle view', `
        <form data-form="midview" class="midview-form">
            <div class="grid-fields">
                ${select('Left capture', 'leftView', choices, left.id)}
                ${select('Right capture', 'rightView', choices, right.id)}
            </div>
            <div class="mid-panes">
                ${pane(context, 'left', left)}${pane(context, 'right', right)}
            </div>
            <p class="muted">Circle the ornament once on each capture, then click the same clear
            sill corner on both. Gaps one capture saw are filled from that capture; anything both
            captured behind the ornament stays transparent (shown as a checkerboard).</p>
            ${field('Middle position (0 = left, 1 = right)', 'mid', midState.mid, 'number', 'step="0.01" min="0.001" max="0.999"')}
            <div class="row">
                ${button('Preview middle view', 'mid-preview', '', 'orange')}
                ${p ? button('Add to angle list', 'mid-commit', '', 'orange') : ''}
            </div>
            ${p ? previewBlock(p) : ''}
        </form>`);
    bindCanvas(context);
}

function previewBlock(p) {
    const r = p.report;
    const metrics = [
        ['Background shift', `${r.background_shift_px[0]} / ${r.background_shift_px[1]} px`],
        ['Ornament shift', `${r.foreground_shift_px[0]} / ${r.foreground_shift_px[1]} px (${r.foreground_shift_method})`],
        ['Filled from right', r.background_from_right_pixels.toLocaleString() + ' px'],
        ['Transparent', r.transparent_pixels.toLocaleString() + ` px (${(r.transparent_fraction * 100).toFixed(2)}%)`],
    ];
    return `<section class="mid-result">
        <h3>Preview — checkerboard marks transparent gaps</h3>
        <div class="mid-preview-grid">
            <figure>${img(p.images.middle, 'Checkerboard middle preview')}<figcaption>Checkerboard proof</figcaption></figure>
            <figure>${img(p.images.markers, 'RGBA middle with markers')}<figcaption>Retained RGBA + corner marks</figcaption></figure>
        </div>
        ${metricsRow(metrics)}
        ${r.warnings?.length ? `<ul class="mid-warnings">${r.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}
        ${readout({...r, synthesis_fingerprint: r.synthesis_fingerprint}, 'Synthesis report')}
    </section>`;
}

function metricsRow(rows) {
    return `<div class="metrics">${rows.map(([label, value]) =>
        `<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
}

export function bindMidForm(context) {
    const form = document.querySelector('form[data-form="midview"]');
    if (!form) return;
    form.addEventListener('submit', event => event.preventDefault());
    const midInput = form.elements.mid;
    if (midInput) midInput.addEventListener('change', () => {
        const value = Number(midInput.value);
        if (Number.isFinite(value) && value > 0 && value < 1) {
            midState.mid = value;
            midState.preview = null;
        }
    });
    const syncSelect = name => {
        const input = form.elements[name];
        if (input) input.addEventListener('change', () => {
            if (midState[name] !== input.value) {
                // Marks belong to the previously chosen capture; start fresh.
                midState[name] = input.value;
                midState.leftPoly = [];
                midState.rightPoly = [];
                midState.leftAnchor = null;
                midState.rightAnchor = null;
                midState.preview = null;
            }
            renderMidview(context);
        });
    };
    syncSelect('leftView');
    syncSelect('rightView');
}

function bindCanvas(context) {
    document.querySelectorAll('[data-mid-canvas]').forEach(canvas => {
        canvas.addEventListener('click', event => {
            const side = canvas.dataset.midCanvas;
            midState.side = side;
            const rect = canvas.getBoundingClientRect();
            const point = [
                Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
                Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
            ];
            if (midState.mode === 'anchor') {
                setAnchor(point);
            } else {
                const poly = currentPoly();
                if (poly.length >= 400) {
                    notice('The outline already has the maximum number of points.', true);
                    return;
                }
                setPoly([...poly, point]);
            }
            renderMidview(context);
        });
    });
    bindMidForm(context);
}

export async function handleMidAction(action, value, context) {
    if (action === 'mid-mode') {
        const [mode, side] = value.split(':');
        midState.mode = mode;
        midState.side = side;
        renderMidview(context);
        return;
    }
    if (action === 'mid-undo') {
        midState.side = value;
        if (midState.mode === 'anchor') setAnchor(null);
        else setPoly(currentPoly().slice(0, -1));
        midState.preview = null;
        renderMidview(context);
        return;
    }
    if (action === 'mid-clear') {
        midState.side = value;
        setPoly([]);
        setAnchor(null);
        midState.preview = null;
        renderMidview(context);
        return;
    }
    if (action === 'mid-preview') {
        await runPreview(context);
        return;
    }
    if (action === 'mid-commit') {
        await commitPreview(context);
    }
}

function buildSynthesis() {
    return {
        left_view: midState.leftView,
        right_view: midState.rightView,
        left_polygon: midState.leftPoly,
        right_polygon: midState.rightPoly,
        anchors: [midState.leftAnchor, midState.rightAnchor],
        mid: Number(midState.mid) || 0.5,
    };
}

async function runPreview(context) {
    if (midState.leftView === midState.rightView) {
        notice('Choose two different captures.', true);
        return;
    }
    if (midState.leftPoly.length < 3 || midState.rightPoly.length < 3) {
        notice('Circle the ornament with at least three points on each capture.', true);
        return;
    }
    if (!midState.leftAnchor || !midState.rightAnchor) {
        notice('Mark the same clear sill corner on both captures.', true);
        return;
    }
    midState.busy = true;
    renderMidview(context);
    try {
        const result = await ask(`${context.api()}/midview/preview`, 'POST', {synthesis: buildSynthesis()});
        midState.preview = result;
        notice('Middle view preview ready. Check the transparent gaps before adding it.');
    } catch (error) {
        notice(error.message, true);
    } finally {
        midState.busy = false;
        renderMidview(context);
    }
}

async function commitPreview(context) {
    if (!midState.preview) return;
    const synthesis = {
        ...buildSynthesis(),
        fingerprint: midState.preview.report.synthesis_fingerprint,
    };
    const title = `Middle ${(synthesis.mid * 100).toFixed(0)}%`;
    try {
        const result = await ask(`${context.api()}/midview/commit`, 'POST', {
            synthesis, version: context.cardVersion(), title,
        });
        notice(`Added "${result.view.title}" at ${result.view.position}°; the two originals remain in place.`);
        midState.open = false;
        midState.preview = null;
        document.getElementById('dialog').close();
        await context.openCard(context.state.card.id);
    } catch (error) {
        notice(error.message, true);
    }
}
