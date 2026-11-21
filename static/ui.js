export const byId = key => document.getElementById(key);
export const esc = value => String(value ?? '').replace(/[&<>"']/g, character =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
export const button = (label, action, value = '', css = '') =>
    `<button type="button" class="${esc(css)}" data-action="${esc(action)}" data-value="${esc(value)}">${esc(label)}</button>`;
export const link = (label, url, name = null) =>
    `<a class="button" href="${esc(url)}"${name !== null ? ` download="${esc(name)}"` : ''}>${esc(label)}</a>`;
export const field = (label, name, value = '', type = 'text', options = '') =>
    `<label>${esc(label)}<input type="${esc(type)}" name="${esc(name)}" value="${esc(value)}" ${options}></label>`;
export const textarea = (label, name, value = '', maxlength = 500) =>
    `<label>${esc(label)}<textarea name="${esc(name)}" maxlength="${maxlength}">${esc(value)}</textarea></label>`;
export const select = (label, name, choices, current) =>
    `<label>${esc(label)}<select name="${esc(name)}">${choices.map(([value, display]) =>
        `<option value="${esc(value)}"${String(current) === String(value) ? ' selected' : ''}>${esc(display)}</option>`).join('')}</select></label>`;
export const check = (label, name, checked = false) =>
    `<label style="display:flex;align-items:center"><input type="checkbox" name="${esc(name)}" ${checked ? 'checked' : ''}>${esc(label)}</label>`;
export const group = values => `<div class="grid-fields">${values.join('')}</div>`;
export const readout = (value, title = 'Measurements and settings') =>
    `<details><summary>${esc(title)}</summary><pre class="readout">${esc(JSON.stringify(value, null, 2))}</pre></details>`;
export const metrics = values => `<div class="metrics">${values.map(([label,value]) =>
    `<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
export const format = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
export const img = (src, alt, id = '') => `<img src="${esc(src)}" alt="${esc(alt)}"${id ? ` id="${esc(id)}"` : ''}>`;
export const titleWords = name => String(name).replace(/[-_]/g, ' ').replace(/^./, c => c.toLocaleUpperCase());
export const submit = label => `<button type="submit" class="orange">${esc(label)}</button>`;
export const optionRows = object => Object.entries(object);
export function modal(title, content) {
    const dialog = byId('dialog');
    dialog.innerHTML = `<header><h2>${esc(title)}</h2>${button('Close', 'close-dialog')}</header>${content}`;
    if (!dialog.open) dialog.showModal();
}
export function dialogForm(title, action, body, label, extra = '') {
    modal(title, `<form data-form="${esc(action)}" ${extra}>${body}<div class="row">${submit(label)}</div></form>`);
}
export function closeDialog() {
    const dialog = byId('dialog');
    if (dialog.open) dialog.close();
}
export function notice(message, error = false) {
    const alert = byId('notice');
    alert.hidden = false;
    alert.classList.toggle('error', error);
    alert.textContent = message;
    clearTimeout(notice.timer);
    notice.timer = setTimeout(() => alert.hidden = true, error ? 8500 : 3300);
}
export async function ask(url, method = 'GET', data = undefined) {
    const config = {method};
    if (data instanceof FormData) {
        config.body = data;
    } else if (data !== undefined) {
        config.headers = {'Content-Type':'application/json'};
        config.body = JSON.stringify(data);
    }
    const response = await fetch(url, config);
    const result = await response.json().catch(() => null);
    if (!response.ok) throw Error(result?.error || `Request failed (${response.status}).`);
    return result;
}
export function values(form) {
    return Object.fromEntries(new FormData(form));
}
export function numbers(input) {
    return String(input).split(/[\s,]+/).filter(Boolean).map(Number);
}
