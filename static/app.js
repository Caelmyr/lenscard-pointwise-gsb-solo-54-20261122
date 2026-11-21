import {
    ask, button, byId, check, closeDialog, dialogForm, esc, field, format, group, img, link,
    metrics, modal, notice, numbers, optionRows, readout, select, submit, textarea, titleWords, values,
} from './ui.js';
import {physicalActions, physicalForms} from './metrology.js';
import {handleMidAction, openMidview} from './midview.js';

const state = {
    library: [], samples: [], tools: {}, card: null, proofs: [], measurements: [], events: [],
    draftPreview: null, retained: null, lastTool: null, reading: null, stage: 'original',
    view: null, point: [0.5, 0.5], tab: 'compose', angle: 0, stageMods: {}, sequence: 0, archive: null,
};
const api = () => `/api/cards/${state.card.id}`;
const view = () => state.card?.views.find(row => row.id === state.view);
const label = key => state.tools[key] || titleWords(key);
const activeProof = () => state.retained || state.proofs.find(item => item.id === state.card.approval?.proof_id) || state.proofs[0];
const toolChoices = () => optionRows(state.tools);
const resetResult = () => { state.lastTool = null; state.reading = null; };
const resetDraft = () => { state.draftPreview = null; state.stageMods = {}; resetResult(); };
const cardVersion = () => state.card.version;
const angles = () => state.card.config.lens.latitude;
const simpleFields = (rows) => group(rows.map(([label,name,value,options='step="0.001"']) => field(label,name,value,'number',options)));
const viewChoices = () => state.card.views.map(v => [v.id, v.title]);

async function library() {
    const [available, samples] = await Promise.all([ask('/api/library'), ask('/api/samples')]);
    state.library = available.cards;
    state.samples = samples.samples;
    state.tools = available.tools;
    state.card = null;
    redraw();
}

async function openCard(id, {keep = false} = {}) {
    const last = state.card?.id;
    const result = await ask(`/api/cards/${id}`);
    Object.assign(state, result);
    if (last !== id) {
        state.view = null;
        state.stage = 'original';
        state.retained = null;
        state.tab = 'compose';
        state.angle = 0;
    }
    if (!view()) state.view = state.card.views[0]?.id || null;
    if (!keep) resetDraft();
    redraw();
}
function renderLibrary() {
    return `<div class="library-heading"><h1>Print card library</h1>${button('Check or import a package','packages')}</div>
        <div class="library-list">${state.library.map(card => button(`${card.title} · ${card.view_count} views`, 'card', card.id)).join('') || '<p class="muted">Your saved cards will appear here.</p>'}</div>
        <h2>Optical samples</h2><div class="sample-grid">${state.samples.map(sample => `<article class="sample-card">
            ${img(sample.cover, sample.subtitle)}<div class="card-copy"><h3>${esc(sample.title)}</h3>
            <p>${esc(sample.description)}</p><div class="row">${button(sample.saved_card ? 'Open sample' : 'Load sample', 'sample', sample.id, 'orange')}<small>${sample.view_count} synthetic views</small></div></div></article>`).join('')}</div>`;
}
function renderSidebar() {
    const card = state.card;
    return `<aside class="side" aria-label="Card originals">
        <h1>${esc(card.title)}</h1><div class="stamp">Draft ${card.version}${card.synthetic ? ' · Synthetic' : ''}</div>
        <div class="viewlist">${card.views.map(item => `<button type="button" data-action="choose-view" data-value="${item.id}" class="${item.id === state.view ? 'selection' : ''}">
            ${img(`${api()}/views/${item.id}/preview`, '')}<b>${esc(item.title)}</b><small>${format(item.position)}° · ${item.enabled ? 'Active' : 'Off'}</small></button>`).join('')}</div>
        <div class="side-buttons">${button('Add view', 'add-view', '', 'orange')}${button('Synthesize middle view', 'midview-open', '', 'orange')}${view() ? button('Edit view', 'edit-view') : ''}</div>
        ${view() ? `<hr><div class="side-buttons">${link('Original', `${api()}/views/${view().id}/original`)}${button('Source info', 'source-info')}</div>` : ''}
        <hr><div class="side-buttons">${link('View set ZIP', `${api()}/export-views`)}</div>
    </aside>`;
}
function stageImage() {
    if (state.stage === 'original') return view() ? `${api()}/views/${view().id}/preview` : null;
    if (state.stage === 'angular') return state.stageMods.angular || state.draftPreview?.images.angles;
    if (state.stage === 'print') return state.draftPreview?.images['print-preview'] || state.stageMods.print;
    return state.stageMods.inspect;
}
function renderStage() {
    const source = stageImage();
    const stages = [['original','Original'], ['angular','Observed angle'], ['print','Print raster'], ['inspect','Inspection']];
    const mode = state.stage === 'original' ? 'Source image' : state.stage === 'angular' ? 'Optical model' : titleWords(state.stage);
    return `<section class="stage" aria-label="Image stage">
        <div class="stagebar"><div role="group" aria-label="Image representation">
        ${stages.map(([key,text])=>`<button type="button" data-action="stage" data-value="${key}" aria-pressed="${state.stage===key}">${text}</button>`).join('')}</div>
        <span class="stamp">${esc(mode)}</span></div>
        ${source ? `<div class="stage-canvas">${img(source, mode, 'stage-image')}${state.stage === 'original' ?
          `<span class="probe-cross" style="left:${state.point[0]*100}%;top:${state.point[1]*100}%" aria-hidden="true"></span>` : ''}</div>` :
          '<div class="empty">Select an original or build a card preview.</div>'}
        <div class="stage-control"><label>Viewing angle<input id="angle-slider" type="range" min="${-angles()}" max="${angles()}" step=".2" value="${state.angle}"><output id="angle-value">${format(state.angle)}°</output></label></div>
    </section>`;
}
function renderComposition() {
    const {lens, sheet, framing, sampling} = state.card.config;
    return `<form class="controls" data-form="settings">
        ${field('Card title','title',state.card.title)}
        <h2>Lens and stock</h2>
        ${simpleFields([
            ['Pitch (LPI)','lpi',lens.lpi],['Phase (cycles)','phase',lens.phase],
            ['Side drift (mm/mm)','shear',lens.shear],['Viewing half-cone (°)','latitude',lens.latitude],
            ['Width (mm)','width',sheet.width_mm],['Height (mm)','height',sheet.height_mm],
            ['Printer DPI','dpi',sheet.dpi,'step="1"'],['Aperture ratio','aperture',sampling.aperture]])}
        ${group([select('View order','order',[['forward','Forward'],['reverse','Reverse']],lens.order),
                  select('Print side','mount',[['face','Observer front'],['reverse','Back mount']],sheet.mount),
                  select('Paper','paper',[['white','Bright white'],['warm','Warm'],['matte','Matte']],sheet.paper),
                  select('Sampling','method',[['solid','Device dot center'],['coverage','Area coverage']],sampling.method)])}
        <h2>Source framing</h2>
        ${group([select('Canvas','frame_mode',[['center','Reference frame'],['shared','Common coverage']],framing.mode),
                  select('Fit','cover',[['fill','Crop to fill'],['fit','Fit with margins']],framing.cover),
                  select('Reference original','reference',[['','Closest to center'],...viewChoices()],framing.reference||''),
                  check('Align global translation and scale','fit_motion',framing.fit_motion)])}
        ${field('Frame bounds (normalized)','frame',framing.frame.join(', '))}
        ${textarea('Working note','note',state.card.note)}
        <div class="row">${submit('Save draft')} ${button('Build print preview','preview','','orange')}</div>
    </form>${renderPreviewActions()}`;
}
function renderPreviewActions(){
    const result = state.draftPreview;
    if (!result) return '';
    const r = result.report;
    return `<hr><h2>Inspected candidate</h2>${metrics([['View count', r.bands.views.length],
        ['Device period', format(r.quanta.device_dots_per_lens) + ' px'],
        ['Supported raster', format(r.bands.supported_print_fraction*100) + '%'],
        ['Physical card', r.bands.requested_mm.join(' × ') + ' mm']])}
        <div class="row">${button('Keep proof for approval','save-proof','','orange')}${button('Band map','preview-detail','view-map')}${button('Angle response','preview-detail','response')}</div>
        ${readout(r,'Print proof settings')}`;
}
function renderInspection() {
    return `<form class="controls" data-form="inspect">
        ${select('Inspection','tool',toolChoices(),state.lastTool?.tool || 'exposure')}
        ${group([select('Original','view',viewChoices(),state.view),
                  select('Compare against','other',viewChoices(),state.card.views.at(-1)?.id)])}
        ${field('Region (left, top, right, bottom)','region','0,0,1,1')}
        <details><summary>Inspection settings</summary><div class="controls">
          ${simpleFields([['Divider','divider',.5],['Probe radius','radius',7,'step="1"']])}
          ${field('Line start','start','0.04,0.55')}${field('Line end','end','0.96,0.55')}
          ${select('Spectrum direction','direction',[['across','Across'],['down','Down']], 'across')}
          ${select('Paper simulation','soft_method',[['compress','Compress chroma'],['clip','Clip'],['neutral','Neutral']], 'compress')}
          ${textarea('Density patches or lens marks (JSON)','rows','[]',7600)}
        </div></details><div class="row">${submit('Run inspection')}</div>
        </form>${renderReading()}`;
}
function renderReading() {
    const data = state.reading;
    if (!data) return '';
    const report = data.report;
    const pictures = Object.entries(data.images || {});
    const title = label(report.kind || state.lastTool?.tool || 'inspection');
    return `<section class="results"><h2>${esc(title)}</h2>
       ${pictures.length ? img(pictures[0][1], titleWords(pictures[0][0])) : ''}
       <div class="row">${pictures.map(([name])=>button(titleWords(name),'result-view',name)).join('')}
       ${state.lastTool ? button('Keep inspection','keep-reading') : ''}
       ${data.pdf ? link('Download actual-size PDF',data.pdf,'lenscard-target.pdf') : ''}${data.csv ? link('Download measurements CSV',data.csv,'lenscard-readings.csv') : ''}</div>
       ${readout(report)}
       </section>`;
}
function renderRecords() {
    const current = state.card.approval;
    return `<div class="records"><h2>Print proofs</h2>
      ${state.proofs.map(proof => `<article><b>${esc(proof.title)}</b><small>Draft ${proof.base_version} · ${current?.proof_id === proof.id ? 'Current approved print' : esc(proof.state)}</small>
      <div class="row">${button('Open proof','open-proof',proof.id)}</div></article>`).join('') || '<p class="muted">No retained print proofs.</p>'}
      ${current ? `<h2>Approved print</h2><div class="row">${link('Actual-size PDF', `${api()}/approved/pdf?download=1`)}
      ${link('Native PNG', `${api()}/approved/plate?download=1`)}
      ${link('Print ZIP', `${api()}/export-approved?originals=1`)}${button('Verify print','verify-proof')}</div>` : ''}
      <hr><h2>Retained inspections</h2>${state.measurements.map(x => `<article><b>${esc(x.title)}</b><small>${esc(label(x.kind))}</small>
      <div class="row">${button('Read inspection','open-reading',x.id)}${link('Evidence ZIP', `${api()}/measurements/${x.id}/export`)}</div></article>`).join('') || '<p class="muted">No retained inspections.</p>'}
      ${readout(state.events,'Card history')}</div>`;
}
function renderMetrology() {
    return `<h2>Lens and paper measurements</h2>
      <div class="controls">${button('Pitch target','pitch-target')}${button('View isolation target','angle-target')}
      ${button('Fit marked lens lines','lens-fit-dialog')}${button('Measure angle photographs','measure-optics')}
      ${button('Rectify camera photograph','rectify')}${button('Plan angle stops','schedule')}${button('Review camera timing','review-sweep')}${button('Check angle sources','source-preflight')}
      ${activeProof() ? `<h2>Frozen raster / ${esc(activeProof().title)}</h2>
        ${button('Inspect device pixels','device-box')}${button('Check measured print size','physical-size')}
        ${button('Physical ruler proof','ruler-proof')}${button('Measure scanned stripe pitch','scan-pitch')}
        ${button('Record print lineage','lineage')}${button('Compare two raster versions','version-compare')}
        ${button('Scan device band coverage','device-audit')}` : ''}</div>${renderReading()}`;
}
function renderControls() {
    const tabs = [['compose','Compose'],['inspect','Inspect'],['records','Records'],['physical','Metrology']];
    const content = {compose: renderComposition, inspect: renderInspection, records: renderRecords, physical: renderMetrology};
    return `<aside class="side" aria-label="Card controls"><div class="control-tabs" role="tablist" aria-label="Workflow">
     ${tabs.map(([key,text])=>`<button type="button" role="tab" aria-selected="${key===state.tab}" data-action="tab" data-value="${key}">${text}</button>`).join('')}</div>
     ${content[state.tab]()}</aside>`;
}
function redraw() {
    const app = byId('application');
    app.className = state.card ? 'workspace' : 'library';
    app.innerHTML = state.card ? renderSidebar()+renderStage()+renderControls() : renderLibrary();
}
async function showAngle(value) {
    if ((state.card?.views.filter(item=>item.enabled).length ?? 0) < 2) return;
    const serial = ++state.sequence;
    const result = await ask(`${api()}/inspect/view-angle`, 'POST', {angle:value});
    if (serial !== state.sequence) return;
    state.angle = value;
    state.stage = 'angular';
    state.stageMods.angular = result.images.angle;
    redraw();
}
function bodyForTool(tool, v) {
    const bounds = numbers(v.region);
    const body = {view:v.view,other:v.other,region:bounds,divider:Number(v.divider),
                  radius:Number(v.radius),start:numbers(v.start),end:numbers(v.end),point:state.point,
                  direction:v.direction,angle:state.angle};
    if (tool === 'soft-proof') body.settings = {method:v.soft_method};
    if (tool === 'density') body.patches = JSON.parse(v.rows || '[]');
    if (tool === 'lens-fit') body.ticks = JSON.parse(v.rows || '[]');
    return body;
}
function displayReading(data) {
    state.reading = data;
    state.stageMods.inspect = Object.values(data.images || {})[0];
    state.stage = state.stageMods.inspect ? 'inspect' : state.stage;
    redraw();
}
function proofModal(proof) {
    const root = `${api()}/proofs/${proof.id}`;
    const approved = state.card.approval?.proof_id === proof.id;
    state.retained = proof;
    modal(proof.title, `${img(`${root}/asset/print-preview`, 'Kept print proof')}
      ${metrics([['Draft version',proof.base_version],['Print status',approved ? 'Current approved' : proof.state]])}
      <div class="row">${button('Show raster on stage','show-proof-raster',proof.id)}
      ${!approved ? button('Approve this print','approve-proof',proof.id,'orange') : link('Actual-size PDF',`${api()}/approved/pdf?download=1`)}</div>
      ${readout(proof.snapshot,'Frozen inputs and geometry')}`);
}
const actions = {
    card: id => openCard(id),
    sample: async key => {const {card} = await ask(`/api/samples/${key}`,'POST'); await openCard(card.id);},
    'midview-open': () => openMidview({state, api, cardVersion, openCard}),
    'choose-view': id => {state.view=id; state.stage='original'; redraw();},
    'close-dialog': closeDialog,
    'add-view': () => dialogForm('Add an original view','add-view',field('Image file','view','','file','accept="image/*,.tif,.tiff" required')+
                        field('View title','title')+field('Observed angle (°)','angle',0,'number','step=".1"'),'Import view'),
    'edit-view': () => dialogForm('Edit view settings','edit-view',field('View title','title',view().title)+
                      simpleFields([['Observed angle (°)','position',view().position],['Exposure EV','exposure_ev',view().exposure_ev]])+
                      check('Use this view','enabled',view().enabled),'Save view changes'),
    'source-info': async () => {const {result} = await ask(`${api()}/views/${view().id}/information`); modal('Original source',readout(result));},
    tab: value => {state.tab=value; redraw();},
    stage: async value => {if (value==='angular' && !state.stageMods.angular && !state.draftPreview) await showAngle(state.angle); else {state.stage=value; redraw();}},
    preview: async () => {
        state.draftPreview = await ask(`${api()}/preview`,'POST',{angle:state.angle});
        state.stage = 'print';
        redraw();
    },
    'preview-detail': key => modal(titleWords(key), img(state.draftPreview.images[key],titleWords(key))),
    'save-proof': () => dialogForm('Keep print proof','save-proof',field('Proof title','title',`Draft ${cardVersion()}`),'Keep proof'),
    'open-proof': async id => {const {proof}=await ask(`${api()}/proofs/${id}`); proofModal(proof);},
    'approve-proof': id => dialogForm('Approve fixed print','approve-proof',field('Proof ID','id',id,'hidden')+
                       textarea('Approval note','note')+ '<p class="muted">Approved files stay bound to these exact pixels and settings.</p>', 'Approve print'),
    'show-proof-raster': id => {closeDialog();state.stageMods.print=`${api()}/proofs/${id}/asset/print-preview`;
                         state.draftPreview=null;state.stage='print';redraw();},
    'verify-proof': async () => modal('Approved print integrity',readout(await ask(`${api()}/approved-integrity`),'Verification')),
    'keep-reading': () => dialogForm('Retain inspection','keep-reading',field('Inspection title','title',label(state.lastTool.tool)),'Keep inspection'),
    'open-reading': async id => {
        const {measurement:m}=await ask(`${api()}/measurements/${id}`);
        const pictures=Object.keys(m.assets).filter(key => m.assets[key].mime==='image/png').map(key =>
            img(`${api()}/measurements/${id}/asset/${key}`,titleWords(key))).join('');
        const csv=m.assets.csv?link('Download measurement CSV',`${api()}/measurements/${id}/asset/csv?download=1`):'';
        modal(m.title,pictures+csv+readout(m.report));
    },
    'result-view': key => {state.stageMods.inspect=state.reading.images[key];state.stage='inspect';redraw();},
};
const forms = {
    new: async (v) => {const {card}=await ask('/api/cards','POST',{title:v.title,note:v.note}); closeDialog(); await openCard(card.id);},
    'add-view': async (v,node) => {const data=new FormData(node);data.set('version',cardVersion());await ask(`${api()}/views`,'POST',data);closeDialog();await openCard(state.card.id);},
    'edit-view': async v => {await ask(`${api()}/views/${view().id}`,'PUT',{version:cardVersion(),settings:{
        title:v.title,position:Number(v.position),exposure_ev:Number(v.exposure_ev),enabled:v.enabled==='on'}});closeDialog();await openCard(state.card.id);},
    settings: async v => {
        const config={
            lens:{lpi:Number(v.lpi),phase:Number(v.phase),shear:Number(v.shear),latitude:Number(v.latitude),order:v.order},
            sheet:{width_mm:Number(v.width),height_mm:Number(v.height),dpi:Number(v.dpi),mount:v.mount,paper:v.paper},
            framing:{mode:v.frame_mode,cover:v.cover,fit_motion:v.fit_motion==='on',reference:v.reference||null,frame:numbers(v.frame)},
            sampling:{method:v.method,aperture:Number(v.aperture)},
        };
        await ask(api(),'PUT',{version:cardVersion(),title:v.title,note:v.note,config});
        await openCard(state.card.id);notice('Draft settings saved.');
    },
    inspect: async v => {
        const payload=bodyForTool(v.tool,v);
        const result=await ask(`${api()}/inspect/${v.tool}`,'POST',payload);
        state.lastTool={tool:v.tool,payload};displayReading(result);
    },
    'save-proof': async v => {
        const preview=state.draftPreview.report;
        const {proof}=await ask(`${api()}/proofs`,'POST',{title:v.title,version:preview.draft_version,candidate_sha:preview.candidate_sha});
        await openCard(state.card.id,{keep:true});state.tab='records';redraw();proofModal(proof);
    },
    'approve-proof': async v => {
        const proof=state.retained;
        await ask(`${api()}/proofs/${proof.id}/approve`,'PUT',{fingerprint:proof.fingerprint,note:v.note});
        closeDialog();await openCard(state.card.id,{keep:true});notice('Print approved.');
    },
    'keep-reading': async v => {
        const last=state.lastTool;
        await ask(`${api()}/retain/${last.tool}`,'POST',{title:v.title,settings:last.payload,version:state.reading.report.draft_version});
        closeDialog();await openCard(state.card.id,{keep:true});notice('Inspection retained.');
    },
};
const context={state,api,activeProof,library,openCard,displayReading,cardVersion,redraw};
Object.assign(actions,physicalActions(context));
Object.assign(forms,physicalForms(context));
async function handle(node, fn) {
    if (!fn || node?.disabled) return;
    if (node) node.disabled=true;
    try {await fn();}
    catch (error) {notice(error.message || 'Could not complete the operation.',true);}
    finally {if(node) node.disabled=false;}
}
document.addEventListener('click', event=>{
    const node=event.target.closest('[data-action]');
    if (node) {
        const action = node.dataset.action;
        if (action.startsWith('mid-')) {
            handle(node, ()=>handleMidAction(action, node.dataset.value, {state, api, cardVersion, openCard}));
            return;
        }
        handle(node, ()=>actions[node.dataset.action]?.(node.dataset.value));
    }
    if (event.target.id==='stage-image' && state.stage==='original') {
        const region=event.target.getBoundingClientRect();
        state.point=[(event.clientX-region.left)/region.width,(event.clientY-region.top)/region.height].map(x=>Math.max(0,Math.min(1,x)));
        const cursor=document.querySelector('.probe-cross');
        if (cursor) { cursor.style.left=state.point[0]*100+'%'; cursor.style.top=state.point[1]*100+'%'; }
    }
});
document.addEventListener('submit',event=>{
    const node=event.target.closest('form[data-form]');
    if(!node) return;event.preventDefault();const v=values(node);
    handle(event.submitter,()=>forms[node.dataset.form]?.(v,node));
});
document.addEventListener('input',event=>{
    if(event.target.id==='angle-slider') {const output=byId('angle-value');if(output) output.textContent=format(Number(event.target.value))+'°';}
});
document.addEventListener('change',event=>{
    if(event.target.id==='angle-slider') handle(null,()=>showAngle(Number(event.target.value)));
});
byId('new-button').addEventListener('click',()=>dialogForm('New print card','new',field('Card title','title')+textarea('Working note','note'),'Create card'));
byId('library-button').addEventListener('click',()=>handle(null,library));
library().catch(error=>notice(error.message,true));
