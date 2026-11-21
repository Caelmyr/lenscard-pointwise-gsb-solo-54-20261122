import {
    ask, button, closeDialog, dialogForm, esc, field, format, group, img, link,
    metrics, modal, notice, numbers, readout, select, submit, textarea, titleWords,
} from './ui.js';
const box = rows => group(rows.map(([label,key,value,step='.001'])=>field(label,key,value,'number',`step="${step}"`)));
const file = (label, key='scan') => field(label,key,'','file','accept="image/*,.tif,.tiff" required');
const packaged = () => field('LensCard ZIP','archive','','file','accept="application/zip,.zip" required');
const routeFor = (context, tool) => `${context.api()}/proofs/${context.activeProof().id}/${tool}`;
const proofChoices = ctx => ctx.state.proofs.map(row=>[row.id,row.title]);
const deliver = (ctx,result) => {
    closeDialog(); ctx.state.lastTool=null; ctx.displayReading(result);
};
function uploadSettings(node, values) {
    const body = new FormData(node);
    body.set('settings',JSON.stringify(values));
    return body;
}
function measuredForm(ctx, support) {
    const active = ctx.state.card.views.filter(item=>item.enabled);
    const rows = Array.from({length:Math.min(active.length,7)},(_,i)=>support?.settings?.[i] ||
        {angle:active[i].position,expected:i+1,frame:[.13,.17,.87,.85],name:active[i].title});
    dialogForm('Read angle photographs','measure-optics',
        `${field('Angle photographs (in settings order)','capture','','file','accept="image/*,.tif,.tiff" multiple required')}
        ${textarea('Angle, patch and frame coordinates as JSON','settings',JSON.stringify(rows,null,2),7800)}
        ${select('Tie to retained proof','proof',[['','Current draft'],...proofChoices(ctx)],'')}
        ${field('Retain as (optional)','title')}`,'Read test photos');
}
function compareModal(ctx) {
    const options=proofChoices(ctx);
    if(options.length<2) throw Error('Retain two proofs before comparing them.');
    dialogForm('Compare frozen print rasters','compare-frozen',
        select('Proof A','first',options,options.at(0)[0])+select('Proof B','other',options,options.at(1)[0])+
        box([['Split position','split',.5],['Difference cutoff','cutoff',.04]]),'Compare recorded rasters');
}
export function physicalActions(ctx) {
    return {
        packages: () => dialogForm('Check or import a package','check-package',packaged(),'Check package'),
        'import-package': () => dialogForm('Create a draft from verified views','import-package',packaged(),'Import same view set'),
        'pitch-target': () => dialogForm('Print a pitch test target','pitch-target',
            box([['Number of rows','increments',7,1],['Pitch change per row (LPI)','delta_lpi',.2]]),'Build pitch target'),
        'angle-target': async () => deliver(ctx,await ask(`${ctx.api()}/angle-target`)),
        'lens-fit-dialog': async () => {
            const demo = ctx.state.card.synthetic ? await ask(`${ctx.api()}/synthetic-reading`) : null;
            dialogForm('Fit marked lens lines','fit-lens',
                textarea('X (mm), Y (mm) and integer stripe index as JSON','ticks',JSON.stringify(demo?.ticks || [],null,2),7800),
                'Fit global lens');
        },
        'apply-lens-fit': async () => {
            const row=ctx.state.lensProposal;
            const answer=await ask(`${ctx.api()}/lens-fit/apply`,'POST',{ticks:row.source_marks,version:row.draft_version,fit_sha:row.fit_sha});
            closeDialog();await ctx.openCard(answer.card.id);notice('Global lens calibration applied to the draft.');
        },
        'measure-optics': async () => {
            const demo = ctx.state.card.synthetic ? await ask(`${ctx.api()}/synthetic-reading`) : null;
            if(demo) {
                modal('Synthetic example photographs',`<div class="row">${demo.settings.map((row,i)=>
                link(row.name,`${ctx.api()}/synthetic-reading/${i+1}`)).join('')}</div>
                <p class="muted">Use these files to exercise the patch measurement without a camera.</p>
                ${button('Enter photograph readings','measured-form','','orange')}`);
                ctx.state.measureHints=demo;
            } else measuredForm(ctx,null);
        },
        'measured-form': () => measuredForm(ctx,ctx.state.measureHints),
        'rectify': () => dialogForm('Rectify a photographed print','rectify',
            file('Original photograph')+field('Four normalized corners (JSON)','corners','[[0.08,0.08],[0.92,0.08],[0.92,0.92],[0.08,0.92]]')+
            box([['Requested width (mm)','width',ctx.state.card.config.sheet.width_mm],
                ['Requested height (mm)','height',ctx.state.card.config.sheet.height_mm],['Pixels per millimetre','resolution',9]]),'Rectify photograph'),
        schedule: () => dialogForm('Plan camera angle stops','schedule',
            box([['Observation points','steps',11,1],['Half sweep (°)','angle_limit',ctx.state.card.config.lens.latitude*.88],
                ['Hold (s)','dwell',.55],['Turn (°/s)','travel',13],['Target main-band fraction','dominance',.68]])+
            select('Direction','side',[['left_to_right','Left to right'],['right_to_left','Right to left']],'left_to_right'),'Build capture plan'),
        'source-preflight': async () => deliver(ctx,await ask(`${ctx.api()}/preflight`)),
        'review-sweep': () => dialogForm('Review actual camera timing','review-sweep',
            textarea('Prior capture settings (JSON)','settings',JSON.stringify(ctx.state.scheduleSettings || {},null,2),6000)+
            field('Camera timestamps in order (seconds)','timings'),'Compare to planned stops'),
        'device-box': () => dialogForm('Inspect a device raster crop','device-box', box([
            ['Paper X (mm)','x_mm',ctx.state.card.config.sheet.width_mm/2],
            ['Paper Y (mm)','y_mm',ctx.state.card.config.sheet.height_mm/2],
            ['Crop width (mm)','width_mm',12],['Crop height (mm)','height_mm',8],
            ['Visual zoom','enlarge',4,1],
        ]),'Read device window'),
        'physical-size': () => dialogForm('Check measured physical print','physical-size',
            box([['Width read (mm)','width',ctx.state.card.config.sheet.width_mm],['Height read (mm)','height',ctx.state.card.config.sheet.height_mm]])+
            field('Instrument','instrument','hand ruler'),'Check size'),
        'ruler-proof': async () => deliver(ctx,await ask(routeFor(ctx,'size-proof'))),
        'scan-pitch': () => dialogForm('Measure pitch from a photo','scan-pitch',file('Photograph of a lens stripe target')+
            field('Camera scale in pixels/mm','scale',32,'number','step=".1"')+
            field('Target rectangle (normalized)','region','.1,.28,.9,.55'),'Measure photographed stripe pitch'),
        lineage: async () => {
            const result=await ask(routeFor(ctx,'lineage'));
            modal('Print source lineage',metrics([['Source verification',result.report.verified?'Verified':'Review'],
                ['Retained originals',result.report.sources.length]])+readout(result.report));
        },
        'version-compare': () => compareModal(ctx),
        'device-audit': () => dialogForm('Check printed device-band coverage','device-audit',
            box([['Start row','top',0,1],['Number of rows','rows',140,1],
                 ['Lowest allowed view fraction','floor',.07]])+select('View representation','side',
                    [['device','Physical print raster'],['observer','Observer-facing']],'device'),'Scan band coverage'),
    };
}
export function physicalForms(ctx) {
    return {
        'check-package': async (v,node) => {
            const outcome=await ask('/api/packages/check','POST',new FormData(node));
            ctx.state.archive=outcome.report;
            modal('Checked package',`<p class="muted">${outcome.report.verified?'Manifest and member bytes agree.':'Package needs review.'}</p>
                  ${outcome.report.verified && outcome.report.kind==='view-set'?button('Import as a new draft','import-package','','orange'):''}
                  ${readout(outcome.report)}`);
        },
        'import-package': async (v,node) => {
            const body=new FormData(node);body.set('digest',ctx.state.archive?.digest||'');
            const {card}=await ask('/api/packages/import','POST',body);
            closeDialog();await ctx.openCard(card.id);
        },
        'pitch-target': async v => deliver(ctx,await ask(`${ctx.api()}/pitch-target`,'POST',
            {increments:Number(v.increments),delta_lpi:Number(v.delta_lpi)})),
        'fit-lens': async v => {
            const result=await ask(`${ctx.api()}/inspect/lens-fit`,'POST',{ticks:JSON.parse(v.ticks)});
            ctx.state.lensProposal=result.report;
            modal('Lens fit residuals',`${img(result.images.residuals,'Observed and fitted lens marks')}${readout(result.report)}
                  ${result.report.fit_status==='supported'?button('Apply measured global lens','apply-lens-fit','','orange'):''}`);
        },
        'measure-optics': async (v,node) => {
            const body=uploadSettings(node,{captures:JSON.parse(v.settings),proof_id:v.proof||null,retain:Boolean(v.title),title:v.title});
            deliver(ctx,await ask(`${ctx.api()}/measured-optics`,'POST',body));
        },
        rectify: async (v,node) => {
            const body=uploadSettings(node,{corners:JSON.parse(v.corners),
                board:{width_mm:Number(v.width),height_mm:Number(v.height),resolution:Number(v.resolution)}});
            deliver(ctx,await ask(`${ctx.api()}/rectify-photo`,'POST',body));
        },
        schedule: async v => {
            const config={steps:Number(v.steps),angle_limit:Number(v.angle_limit),dwell:Number(v.dwell),
                          travel:Number(v.travel),dominance:Number(v.dominance),side:v.side};
            ctx.state.scheduleSettings=config;
            deliver(ctx,await ask(`${ctx.api()}/angle-sweep`,'POST',config));
        },
        'review-sweep': async v => deliver(ctx,await ask(`${ctx.api()}/angle-sweep/review`,'POST',{
            settings:JSON.parse(v.settings),timings:numbers(v.timings),
        })),
        'device-box': async v => deliver(ctx,await ask(routeFor(ctx,'raster-region'),'POST',
            Object.fromEntries(Object.keys(v).map(k=>[k,Number(v[k])])))),
        'physical-size': async v => deliver(ctx,await ask(routeFor(ctx,'size-readback'),'POST',{
            width:Number(v.width),height:Number(v.height),instrument:v.instrument,
        })),
        'scan-pitch': async (v,node) => {
            const body=uploadSettings(node,{camera_px_per_mm:Number(v.scale),region:numbers(v.region)});
            deliver(ctx,await ask(routeFor(ctx,'pitch-readback'),'POST',body));
        },
        'compare-frozen': async v => deliver(ctx,await ask(`${ctx.api()}/compare-proofs`,'POST',{
            first:v.first,other:v.other,divider:Number(v.split),cutoff:Number(v.cutoff),
        })),
        'device-audit': async v => deliver(ctx,await ask(routeFor(ctx,'band-audit'),'POST',{
            top:Number(v.top),rows:Number(v.rows),minimum:Number(v.floor),side:v.side,
        })),
    };
}
