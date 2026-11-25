/* XR17C — deterministic read-only instrument review and compositor safety. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
function get(path){return readFileSync(new URL(path,import.meta.url),'utf8')}
const ctx={window:{}};
for(const src of ['../xr/physical-instrument-bench.js',
 '../xr/instrument-replay.js','../xr/display-path.js'])
 runInNewContext(get(src),ctx);
const bench=ctx.window.PulmoXRInstrumentBench;
const replayApi=ctx.window.PulmoXRInstrumentReplay;
const display=ctx.window.PulmoXRDisplayPath;
function fixture(){
 const samples=Array.from({length:80},(_,i)=>({
  tMs:i*100,cmdInsertionMm:i*2,measInsertionMm:i*2+1.2,
  cmdTipHDeg:5,measTipHDeg:6,cmdTipVDeg:-2,measTipVDeg:-2.5,
  cmdRollDeg:179,measRollDeg:-179,tracking:true
 }));
 return {version:'pulmosimu-instrument-trace/1',
  provenance:{deviceModel:'Synthetic rig for TEST ONLY',
    instrumentId:'TEST-ONLY-001',referenceSensor:'TEST-ONLY-ENCODER',
    operator:'Synthetic QA',measuredAt:'2026-11-25T08:00:00.000Z',
    measurementSource:'external-bench'},
  units:{time:'ms',insertion:'mm',angles:'deg'},samples};
}
test('XR17C measured trace assesses using original XR16F acceptance, without clinical approval',()=>{
 const r=replayApi.create({assess:bench.assess});
 assert.equal(r.snapshot().loaded,false);
 assert.equal(r.snapshot().deviceToScopeCoupled,false);
 const check=r.load(fixture());
 assert.equal(check.ok,true);
 assert.equal(check.total,80);
 assert.equal(check.clinicalValidated,false);
 assert.equal(r.snapshot().engineeringStatus,'BENCH_ENGINEERING_TARGET_MET');
 assert.equal(r.snapshot().measuredMetrics.sampleCount,80);
 assert.equal(r.snapshot().observation.residual.insertionMm,1.2);
 assert.equal(r.snapshot().observation.residual.rollDeg,2);
 assert.equal(r.snapshot().observation.tracking,true);
 assert.equal(r.snapshot().actualLiveInstrumentTracking,false);
 assert.equal(r.snapshot().clinicalValidated,false);
});
test('XR17C plays source measurement timeline independently of procedure engine',()=>{
 const r=replayApi.create({assess:bench.assess});
 r.load(fixture());
 assert.equal(r.play(1000),true);
 assert.equal(r.tick(3300).reviewIndex,23);
 assert.equal(r.snapshot().observation.measured.insertionMm,47.2);
 assert.equal(r.pause(3400).playing,false);
 const pos=r.snapshot().reviewIndex;
 assert.equal(r.step(1).reviewIndex,pos+1);
 assert.equal(r.step(-1).reviewIndex,pos);
 assert.equal(r.seek(99999).reviewIndex,79);
 assert.equal(r.play(5500),true);
 assert.equal(r.tick(5600).reviewIndex,1);
 assert.equal(r.snapshot().deviceToScopeCoupled,false);
 r.clear();
 assert.equal(r.snapshot().loaded,false);
});
test('XR17C rejects unsafe or unreviewed trace before import, preserves previous valid file',()=>{
 const r=replayApi.create({assess:bench.assess});
 r.load(fixture());
 const invalid=fixture();
 invalid.samples[2].cmdTipHDeg=Number.NaN;
 assert.equal(r.load(invalid).ok,false);
 assert.equal(r.snapshot().loaded,true);
 assert.equal(r.snapshot().total,80);
 assert.equal(r.snapshot().rejectedCount,1);
 const noSource=fixture();noSource.provenance.measurementSource='controller-only';
 assert.equal(r.load(noSource).ok,false);
 assert.equal(r.snapshot().clinicalValidated,false);
 assert.equal(r.snapshot().deviceToScopeCoupled,false);
});
test('XR17C display evaluator never steals Three r128 XR framebuffer',()=>{
 const path=display.create({bindingConstructor:function XRWebGLBinding(){}});
 const r=path.inspect({
  session:{enabledFeatures:new Set(['layers']),
   renderState:{baseLayer:{framebuffer:{}}}},
  renderer:{xr:{}},gl:{}
 });
 assert.equal(r.requestedLayers,true);
 assert.equal(r.bindingAvailable,true);
 assert.equal(r.legacyBaseLayer,true);
 assert.equal(r.actualDisplayPath,'SAME_CONTEXT_OFFSCREEN_WEBGL_RENDER_TARGET');
 assert.equal(r.compositorQuadActive,false);
 assert.equal(r.reason,'THREE_R128_BASE_LAYER_OWNS_RENDERER');
 assert.equal(r.safeFallback,true);
 assert.equal(display.create().inspect({session:{}}).reason,
  'SESSION_FEATURES_NOT_REPORTED');
 assert.equal(display.create().snapshot().quadEligibility,'NOT_CHECKED');
 const src=get('../xr/display-path.js');
 assert.equal(src.includes('updateRenderState('),false);
 assert.equal(src.includes('createQuadLayer('),false);
});
test('XR17C source-only integration preserves exact original medical engine',()=>{
 const xr=get('../xr/procedural-immersive.js');
 const html=get('../../index.html');
 const caps=get('../xr/xr-capabilities.js');
 for(const marker of [
  'instrument-replay.js?v=XR17C-READONLY-REPLAY',
  'physical-instrument-bench.js?v=XR17C-MEASUREMENT-REVIEW',
  'display-path.js?v=XR17C-COMPOSITOR-READINESS'
 ])assert.ok(html.includes(marker),marker);
 for(const marker of [
  'const replay=global.PulmoXRInstrumentReplay.create',
  'assess:global.PulmoXRInstrumentBench.assess',
  'replay.load(measured)',
  "action==='trace-open'",
  "action==='trace-play'",
  'replay.tick(time)',
  'XR17CDisplayPath:displayPath.snapshot()',
  'XR17CInstrumentTrace:traceSummary()',
  "id='pulmo-xr-trace-import'",
  'displayPath.inspect({session:rBro.xr.getSession()',
  'scopeScreen.visible=next===\'procedure\'',
  'enterBranch(current.branches[index].index)',
  'getMeshMode()==='+"'patient'",
  "station.set('video',recommended)"
 ])assert.ok(xr.includes(marker),'XR17C missing existing engine or trace boundary: '+marker);
 assert.ok(caps.includes("'hand-tracking','layers'"));
 const src=get('../xr/instrument-replay.js');
 assert.equal(/\\bscope\\.(velocity|deflH|deflV|roll|depth|t)\\b/.test(src),false,
  'Trace reviewer must not manipulate original scope fields');
 assert.equal(src.includes('fetch('),false);
 assert.equal(src.includes('WebSocket('),false);
});
